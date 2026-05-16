"""Hand-annotation tool for chewing window labeling (blind).

Split-screen UI:
  Left  — video frame playback with label HUD
  Right — jaw_open / MAR signal trace, current window highlighted
  Bottom — progress strip (one colored block per window)

Blind mode: engine labels are never shown during annotation.
Only raw signal values (jaw_open_mean, mar_mean) and the video are visible,
which are the same information the annotator would use from the video anyway.

Usage:
    python ml/annotate.py <session_dir> [--all]

    --all   Annotate every window, not just disagreement windows.

Keys:
    c       label as chewing  (auto-advance to next)
    r       label as rest     (auto-advance to next)
    b       label as bad_face (auto-advance to next)
    a       previous window
    d       next window
    Tab     jump to next unlabeled window
    s       save progress
    q       save and quit
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

# ── Layout ──────────────────────────────────────────────────────────────────
VIDEO_W  = 720
VIDEO_H  = 480
SIGNAL_W = 460
STRIP_H  = 52
TOTAL_W  = VIDEO_W + SIGNAL_W          # 1180
TOTAL_H  = VIDEO_H + STRIP_H           # 532

# ── Colors (BGR) ─────────────────────────────────────────────────────────────
C_BG         = (28,  28,  28)
C_CHEWING    = (55,  200,  60)   # green
C_REST       = (55,  200, 220)   # cyan
C_BAD        = (55,   55, 220)   # red-blue
C_UNLABELED  = (85,   85,  85)   # gray
C_CURRENT_HI = (255, 200,  60)   # amber highlight
C_TEXT       = (220, 220, 220)
C_DIM        = (120, 120, 120)
C_JAW        = (55,  170, 230)   # signal line — jaw_open
C_MAR        = (55,  220, 130)   # signal line — MAR

COL_T_START = "t_start"
COL_T_END   = "t_end"
COL_LABEL   = "label"

MAX_FRAME_CACHE = 6   # windows kept in memory at once


def _label_color(label: Optional[str]) -> tuple:
    return {"chewing": C_CHEWING, "rest": C_REST, "bad_face": C_BAD}.get(
        label or "", C_UNLABELED
    )


def _read_csv(path: Path) -> list[dict]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _write_human_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fieldnames = ["t_start", "t_end", "label", "jaw_open_label", "composite_label"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def _find_disagree_windows(jaw_rows: list[dict], comp_rows: list[dict]) -> list[tuple]:
    jaw_by_t  = {float(r[COL_T_START]): r for r in jaw_rows}
    comp_by_t = {float(r[COL_T_START]): r for r in comp_rows}
    return [
        (jrow, comp_by_t.get(t))
        for t, jrow in sorted(jaw_by_t.items())
        if comp_by_t.get(t) and jrow[COL_LABEL] != comp_by_t[t][COL_LABEL]
    ]


def _all_windows(jaw_rows: list[dict], comp_rows: list[dict]) -> list[tuple]:
    jaw_by_t  = {float(r[COL_T_START]): r for r in jaw_rows}
    comp_by_t = {float(r[COL_T_START]): r for r in comp_rows}
    return [(jaw_by_t[t], comp_by_t.get(t)) for t in sorted(jaw_by_t.keys())]


# ── Frame extraction (lazy, with small LRU cache) ────────────────────────────

def _extract_frames(
    cap: cv2.VideoCapture, fps: float, t_start: float, t_end: float
) -> list[np.ndarray]:
    start_frame = int(t_start * fps)
    end_frame   = int(t_end   * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    frames = []
    for _ in range(end_frame - start_frame):
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(cv2.resize(frame, (VIDEO_W, VIDEO_H)))
    if not frames:
        frames.append(np.zeros((VIDEO_H, VIDEO_W, 3), dtype=np.uint8))
    return frames


class FrameCache:
    def __init__(self, cap: cv2.VideoCapture, fps: float, annotated: list[dict]) -> None:
        self._cap = cap
        self._fps = fps
        self._ann = annotated
        self._cache: dict[int, list[np.ndarray]] = {}

    def get(self, idx: int) -> list[np.ndarray]:
        if idx not in self._cache:
            r = self._ann[idx]
            self._cache[idx] = _extract_frames(
                self._cap, self._fps, float(r["t_start"]), float(r["t_end"])
            )
            if len(self._cache) > MAX_FRAME_CACHE:
                # evict the index farthest from current
                farthest = max(self._cache, key=lambda k: abs(k - idx))
                if farthest != idx:
                    del self._cache[farthest]
        return self._cache[idx]

    def prefetch(self, idx: int) -> None:
        """Warm adjacent windows in the background (best-effort)."""
        for ni in (idx - 1, idx + 1):
            if 0 <= ni < len(self._ann) and ni not in self._cache:
                self.get(ni)


# ── Signal data ──────────────────────────────────────────────────────────────

def _load_frame_signals(sdir: Path) -> Optional[dict]:
    p = sdir / "frame_signals_ours.csv"
    if not p.exists():
        return None
    rows = _read_csv(p)
    try:
        t        = np.array([float(r["t_sec"])    for r in rows])
        jaw_open = np.array([float(r["jaw_open"]) for r in rows])
        mar      = np.array([float(r["mar"])      for r in rows])
        return {"t": t, "jaw_open": jaw_open, "mar": mar}
    except (KeyError, ValueError):
        return None


def _norm01(arr: np.ndarray) -> np.ndarray:
    lo, hi = arr.min(), arr.max()
    return np.zeros_like(arr) if hi == lo else (arr - lo) / (hi - lo)


# ── Signal panel ─────────────────────────────────────────────────────────────

def _build_signal_panel(
    signals: Optional[dict],
    annotated: list[dict],
    current_idx: int,
) -> np.ndarray:
    panel = np.full((VIDEO_H, SIGNAL_W, 3), C_BG, dtype=np.uint8)
    h, w = VIDEO_H, SIGNAL_W

    def txt(text: str, x: int, y: int, color=C_TEXT, scale=0.52, thick=1) -> None:
        cv2.putText(panel, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                    scale, color, thick, cv2.LINE_AA)

    r    = annotated[current_idx]
    t_s  = float(r["t_start"])
    t_e  = float(r["t_end"])
    done = sum(1 for a in annotated if a["label"])
    total = len(annotated)

    # ── Header ──
    txt(f"Window {current_idx + 1} / {total}", 10, 26, scale=0.65, thick=2)
    txt(f"t = {t_s:.1f} – {t_e:.1f} s", 10, 48, color=C_DIM, scale=0.48)
    txt(f"labeled: {done}/{total}", w - 115, 26, color=C_DIM, scale=0.46)

    # ── Trace area ──────────────────────────────────────────────────────────
    TRACE_Y0 = 62
    TRACE_Y1 = h - 110
    TH       = TRACE_Y1 - TRACE_Y0
    PAD_L, PAD_R = 8, 8
    TRACE_X0 = PAD_L
    TRACE_X1 = w - PAD_R

    # Background box
    cv2.rectangle(panel, (TRACE_X0, TRACE_Y0), (TRACE_X1, TRACE_Y1), (45, 45, 45), 1)

    if signals is not None:
        t_all = signals["t"]
        t_min, t_max = float(t_all[0]), float(t_all[-1])
        total_dur = max(t_max - t_min, 1.0)
        TW = TRACE_X1 - TRACE_X0

        def t2x(t: float) -> int:
            return int(TRACE_X0 + (t - t_min) / total_dur * TW)

        def v2y(v: float) -> int:
            return int(TRACE_Y1 - v * TH)

        # Faint window boundary lines
        for ann in annotated:
            xs = t2x(float(ann["t_start"]))
            cv2.line(panel, (xs, TRACE_Y0), (xs, TRACE_Y1), (55, 55, 55), 1)

        # Labeled window bands at trace top
        BAND_H = 7
        for ann in annotated:
            if ann["label"]:
                bx0 = t2x(float(ann["t_start"]))
                bx1 = t2x(float(ann["t_end"]))
                cv2.rectangle(panel, (bx0, TRACE_Y0), (bx1, TRACE_Y0 + BAND_H),
                              _label_color(ann["label"]), -1)

        # Current window highlight
        cx0, cx1 = t2x(t_s), t2x(t_e)
        overlay = panel.copy()
        cv2.rectangle(overlay, (cx0, TRACE_Y0), (cx1, TRACE_Y1), (70, 90, 140), -1)
        cv2.addWeighted(overlay, 0.30, panel, 0.70, 0, panel)
        cv2.rectangle(panel, (cx0, TRACE_Y0), (cx1, TRACE_Y1), C_CURRENT_HI, 1)

        # MAR line
        jaw_norm = _norm01(signals["jaw_open"])
        mar_norm = _norm01(signals["mar"])

        pts_mar = np.array(
            [(t2x(float(signals["t"][i])), v2y(float(mar_norm[i])))
             for i in range(len(t_all))],
            dtype=np.int32,
        )
        cv2.polylines(panel, [pts_mar], False, C_MAR, 1, cv2.LINE_AA)

        # jaw_open line (on top, thicker)
        pts_jaw = np.array(
            [(t2x(float(signals["t"][i])), v2y(float(jaw_norm[i])))
             for i in range(len(t_all))],
            dtype=np.int32,
        )
        cv2.polylines(panel, [pts_jaw], False, C_JAW, 2, cv2.LINE_AA)

        # Legend
        LX = TRACE_X0
        LY = TRACE_Y0 - 10
        cv2.line(panel, (LX, LY), (LX + 18, LY), C_JAW, 2)
        txt("jaw_open", LX + 22, LY + 4, color=C_JAW, scale=0.40)
        cv2.line(panel, (LX + 95, LY), (LX + 113, LY), C_MAR, 1)
        txt("MAR", LX + 117, LY + 4, color=C_MAR, scale=0.40)

    else:
        mid_y = (TRACE_Y0 + TRACE_Y1) // 2
        txt("(no frame_signals_ours.csv)", TRACE_X0 + 8, mid_y, color=C_DIM, scale=0.44)

    # ── Window stats ────────────────────────────────────────────────────────
    SY = TRACE_Y1 + 14
    jaw_mean_str = r.get("jaw_open_mean", "")
    mar_mean_str = r.get("mar_mean",      "")
    if jaw_mean_str:
        txt(f"jaw_open mean  {float(jaw_mean_str):.4f}", 10, SY + 18, color=C_DIM, scale=0.46)
    if mar_mean_str:
        txt(f"MAR mean       {float(mar_mean_str):.4f}", 10, SY + 36, color=C_DIM, scale=0.46)

    # ── Controls ────────────────────────────────────────────────────────────
    CY = h - 54
    txt("[c] chewing   [r] rest   [b] bad_face",     10, CY,      color=C_DIM, scale=0.43)
    txt("[a] prev   [d] next   [Tab] next unlabeled", 10, CY + 18, color=C_DIM, scale=0.43)
    txt("[s] save   [q] save & quit",                 10, CY + 36, color=C_DIM, scale=0.43)

    return panel


# ── Progress strip ────────────────────────────────────────────────────────────

def _build_progress_strip(annotated: list[dict], current_idx: int) -> np.ndarray:
    strip = np.full((STRIP_H, TOTAL_W, 3), (18, 18, 18), dtype=np.uint8)
    n = len(annotated)
    if n == 0:
        return strip
    for i, ann in enumerate(annotated):
        x0 = i * TOTAL_W // n
        x1 = (i + 1) * TOTAL_W // n
        color = _label_color(ann["label"] or None)
        cv2.rectangle(strip, (x0 + 1, 6), (x1 - 1, STRIP_H - 6), color, -1)
        if i == current_idx:
            cv2.rectangle(strip, (x0, 5), (x1, STRIP_H - 5), (255, 255, 255), 2)
    # summary text
    done = sum(1 for a in annotated if a["label"])
    cv2.putText(strip, f" {done}/{n} labeled",
                (6, STRIP_H - 10), cv2.FONT_HERSHEY_SIMPLEX,
                0.42, C_TEXT, 1, cv2.LINE_AA)
    return strip


# ── Video HUD ────────────────────────────────────────────────────────────────

def _draw_video_hud(
    frame: np.ndarray,
    idx: int,
    total: int,
    t_start: float,
    t_end: float,
    human_label: Optional[str],
) -> np.ndarray:
    out = frame.copy()
    lcolor = _label_color(human_label)
    lstr   = human_label if human_label else "—"

    # Top bar
    cv2.rectangle(out, (0, 0), (VIDEO_W, 46), (0, 0, 0), -1)
    cv2.putText(out, f"Window {idx + 1}/{total}   t = {t_start:.1f} – {t_end:.1f} s",
                (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, C_TEXT, 1, cv2.LINE_AA)

    # Bottom bar
    cv2.rectangle(out, (0, VIDEO_H - 52), (VIDEO_W, VIDEO_H), (0, 0, 0), -1)
    cv2.putText(out, "Label:", (12, VIDEO_H - 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, C_DIM, 1, cv2.LINE_AA)
    cv2.putText(out, lstr, (88, VIDEO_H - 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.80, lcolor, 2, cv2.LINE_AA)

    # Left edge color bar
    cv2.rectangle(out, (0, 46), (6, VIDEO_H - 52), lcolor, -1)

    return out


# ── Report ───────────────────────────────────────────────────────────────────

def _print_report(annotated: list[dict]) -> None:
    labeled = [r for r in annotated if r.get("label")]
    if not labeled:
        print("[annotate] No completed labels — skipping report.")
        return

    jaw_correct  = sum(1 for r in labeled if r["jaw_open_label"]  == r["label"])
    comp_correct = sum(1 for r in labeled if r["composite_label"] == r["label"])
    n = len(labeled)

    print()
    print("=" * 60)
    print(f"  Disagreement windows annotated : {n}")
    print(f"  jaw_open  matches human GT     : {jaw_correct}/{n}  ({100*jaw_correct/n:.1f}%)")
    print(f"  composite matches human GT     : {comp_correct}/{n}  ({100*comp_correct/n:.1f}%)")
    print("=" * 60)
    print(f"\n  {'t_start':>7}  {'jaw_open':>9}  {'composite':>9}  {'human':>9}")
    print(f"  {'-'*7}  {'-'*9}  {'-'*9}  {'-'*9}")
    for r in labeled:
        jmark = "✓" if r["jaw_open_label"]  == r["label"] else "✗"
        cmark = "✓" if r["composite_label"] == r["label"] else "✗"
        print(
            f"  {float(r['t_start']):>7.1f}  "
            f"{r['jaw_open_label']:>8} {jmark}  "
            f"{r['composite_label']:>8} {cmark}  "
            f"{r['label']:>9}"
        )
    print()


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Blind chewing annotation tool")
    parser.add_argument("session_dir", type=Path)
    parser.add_argument("--all", action="store_true",
                        help="Annotate all windows (default: disagree only)")
    args = parser.parse_args()

    sdir      = args.session_dir.resolve()
    jaw_path  = sdir / "labels_ours_jaw_open.csv"
    comp_path = sdir / "labels_ours.csv"
    out_path  = sdir / "labels_human.csv"

    if not jaw_path.exists():
        sys.exit(f"[annotate] Missing {jaw_path}")
    if not comp_path.exists():
        sys.exit(f"[annotate] Missing {comp_path}")

    videos = sorted(sdir.glob("video_*.mp4"))
    if not videos:
        sys.exit(f"[annotate] No video_*.mp4 found in {sdir}")
    video_path = str(videos[0])

    jaw_rows  = _read_csv(jaw_path)
    comp_rows = _read_csv(comp_path)
    comp_by_t = {float(r[COL_T_START]): r for r in comp_rows}

    pairs = (
        _all_windows(jaw_rows, comp_rows)
        if args.all
        else _find_disagree_windows(jaw_rows, comp_rows)
    )

    if not pairs:
        print("[annotate] No disagreement windows — engines agree on everything.")
        print("           Use --all to annotate all windows.")
        return

    print(f"[annotate] Session : {sdir.name}")
    print(f"[annotate] Windows : {len(pairs)} {'(all)' if args.all else '(disagree only)'}")
    if not args.all:
        print(f"[annotate] Agree   : {len(jaw_rows) - len(pairs)}/{len(jaw_rows)} skipped")
    print()
    print("  *** BLIND MODE — engine labels hidden during annotation ***")
    print("  c=chewing  r=rest  b=bad_face  a/d=prev/next  Tab=next unlabeled  s=save  q=quit")
    print()

    existing: dict[float, str] = {}
    if out_path.exists():
        for row in _read_csv(out_path):
            try:
                existing[float(row["t_start"])] = row["label"]
            except (KeyError, ValueError):
                pass
        print(f"[annotate] Resuming — {len(existing)} labels already saved.")

    # Build annotated list (extra stats keys stripped on CSV write via extrasaction=ignore)
    annotated: list[dict] = []
    for jrow, crow in pairs:
        t_start   = float(jrow[COL_T_START])
        comp_row  = comp_by_t.get(t_start, {})
        annotated.append({
            "t_start":         jrow[COL_T_START],
            "t_end":           jrow[COL_T_END],
            "label":           existing.get(t_start, ""),
            "jaw_open_label":  jrow[COL_LABEL],
            "composite_label": crow[COL_LABEL] if crow else "",
            "jaw_open_mean":   comp_row.get("jaw_open_mean", ""),
            "mar_mean":        comp_row.get("mar_mean", ""),
        })

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        sys.exit(f"[annotate] Cannot open: {video_path}")
    fps   = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cache = FrameCache(cap, fps, annotated)

    signals = _load_frame_signals(sdir)
    if signals is not None:
        print(f"[annotate] Signal trace loaded ({len(signals['t'])} frames)")
    else:
        print("[annotate] No frame_signals_ours.csv — signal panel will be empty")

    cv2.namedWindow("annotate", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("annotate", TOTAL_W, TOTAL_H)

    total = len(annotated)
    # Start at first unlabeled window
    idx = next((i for i, a in enumerate(annotated) if not a["label"]), 0)
    frame_pos = 0

    while True:
        r      = annotated[idx]
        frames = cache.get(idx)
        frame  = frames[frame_pos % len(frames)]

        video_panel  = _draw_video_hud(frame, idx, total,
                                       float(r["t_start"]), float(r["t_end"]),
                                       r["label"] or None)
        signal_panel = _build_signal_panel(signals, annotated, idx)
        top_row      = np.hstack([video_panel, signal_panel])
        strip        = _build_progress_strip(annotated, idx)
        canvas       = np.vstack([top_row, strip])

        cv2.imshow("annotate", canvas)
        key = cv2.waitKey(33) & 0xFF

        if key == 255:   # no key — advance playback
            frame_pos += 1
            continue

        frame_pos = 0

        if key == ord("c"):
            annotated[idx]["label"] = "chewing"
            idx = min(idx + 1, total - 1)
        elif key == ord("r"):
            annotated[idx]["label"] = "rest"
            idx = min(idx + 1, total - 1)
        elif key == ord("b"):
            annotated[idx]["label"] = "bad_face"
            idx = min(idx + 1, total - 1)
        elif key == ord("a"):
            idx = max(idx - 1, 0)
        elif key == ord("d"):
            idx = min(idx + 1, total - 1)
        elif key == 9:   # Tab — next unlabeled
            nxt = next((i for i in range(idx + 1, total) if not annotated[i]["label"]), None)
            if nxt is not None:
                idx = nxt
        elif key == ord("s"):
            _write_human_csv(out_path, annotated)
            done = sum(1 for a in annotated if a["label"])
            print(f"[annotate] Saved {done}/{total} → {out_path}")
        elif key == ord("q"):
            break

        cache.prefetch(idx)

    cv2.destroyAllWindows()
    cap.release()
    _write_human_csv(out_path, annotated)
    done = sum(1 for a in annotated if a["label"])
    print(f"[annotate] Saved {done}/{total} → {out_path}")
    _print_report(annotated)


if __name__ == "__main__":
    main()
