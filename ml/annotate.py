"""Hand-annotation tool for chewing window labeling (blind).

Finds windows where jaw_open and composite engines *disagree*, shows the video
frames for each window (BLIND — no engine labels shown during annotation), and
collects a human ground-truth label. After annotation, prints a comparison
report (jaw_open accuracy vs composite accuracy on the disagreement set).

Usage:
    python ml/annotate.py <session_dir> [--all]

    --all   Annotate every window, not just disagreement windows.

Keys during annotation:
    c       label as chewing
    r       label as rest
    b       label as bad_face
    a / ←   go to previous window (without labeling)
    d / →   go to next window (without labeling)
    s       save progress and continue
    q       save progress and quit
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

COL_T_START = "t_start"
COL_T_END   = "t_end"
COL_LABEL   = "label"


def _read_csv(path: Path) -> list[dict]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _write_human_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fieldnames = ["t_start", "t_end", "label", "jaw_open_label", "composite_label"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def _find_disagree_windows(
    jaw_rows: list[dict], comp_rows: list[dict]
) -> list[tuple[dict, dict]]:
    jaw_by_t  = {float(r[COL_T_START]): r for r in jaw_rows}
    comp_by_t = {float(r[COL_T_START]): r for r in comp_rows}
    disagree = []
    for t, jrow in sorted(jaw_by_t.items()):
        crow = comp_by_t.get(t)
        if crow and jrow[COL_LABEL] != crow[COL_LABEL]:
            disagree.append((jrow, crow))
    return disagree


def _all_windows(
    jaw_rows: list[dict], comp_rows: list[dict]
) -> list[tuple[dict, dict]]:
    jaw_by_t  = {float(r[COL_T_START]): r for r in jaw_rows}
    comp_by_t = {float(r[COL_T_START]): r for r in comp_rows}
    return [
        (jaw_by_t[t], comp_by_t.get(t))
        for t in sorted(jaw_by_t.keys())
    ]


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
        frames.append(frame)
    if not frames:
        frames.append(np.zeros((480, 640, 3), dtype=np.uint8))
    return frames


def _draw_hud(
    frame: np.ndarray,
    idx: int,
    total: int,
    t_start: float,
    t_end: float,
    human_label: Optional[str],
) -> np.ndarray:
    out = frame.copy()
    h, w = out.shape[:2]

    overlay = out.copy()
    cv2.rectangle(overlay, (0, h - 90), (w, h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, out, 0.45, 0, out)

    bar_w = int(w * idx / max(total - 1, 1))
    cv2.rectangle(out, (0, h - 90), (bar_w, h - 84), (80, 200, 80), -1)

    label_color = {
        "chewing":  (80, 220, 80),
        "rest":     (200, 200, 80),
        "bad_face": (80, 80, 220),
    }.get(human_label or "", (200, 200, 200))

    def txt(text, x, y, color=(220, 220, 220), scale=0.6, thick=1):
        cv2.putText(out, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)

    txt(f"Window {idx + 1}/{total}   t={t_start:.1f}-{t_end:.1f}s", 10, h - 65)
    label_str = human_label if human_label else "?"
    txt(f"Label: {label_str}", 10, h - 40, color=label_color, scale=0.7, thick=2)
    txt("[c]hewing  [r]est  [b]ad_face  [a/d]prev/next  [s]save  [q]quit", 10, h - 12, scale=0.45)
    return out


def _print_report(annotated: list[dict]) -> None:
    labeled = [r for r in annotated if r["label"] not in (None, "")]
    if not labeled:
        print("[annotate] No completed labels — skipping report.")
        return

    jaw_correct  = sum(1 for r in labeled if r["jaw_open_label"]  == r["label"])
    comp_correct = sum(1 for r in labeled if r["composite_label"] == r["label"])
    n = len(labeled)

    print()
    print("=" * 58)
    print(f"  Disagreement windows annotated : {n}")
    print(f"  jaw_open  matches human GT     : {jaw_correct}/{n}  ({100*jaw_correct/n:.1f}%)")
    print(f"  composite matches human GT     : {comp_correct}/{n}  ({100*comp_correct/n:.1f}%)")
    print("=" * 58)
    print(f"\n  {'t_start':>7}  {'jaw_open':>9}  {'composite':>9}  {'human':>9}")
    print(f"  {'-'*7}  {'-'*9}  {'-'*9}  {'-'*9}")
    for r in labeled:
        jmark = "v" if r["jaw_open_label"]  == r["label"] else "x"
        cmark = "v" if r["composite_label"] == r["label"] else "x"
        print(
            f"  {float(r['t_start']):>7.1f}  "
            f"{r['jaw_open_label']:>8} {jmark}  "
            f"{r['composite_label']:>8} {cmark}  "
            f"{r['label']:>9}"
        )
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Blind chewing annotation tool")
    parser.add_argument("session_dir", type=Path)
    parser.add_argument("--all", action="store_true", help="Annotate all windows (default: disagree only)")
    args = parser.parse_args()

    sdir = args.session_dir.resolve()
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

    pairs = _all_windows(jaw_rows, comp_rows) if args.all else _find_disagree_windows(jaw_rows, comp_rows)

    if not pairs:
        print("[annotate] No disagreement windows — engines agree on everything.")
        print("           Use --all to annotate all windows.")
        return

    print(f"[annotate] Session : {sdir.name}")
    print(f"[annotate] Windows : {len(pairs)} {'(all)' if args.all else '(disagree only)'}")
    if not args.all:
        print(f"[annotate] Agree   : {len(jaw_rows) - len(pairs)}/{len(jaw_rows)} windows skipped")
    print()
    print("  *** BLIND MODE — engine labels hidden during annotation ***")
    print("  [c] chewing  [r] rest  [b] bad_face  [a/d] prev/next  [s] save  [q] quit")
    print()

    existing: dict[float, str] = {}
    if out_path.exists():
        for row in _read_csv(out_path):
            try:
                existing[float(row["t_start"])] = row["label"]
            except (KeyError, ValueError):
                pass
        print(f"[annotate] Resuming — {len(existing)} labels already saved.")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        sys.exit(f"[annotate] Cannot open: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    annotated: list[dict] = []
    for jrow, crow in pairs:
        t_start = float(jrow[COL_T_START])
        annotated.append({
            "t_start":         jrow[COL_T_START],
            "t_end":           jrow[COL_T_END],
            "label":           existing.get(t_start, ""),
            "jaw_open_label":  jrow[COL_LABEL],
            "composite_label": crow[COL_LABEL] if crow else "",
        })

    print("[annotate] Extracting frames... ", end="", flush=True)
    window_frames: list[list[np.ndarray]] = []
    for r in annotated:
        window_frames.append(_extract_frames(cap, fps, float(r["t_start"]), float(r["t_end"])))
    cap.release()
    print("done.")

    cv2.namedWindow("annotate", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("annotate", 800, 600)

    idx, frame_pos = 0, 0
    total = len(annotated)

    while True:
        r      = annotated[idx]
        frames = window_frames[idx]
        frame  = frames[frame_pos % len(frames)]
        cv2.imshow("annotate", _draw_hud(frame, idx, total, float(r["t_start"]), float(r["t_end"]), r["label"] or None))

        key = cv2.waitKey(33) & 0xFF

        if key == 255:
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
        elif key in (ord("a"), 81):
            idx = max(idx - 1, 0)
        elif key in (ord("d"), 83):
            idx = min(idx + 1, total - 1)
        elif key == ord("s"):
            _write_human_csv(out_path, annotated)
            done = sum(1 for r in annotated if r["label"])
            print(f"[annotate] Saved {done}/{total} → {out_path}")
        elif key == ord("q"):
            break

    cv2.destroyAllWindows()
    _write_human_csv(out_path, annotated)
    done = sum(1 for r in annotated if r["label"])
    print(f"[annotate] Saved {done}/{total} → {out_path}")
    _print_report(annotated)


if __name__ == "__main__":
    main()
