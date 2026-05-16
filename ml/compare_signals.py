"""Compare jaw_open vs composite signal mode on a single video.

Usage:
    python ml/compare_signals.py <video_path> [--mar-weight 0.7] [--output-dir ml/outputs]
    python ml/compare_signals.py sessions/.../video.mp4 --no-video
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path
from typing import List

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from chewing.engines.ours import OursEngine, _interp_nan, _norm
from chewing.smoothing import apply_smoothing
from chewing.types import Result

# ── canvas constants ─────────────────────────────────────────────────────────
PANEL_W = 800
VIDEO_H = 450
TRACE_H = 110
HEADER_H = 60
CANVAS_W = PANEL_W * 2          # 1600
CANVAS_H = HEADER_H + VIDEO_H + TRACE_H  # 620

BG       = (18, 24, 32)
HEADER   = (28, 36, 50)
JAW_CLR  = (230, 130, 30)   # orange-ish (BGR)
COMP_CLR = (60, 100, 230)   # blue (BGR)
CHEW_CLR = (40, 190, 60)    # green
REST_CLR = (100, 100, 100)
BAD_CLR  = (40, 40, 160)
TEXT_CLR = (240, 245, 252)
MUTED    = (150, 160, 175)
DIVIDER  = (60, 70, 85)


def _summarise(result: Result, label: str) -> dict:
    chew_wins = sum(1 for w in result.windows if w.label == "chewing")
    rest_wins = sum(1 for w in result.windows if w.label == "rest")
    total_wins = len(result.windows)
    chew_pct = 100 * chew_wins / total_wins if total_wins else 0
    return {
        "label": label,
        "n_chews": result.n_chews,
        "chews_per_min": result.chews_per_min,
        "n_windows": total_wins,
        "chew_windows": chew_wins,
        "rest_windows": rest_wins,
        "chew_pct": chew_pct,
        "duration_sec": result.duration_sec,
    }


def _print_table(jaw_s: dict, comp_s: dict, mar_weight: float) -> None:
    print()
    print("=" * 60)
    print(f"{'Metric':<22} {'jaw_open':>16} {f'composite(w={mar_weight})':>16}")
    print("-" * 60)
    rows = [
        ("n_chews", "n_chews"),
        ("chews_per_min", "chews_per_min"),
        ("chew_windows", "chew_windows"),
        ("rest_windows", "rest_windows"),
        ("chew_pct (%)", "chew_pct"),
        ("duration_sec", "duration_sec"),
    ]
    for display, key in rows:
        jv, cv = jaw_s[key], comp_s[key]
        if isinstance(jv, float):
            print(f"  {display:<20} {jv:>16.1f} {cv:>16.1f}")
        else:
            print(f"  {display:<20} {jv:>16} {cv:>16}")
    print("=" * 60)


def _put(canvas, text, xy, scale=0.6, color=TEXT_CLR, thick=1):
    cv2.putText(canvas, text, xy, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


def _border_color(label):
    return {"chewing": CHEW_CLR, "bad_face": BAD_CLR}.get(label, REST_CLR)


def _current_label(windows, t_sec):
    for w in windows:
        if w.t_start <= t_sec < w.t_end:
            return w.label
    return None


def _count_up_to(events, t_sec, source):
    return sum(1 for e in events if e.source_signal == source and e.t_sec <= t_sec)


def _draw_panel_overlay(canvas, x0, result, t_sec, primary_source, accent_clr, label_str):
    """Draw stats + border on one video half."""
    label = _current_label(result.windows, t_sec)
    # border
    cv2.rectangle(canvas, (x0 + 3, HEADER_H + 3),
                  (x0 + PANEL_W - 4, HEADER_H + VIDEO_H - 4),
                  _border_color(label), 5)
    # stats box top-left
    count = _count_up_to(result.events, t_sec, primary_source)
    rate = 60.0 * count / max(t_sec, 0.5)
    box_x, box_y = x0 + 10, HEADER_H + 10
    cv2.rectangle(canvas, (box_x, box_y), (box_x + 230, box_y + 80), (0, 0, 0), -1)
    cv2.rectangle(canvas, (box_x, box_y), (box_x + 230, box_y + 80), accent_clr, 1)
    _put(canvas, label_str, (box_x + 8, box_y + 22), 0.55, accent_clr, 1)
    _put(canvas, f"chews: {count}   {rate:.1f}/min", (box_x + 8, box_y + 48), 0.58, TEXT_CLR, 1)
    state_clr = CHEW_CLR if label == "chewing" else (MUTED if label == "rest" else BAD_CLR)
    _put(canvas, label or "??", (box_x + 8, box_y + 72), 0.65, state_clr, 2)


def _draw_trace_half(canvas, x0, result, primary_source, primary_signal_vals, frames, t_sec, accent_clr):
    """Draw rolling signal trace for one half of the canvas."""
    y0 = HEADER_H + VIDEO_H
    # background
    canvas[y0:CANVAS_H, x0:x0 + PANEL_W] = (12, 18, 26)

    # window shading
    for w in result.windows:
        wx0 = x0
        wx1 = x0 + PANEL_W - 1
        if w.label == "chewing":
            cv2.rectangle(canvas, (wx0, y0), (wx1, CANVAS_H - 1), (20, 60, 20), -1)
            break  # just mark full strip if any chewing — simple approach
    # (reset and do proper shading)
    canvas[y0:CANVAS_H, x0:x0 + PANEL_W] = (12, 18, 26)
    for w in result.windows:
        if w.label != "chewing":
            continue
        wx0_t = w.t_start
        wx1_t = w.t_end
        left_t = max(frames[0].t_sec, t_sec - 4.0)
        right_t = min(frames[-1].t_sec, t_sec + 4.0)
        if wx1_t < left_t or wx0_t > right_t:
            continue
        def to_x(ts):
            return x0 + int((ts - left_t) / max(right_t - left_t, 1e-6) * PANEL_W)
        cv2.rectangle(canvas, (max(x0, to_x(wx0_t)), y0),
                      (min(x0 + PANEL_W, to_x(wx1_t)), CANVAS_H - 1),
                      (25, 55, 25), -1)

    # label
    _put(canvas, primary_source + " trace", (x0 + 8, y0 + 18), 0.5, MUTED)

    if not frames or primary_signal_vals is None:
        return
    left_t = max(frames[0].t_sec, t_sec - 4.0)
    right_t = min(frames[-1].t_sec, t_sec + 4.0)
    if right_t <= left_t:
        return

    sig = np.array(primary_signal_vals)
    times_arr = np.array([f.t_sec for f in frames])
    mask = (times_arr >= left_t) & (times_arr <= right_t)
    if mask.sum() < 2:
        return

    vals = sig[mask]
    ts = times_arr[mask]
    v_min, v_max = vals.min(), vals.max()
    if v_max - v_min < 1e-9:
        v_max = v_min + 1.0

    def to_xy(t, v):
        px = x0 + int((t - left_t) / (right_t - left_t) * PANEL_W)
        py = CANVAS_H - 8 - int((v - v_min) / (v_max - v_min) * (TRACE_H - 20))
        return px, py

    pts = [to_xy(t, v) for t, v in zip(ts, vals)]
    cv2.polylines(canvas, [np.array(pts, np.int32)], False, accent_clr, 2)

    # peaks
    for e in result.events:
        if e.source_signal != primary_source or not (left_t <= e.t_sec <= right_t):
            continue
        idx = np.argmin(np.abs(times_arr - e.t_sec))
        x, y = to_xy(e.t_sec, sig[idx])
        cv2.circle(canvas, (x, y), 5, (60, 60, 220), -1)

    # cursor
    cx = x0 + int((t_sec - left_t) / (right_t - left_t) * PANEL_W)
    cv2.line(canvas, (cx, y0 + 22), (cx, CANVAS_H - 4), TEXT_CLR, 1)


def render_comparison_video(
    video_path: str,
    jaw_result: Result,
    comp_result: Result,
    output_path: str,
    mar_weight: float = 0.7,
) -> None:
    """Render split-screen comparison demo video."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = float(jaw_result.fps) or 30.0
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    writer = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (CANVAS_W, CANVAS_H),
    )

    frames = jaw_result.frames  # same for both (same video)
    jaw_sig = jaw_result.extra.get("primary_signal_values")
    comp_sig = comp_result.extra.get("primary_signal_values")

    frame_map = {f.frame_index: f for f in frames}
    start_frame = min(frame_map) if frame_map else 0
    end_frame = max(frame_map) if frame_map else int(jaw_result.duration_sec * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    try:
        fi = start_frame
        while fi <= end_frame:
            ok, src = cap.read()
            if not ok:
                break
            frame_obj = frame_map.get(fi)
            t_sec = frame_obj.t_sec if frame_obj else fi / fps

            canvas = np.zeros((CANVAS_H, CANVAS_W, 3), dtype=np.uint8)
            canvas[:] = BG

            # Header
            canvas[0:HEADER_H, :] = HEADER
            _put(canvas, "jaw_open  vs  composite", (20, 38), 0.75, TEXT_CLR, 2)
            _put(canvas, f"t = {t_sec:6.2f}s  |  mar_weight={mar_weight}", (380, 38), 0.6, MUTED)
            _put(canvas, Path(video_path).name, (CANVAS_W - 480, 38), 0.55, MUTED)

            # Video — scale to PANEL_W x VIDEO_H
            resized = cv2.resize(src, (PANEL_W, VIDEO_H), interpolation=cv2.INTER_AREA)
            canvas[HEADER_H:HEADER_H + VIDEO_H, 0:PANEL_W] = resized
            canvas[HEADER_H:HEADER_H + VIDEO_H, PANEL_W:CANVAS_W] = resized

            # Overlays
            _draw_panel_overlay(canvas, 0, jaw_result, t_sec, "jaw_open", JAW_CLR, "jaw_open")
            _draw_panel_overlay(canvas, PANEL_W, comp_result, t_sec, "composite", COMP_CLR,
                                f"composite (w={mar_weight})")

            # Divider
            cv2.line(canvas, (PANEL_W, 0), (PANEL_W, CANVAS_H), DIVIDER, 2)

            # Traces
            _draw_trace_half(canvas, 0, jaw_result, "jaw_open", jaw_sig, frames, t_sec, JAW_CLR)
            _draw_trace_half(canvas, PANEL_W, comp_result, "composite", comp_sig, frames, t_sec, COMP_CLR)

            writer.write(canvas)
            fi += 1
    finally:
        writer.release()
        cap.release()


def make_static_plot(
    jaw_result: Result,
    comp_result: Result,
    mar_weight: float,
    output_path: Path,
) -> None:
    frames = jaw_result.frames
    times = np.array([f.t_sec for f in frames])
    fps = jaw_result.fps

    jaw_arr = np.array([f.jaw_open if f.jaw_open is not None else np.nan for f in frames])
    mar_arr = np.array([f.mar if f.mar is not None else np.nan for f in frames])
    jaw_sm = apply_smoothing(_interp_nan(jaw_arr), "default", fps=fps)
    mar_sm = apply_smoothing(_interp_nan(mar_arr), "default", fps=fps)
    comp_sm = mar_weight * _norm(mar_sm) + (1 - mar_weight) * _norm(jaw_sm)

    jaw_peaks_t = [e.t_sec for e in jaw_result.events if e.source_signal == "jaw_open"]
    comp_peaks_t = [e.t_sec for e in comp_result.events if e.source_signal == "composite"]

    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    fig.suptitle(f"Signal comparison — jaw_open vs composite (w={mar_weight})", fontsize=13)

    def shade(ax, windows):
        for w in windows:
            if w.label == "chewing":
                ax.axvspan(w.t_start, w.t_end, alpha=0.15, color="#FF5722")

    ax0 = axes[0]
    ax0.plot(times, jaw_sm, color="#F57C00", lw=1.2, label="jaw_open (smoothed)")
    for t in jaw_peaks_t:
        ax0.axvline(t, color="#F57C00", alpha=0.4, lw=0.8)
    shade(ax0, jaw_result.windows)
    ax0.set_ylabel("jaw_open")
    ax0.set_title(f"jaw_open — {jaw_result.n_chews} chews detected")
    ax0.legend(fontsize=8)

    ax1 = axes[1]
    ax1.plot(times, comp_sm, color="#1565C0", lw=1.2, label=f"composite (w={mar_weight})")
    for t in comp_peaks_t:
        ax1.axvline(t, color="#1565C0", alpha=0.4, lw=0.8)
    shade(ax1, comp_result.windows)
    ax1.set_ylabel("composite")
    ax1.set_title(f"composite — {comp_result.n_chews} chews detected")
    ax1.legend(fontsize=8)

    ax2 = axes[2]
    ax2.plot(times, _norm(mar_sm), color="#2E7D32", lw=1.0, label="MAR (norm)")
    ax2.plot(times, _norm(jaw_sm), color="#F57C00", lw=0.8, alpha=0.6, label="jaw_open (norm)")
    ax2.set_ylabel("normalised")
    ax2.set_xlabel("Time (s)")
    ax2.set_title("Raw signals normalised for reference")
    ax2.legend(fontsize=8)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=120)
    plt.close(fig)
    print(f"[compare_signals] Plot saved → {output_path}")


def run_comparison(video_path: str, mar_weight: float = 0.7, output_dir: Path = None,
                   render_video: bool = True):
    output_dir = Path(output_dir) if output_dir else Path(video_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n[compare_signals] Video : {video_path}")
    print(f"[compare_signals] MAR weight : {mar_weight}")
    print("[compare_signals] Running jaw_open mode ...")
    jaw_result = OursEngine(signal_mode="jaw_open").analyze(video_path)

    print("[compare_signals] Running composite mode ...")
    comp_result = OursEngine(signal_mode="composite", mar_weight=mar_weight).analyze(video_path)

    jaw_s = _summarise(jaw_result, "jaw_open")
    comp_s = _summarise(comp_result, f"composite(w={mar_weight})")
    _print_table(jaw_s, comp_s, mar_weight)

    make_static_plot(jaw_result, comp_result, mar_weight, output_dir / "signal_comparison.png")

    if render_video:
        out_mp4 = output_dir / "comparison_demo.mp4"
        print(f"[compare_signals] Rendering comparison video → {out_mp4}")
        render_comparison_video(video_path, jaw_result, comp_result, str(out_mp4), mar_weight)
        print(f"[compare_signals] Video saved → {out_mp4}")

    return jaw_s, comp_s


def main():
    parser = argparse.ArgumentParser(description="Compare jaw_open vs composite signal mode")
    parser.add_argument("video", help="Path to video file")
    parser.add_argument("--mar-weight", type=float, default=0.7)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--no-video", action="store_true", help="Skip demo video rendering")
    args = parser.parse_args()
    run_comparison(
        args.video,
        mar_weight=args.mar_weight,
        output_dir=args.output_dir,
        render_video=not args.no_video,
    )


if __name__ == "__main__":
    main()
