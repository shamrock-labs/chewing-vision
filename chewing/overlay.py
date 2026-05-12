"""Demo overlay MP4 rendering (US-009, SPEC §12.3).

v1 renders the AC-gated surfaces: 1600x840 canvas, 1280x720 video,
320x720 sidebar, 1600x120 rolling trace, state border, and red event
markers. It intentionally skips 478 landmark dots because Result does not
persist landmarks and re-running MediaPipe would roughly double smoke time.
"""

from __future__ import annotations

import math
import os
from typing import Iterable

import cv2
import numpy as np

from chewing.types import Bout, Result, WindowLabel


CANVAS_W = 1600
CANVAS_H = 840
VIDEO_W = 1280
VIDEO_H = 720
SIDEBAR_W = 320
TRACE_H = 120

CHEWING_BGR = (94, 197, 34)  # #22c55e
PEAK_BGR = (68, 68, 239)  # #ef4444
REST_BGR = (128, 128, 128)
BAD_FACE_BGR = (32, 32, 200)
BG_BGR = (18, 24, 32)
SIDEBAR_BG_BGR = (28, 36, 48)
TRACE_BG_BGR = (12, 18, 26)
TEXT_BGR = (245, 247, 250)
MUTED_BGR = (160, 170, 185)


def _safe_signal_value(frame, signal: str) -> float:
    value = getattr(frame, signal, None)
    return float(value) if value is not None else math.nan


def _current_window(windows: Iterable[WindowLabel], t_sec: float) -> WindowLabel | None:
    for window in windows:
        if window.t_start <= t_sec < window.t_end:
            return window
    return None


def _border_color(label: str | None) -> tuple[int, int, int]:
    if label == "chewing":
        return CHEWING_BGR
    if label in {"bad_face", "occluded"}:
        return BAD_FACE_BGR
    return REST_BGR


def _count_so_far(result: Result, t_sec: float, signal: str) -> int:
    return sum(
        1
        for event in result.events
        if event.source_signal == signal and event.t_sec <= t_sec
    )


def _bouts_so_far(bouts: list[Bout], t_sec: float) -> tuple[int, float]:
    seen = 0
    current_duration = 0.0
    for bout in bouts:
        if bout.t_start <= t_sec:
            seen += 1
        if bout.t_start <= t_sec <= bout.t_end:
            current_duration = t_sec - bout.t_start
    return seen, current_duration


def _draw_sidebar(canvas: np.ndarray, result: Result, t_sec: float, signal: str) -> None:
    x0 = VIDEO_W
    canvas[0:VIDEO_H, x0:CANVAS_W] = SIDEBAR_BG_BGR
    count = _count_so_far(result, t_sec, signal)
    rate = 60.0 * count / max(t_sec, 1.0)
    n_bouts, current_bout_sec = _bouts_so_far(result.bouts, t_sec)

    def put(text: str, y: int, scale: float = 0.7, color=TEXT_BGR, thick: int = 1) -> None:
        cv2.putText(
            canvas,
            text,
            (x0 + 24, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            color,
            thick,
            cv2.LINE_AA,
        )

    put("CHEW COUNT", 70, 0.8, MUTED_BGR, 1)
    put(str(count), 145, 2.0, TEXT_BGR, 3)
    put("rate", 205, 0.75, MUTED_BGR, 1)
    put(f"{rate:5.1f} /min", 250, 1.0, TEXT_BGR, 2)
    put("cycle stats", 325, 0.75, MUTED_BGR, 1)
    put(f"bouts: {n_bouts}", 370, 0.85, TEXT_BGR, 2)
    put(f"current: {current_bout_sec:4.1f}s", 410, 0.75, TEXT_BGR, 1)
    put(f"engine: {result.engine_name}", 500, 0.72, TEXT_BGR, 1)
    put(f"signal: {signal}", 535, 0.72, TEXT_BGR, 1)
    put(f"t = {t_sec:06.2f}s", 570, 0.72, TEXT_BGR, 1)

    cv2.rectangle(canvas, (x0 + 24, 620), (x0 + 70, 646), CHEWING_BGR, -1)
    put("chewing", 642, 0.58, TEXT_BGR, 1)
    cv2.circle(canvas, (x0 + 47, 682), 10, PEAK_BGR, -1)
    put("peak", 688, 0.58, TEXT_BGR, 1)


def _draw_trace(canvas: np.ndarray, result: Result, t_sec: float, signal: str) -> None:
    y0 = VIDEO_H
    canvas[y0:CANVAS_H, 0:CANVAS_W] = TRACE_BG_BGR
    cv2.putText(
        canvas,
        f"{signal} rolling trace",
        (24, y0 + 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        MUTED_BGR,
        1,
        cv2.LINE_AA,
    )

    frames = result.frames
    if not frames:
        return
    left_t = max(frames[0].t_sec, t_sec - 3.0)
    right_t = min(frames[-1].t_sec, t_sec + 3.0)
    if right_t <= left_t:
        right_t = left_t + 1e-6

    samples = [
        (f.t_sec, _safe_signal_value(f, signal))
        for f in frames
        if left_t <= f.t_sec <= right_t
    ]
    values = np.array([v for _, v in samples if not math.isnan(v)], dtype=float)
    if values.size == 0:
        return
    v_min = float(np.min(values))
    v_max = float(np.max(values))
    if v_max - v_min < 1e-9:
        v_max = v_min + 1.0

    def to_xy(ts: float, value: float) -> tuple[int, int]:
        x = int((ts - left_t) / (right_t - left_t) * (CANVAS_W - 48)) + 24
        y_norm = (value - v_min) / (v_max - v_min)
        y = int(y0 + TRACE_H - 18 - y_norm * 72)
        return x, y

    pts = [
        to_xy(ts, value)
        for ts, value in samples
        if not math.isnan(value)
    ]
    if len(pts) >= 2:
        cv2.polylines(canvas, [np.array(pts, dtype=np.int32)], False, (180, 230, 220), 2)

    for event in result.events:
        if event.source_signal != signal or not (left_t <= event.t_sec <= right_t):
            continue
        x, y = to_xy(event.t_sec, float(event.signal_value))
        cv2.circle(canvas, (x, y), 5, PEAK_BGR, -1)

    cursor_x = int((t_sec - left_t) / (right_t - left_t) * (CANVAS_W - 48)) + 24
    cv2.line(canvas, (cursor_x, y0 + 34), (cursor_x, CANVAS_H - 12), TEXT_BGR, 1)


def render_overlay(
    video_path: str,
    result: Result,
    output_path: str,
    signal: str = "jaw_open",
) -> None:
    """Render a deterministic demo MP4 for an analyzed chewing result."""

    if signal not in {"jaw_open", "mar"}:
        raise ValueError("signal must be 'jaw_open' or 'mar'")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = float(result.fps) if result.fps > 0 else 30.0
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    writer = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (CANVAS_W, CANVAS_H),
    )
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Cannot open output writer: {output_path}")

    frame_by_index = {frame.frame_index: frame for frame in result.frames}
    start_frame = min(frame_by_index) if frame_by_index else 0
    end_frame = max(frame_by_index) if frame_by_index else int(result.duration_sec * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    try:
        frame_idx = start_frame
        while frame_idx <= end_frame:
            ok, src = cap.read()
            if not ok:
                break
            frame = frame_by_index.get(frame_idx)
            t_sec = frame.t_sec if frame is not None else frame_idx / fps
            canvas = np.zeros((CANVAS_H, CANVAS_W, 3), dtype=np.uint8)
            canvas[:] = BG_BGR

            resized = cv2.resize(src, (VIDEO_W, VIDEO_H), interpolation=cv2.INTER_AREA)
            canvas[0:VIDEO_H, 0:VIDEO_W] = resized
            window = _current_window(result.windows, t_sec)
            label = window.label if window is not None else None
            cv2.rectangle(
                canvas,
                (3, 3),
                (VIDEO_W - 4, VIDEO_H - 4),
                _border_color(label),
                6,
            )

            _draw_sidebar(canvas, result, t_sec, signal)
            _draw_trace(canvas, result, t_sec, signal)
            writer.write(canvas)
            frame_idx += 1
    finally:
        writer.release()
        cap.release()
