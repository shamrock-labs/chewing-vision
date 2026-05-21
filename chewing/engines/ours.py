"""OursEngine: MediaPipe Face Landmarker-based chewing analysis (US-002, SPEC §8).

Signal primitives (compute_mar/jaw_open/chin_y/head_motion) live in
chewing.signals (US-004); smoothing dispatcher lives in chewing.smoothing,
peak detector in chewing.peaks (US-005). Quality + window/bout segmentation
helpers remain inline until US-014 extracts them.

All thresholds are fps-relative or signal-std-relative — no pixel-absolute
values (SPEC §18, US-002 AC7).
"""

from __future__ import annotations

import urllib.request
from pathlib import Path
from typing import List, Optional

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

from chewing.engines.base import EngineBase
from chewing.peaks import find_chew_peaks
from chewing.quality import compute_frame_quality, compute_window_quality
from chewing.segmentation import segment_bouts
from chewing.signals import (
    compute_chin_y,
    compute_head_motion,
    compute_jaw_open,
    compute_mar,
    landmarks_to_bbox,
)
from chewing.smoothing import apply_smoothing
from chewing.types import ChewEvent, FrameSignal, Result, WindowLabel


# MediaPipe Face Landmarker model — downloaded once into user cache (not repo root)
# to keep the working tree clean and the cache shared across project checkouts.
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)
MODEL_CACHE_DIR = Path.home() / ".cache" / "chewing-vision"
MODEL_PATH = MODEL_CACHE_DIR / "face_landmarker.task"

WINDOW_SEC = 1.0


def _ensure_model(path: Path = MODEL_PATH) -> Path:
    """Download face_landmarker.task into user cache on first call."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        urllib.request.urlretrieve(MODEL_URL, str(path))
    return path


def _interp_nan(signal: np.ndarray) -> np.ndarray:
    """Linear-interpolate NaN entries. scipy.find_peaks cannot accept NaN inputs."""
    valid = ~np.isnan(signal)
    if valid.sum() == 0:
        return np.zeros_like(signal)
    idx = np.arange(len(signal))
    return np.interp(idx, idx[valid], signal[valid])


def _build_windows(
    frames: List[FrameSignal],
    events: List[ChewEvent],
    window_sec: float = WINDOW_SEC,
) -> List[WindowLabel]:
    """Aggregate frames into 1-sec windows (SPEC §8.7).

    Label vocabulary restricted to {chewing, rest, bad_face} for US-002; the
    7-vocab expansion belongs to US-015.
    """
    if not frames:
        return []
    duration = frames[-1].t_sec
    n_windows = int(duration // window_sec) + 1
    windows: List[WindowLabel] = []
    for i in range(n_windows):
        t_start = i * window_sec
        t_end = t_start + window_sec
        win_frames = [f for f in frames if t_start <= f.t_sec < t_end]
        if not win_frames:
            continue
        face_rate = sum(1 for f in win_frames if f.face_found) / len(win_frames)
        mar_vals = [f.mar for f in win_frames if f.mar is not None]
        jaw_vals = [f.jaw_open for f in win_frames if f.jaw_open is not None]
        mar_mean = float(np.mean(mar_vals)) if mar_vals else 0.0
        jaw_mean = float(np.mean(jaw_vals)) if jaw_vals else 0.0
        # Count events from the primary signal only (caller pre-filters to primary_events).
        n_events = sum(1 for e in events if t_start <= e.t_sec < t_end)
        if face_rate < 0.6:
            label, confidence = "bad_face", 0.3
        elif n_events >= 1:
            label = "chewing"
            confidence = min(1.0, 0.5 + 0.1 * n_events)
        else:
            label, confidence = "rest", 0.7
        window_quality = compute_window_quality(
            [f.quality for f in win_frames], face_found_rate=face_rate
        )
        windows.append(
            WindowLabel(
                t_start=t_start,
                t_end=t_end,
                label=label,
                mar_mean=mar_mean,
                jaw_open_mean=jaw_mean,
                confidence=confidence,
                quality=window_quality,
                n_events=n_events,
            )
        )
    return windows


def _chin_jaw_crossval(
    windows: List[WindowLabel],
    events: List[ChewEvent],
    primary_label: str,
    lookahead: int = 2,
) -> List[WindowLabel]:
    """Two-pass FP reduction for chin_y fallback.

    Windows whose chewing evidence comes from the primary signal → "anchored".
    Windows with only chin_y events → "tentative"; promoted to chewing only when
    an anchored window exists within ±lookahead positions.
    bad_face windows are non-chewing and do NOT count as anchors.
    Tentative windows cannot validate each other (chain-prevention).
    """
    if not windows:
        return windows

    statuses = []
    for w in windows:
        if w.label != "chewing":
            statuses.append("non_chewing")
            continue
        has_primary = any(
            w.t_start <= e.t_sec < w.t_end and e.source_signal == primary_label
            for e in events
        )
        statuses.append("anchored" if has_primary else "tentative")

    result = []
    for i, (w, status) in enumerate(zip(windows, statuses)):
        if status != "tentative":
            result.append(w)
            continue
        lo = max(0, i - lookahead)
        hi = min(len(statuses) - 1, i + lookahead)
        has_anchor = any(statuses[j] == "anchored" for j in range(lo, hi + 1) if j != i)
        if has_anchor:
            result.append(w)
        else:
            result.append(WindowLabel(
                t_start=w.t_start, t_end=w.t_end, label="rest",
                mar_mean=w.mar_mean, jaw_open_mean=w.jaw_open_mean,
                confidence=w.confidence, quality=w.quality, n_events=w.n_events,
            ))
    return result


def _smooth_labels(windows: List[WindowLabel]) -> List[WindowLabel]:
    """Temporal smoothing: flip isolated labels to match both neighbors.

    An isolated "rest" surrounded by "chewing" on both sides is likely a FN;
    an isolated "chewing" surrounded by "rest" is likely a FP.
    bad_face windows act as boundaries — smoothing does not cross them.
    """
    if len(windows) < 3:
        return windows
    labels = [w.label for w in windows]
    smoothed = list(labels)
    for i in range(1, len(labels) - 1):
        prev, curr, nxt = labels[i - 1], labels[i], labels[i + 1]
        if curr == "bad_face":
            continue
        if prev in ("chewing", "rest") and nxt in ("chewing", "rest") and prev == nxt and curr != prev:
            smoothed[i] = prev

    result = []
    for w, new_label in zip(windows, smoothed):
        if new_label == w.label:
            result.append(w)
        else:
            result.append(WindowLabel(
                t_start=w.t_start, t_end=w.t_end, label=new_label,
                mar_mean=w.mar_mean, jaw_open_mean=w.jaw_open_mean,
                confidence=w.confidence, quality=w.quality, n_events=w.n_events,
            ))
    return result


class OursEngine(EngineBase):
    """In-house chewing detector built on MediaPipe Face Landmarker.

    signal_mode controls which signal drives peak detection and window labeling:
      "jaw_open"  — original behaviour (jaw_open primary, mar stored only)
      "mar"       — MAR primary
      "composite" — weighted sum: mar_weight*MAR + (1-mar_weight)*jaw_open

    use_chin_fallback: also detect peaks in chin-to-nose displacement (chin_y)
      and merge them with primary events. Catches closed-mouth chewing where
      jaw_open/MAR stay near zero but chin still oscillates.
    """

    def __init__(
        self,
        signal_mode: str = "jaw_open",
        mar_weight: float = 0.7,
        use_chin_fallback: bool = False,
        use_temporal_smooth: bool = False,
    ) -> None:
        assert signal_mode in ("jaw_open", "mar", "composite"), f"Unknown signal_mode: {signal_mode}"
        self.signal_mode = signal_mode
        self.mar_weight = mar_weight
        self.use_chin_fallback = use_chin_fallback
        self.use_temporal_smooth = use_temporal_smooth

    @property
    def engine_name(self) -> str:
        suffix = "" if self.signal_mode == "jaw_open" else f"_{self.signal_mode}"
        chin_suffix = "_chin" if self.use_chin_fallback else ""
        smooth_suffix = "_smooth" if self.use_temporal_smooth else ""
        return f"ours{suffix}{chin_suffix}{smooth_suffix}"

    def analyze(
        self,
        video_path: str,
        start: Optional[float] = None,
        end: Optional[float] = None,
    ) -> Result:
        model_path = _ensure_model()
        base_options = mp_python.BaseOptions(model_asset_path=str(model_path))
        options = mp_vision.FaceLandmarkerOptions(
            base_options=base_options,
            output_face_blendshapes=True,
            running_mode=mp_vision.RunningMode.VIDEO,
            num_faces=1,
        )

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")

        raw_fps = cap.get(cv2.CAP_PROP_FPS)
        fps = float(raw_fps) if raw_fps and raw_fps > 0 else 30.0
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        start_frame = int(start * fps) if start is not None else 0
        end_frame = int(end * fps) if end is not None else n_frames
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        frames: List[FrameSignal] = []
        prev_landmarks = None

        try:
            with mp_vision.FaceLandmarker.create_from_options(options) as landmarker:
                frame_idx = start_frame
                while frame_idx < end_frame:
                    ok, frame_bgr = cap.read()
                    if not ok:
                        break
                    t = frame_idx / fps
                    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                    h, w = frame_rgb.shape[:2]
                    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
                    detection = landmarker.detect_for_video(mp_image, int(t * 1000))

                    if detection.face_landmarks:
                        landmarks = detection.face_landmarks[0]
                        face_bbox = landmarks_to_bbox(landmarks)
                        mar = compute_mar(landmarks, w, h)
                        jaw_open = (
                            compute_jaw_open(detection.face_blendshapes[0])
                            if detection.face_blendshapes
                            else 0.0
                        )
                        chin_y = compute_chin_y(landmarks, w, h)
                        head_motion = compute_head_motion(
                            prev_landmarks, landmarks, face_bbox
                        )
                        quality = compute_frame_quality(True, 1.0, head_motion)
                        frames.append(
                            FrameSignal(
                                t_sec=t,
                                frame_index=frame_idx,
                                face_found=True,
                                mar=mar,
                                jaw_open=jaw_open,
                                chin_y=chin_y,
                                head_motion=head_motion,
                                quality=quality,
                            )
                        )
                        prev_landmarks = landmarks
                    else:
                        frames.append(
                            FrameSignal(
                                t_sec=t,
                                frame_index=frame_idx,
                                face_found=False,
                                mar=None,
                                jaw_open=None,
                                chin_y=None,
                                head_motion=None,
                                quality=0.0,
                            )
                        )
                        # Reset prev_landmarks so the next-detected frame is treated
                        # as a fresh segment (motion=0 instead of a stale delta).
                        prev_landmarks = None
                    frame_idx += 1
        finally:
            cap.release()

        if not frames:
            return Result(
                engine_name=self.engine_name,
                duration_sec=0.0,
                fps=fps,
                face_detection_rate=0.0,
                n_chews=0,
                chews_per_min=0.0,
                video_path=video_path,
                frame_count=n_frames,
                usable_duration_sec=0.0,
                warnings=["No frames could be read from video."],
            )

        face_rate = sum(1 for f in frames if f.face_found) / len(frames)
        duration = (
            frames[-1].t_sec - frames[0].t_sec if len(frames) > 1 else 1.0 / fps
        )
        usable_duration = sum(1 for f in frames if f.face_found) / fps

        jaw_array = np.array(
            [f.jaw_open if f.jaw_open is not None else np.nan for f in frames]
        )
        mar_array = np.array(
            [f.mar if f.mar is not None else np.nan for f in frames]
        )
        jaw_smoothed = apply_smoothing(_interp_nan(jaw_array), "default", fps=fps)
        mar_smoothed = apply_smoothing(_interp_nan(mar_array), "default", fps=fps)

        def _norm(sig: np.ndarray) -> np.ndarray:
            lo, hi = sig.min(), sig.max()
            return (sig - lo) / (hi - lo + 1e-9)

        if self.signal_mode == "composite":
            primary = self.mar_weight * _norm(mar_smoothed) + (1 - self.mar_weight) * _norm(jaw_smoothed)
            primary_label = "composite"
        elif self.signal_mode == "mar":
            primary = mar_smoothed
            primary_label = "mar"
        else:
            primary = jaw_smoothed
            primary_label = "jaw_open"

        primary_peaks = find_chew_peaks(primary, fps)

        # always compute both for storage
        jaw_peaks = find_chew_peaks(jaw_smoothed, fps) if primary_label != "jaw_open" else primary_peaks
        mar_peaks = find_chew_peaks(mar_smoothed, fps) if primary_label != "mar" else primary_peaks

        # chin_y fallback: detect peaks in chin-nose relative displacement with
        # lower prominence threshold (0.3 vs 0.5) since closed-mouth amplitude is smaller.
        chin_y_array = np.array(
            [f.chin_y if f.chin_y is not None else np.nan for f in frames]
        )
        chin_smoothed = apply_smoothing(_interp_nan(chin_y_array), "default", fps=fps)
        chin_peaks = (
            find_chew_peaks(chin_smoothed, fps, prominence_std=0.3)
            if self.use_chin_fallback
            else np.array([], dtype=int)
        )

        events: List[ChewEvent] = []
        for i in primary_peaks:
            events.append(
                ChewEvent(
                    t_sec=float(frames[i].t_sec),
                    signal_value=float(primary[i]),
                    source_signal=primary_label,
                    frame_index=int(frames[i].frame_index),
                )
            )
        # store secondary signals for inspection (not counted)
        if primary_label != "jaw_open":
            for i in jaw_peaks:
                events.append(ChewEvent(
                    t_sec=float(frames[i].t_sec),
                    signal_value=float(jaw_smoothed[i]),
                    source_signal="jaw_open",
                    frame_index=int(frames[i].frame_index),
                ))
        if primary_label != "mar":
            for i in mar_peaks:
                events.append(ChewEvent(
                    t_sec=float(frames[i].t_sec),
                    signal_value=float(mar_smoothed[i]),
                    source_signal="mar",
                    frame_index=int(frames[i].frame_index),
                ))
        for i in chin_peaks:
            events.append(ChewEvent(
                t_sec=float(frames[i].t_sec),
                signal_value=float(chin_smoothed[i]),
                source_signal="chin_y",
                frame_index=int(frames[i].frame_index),
            ))
        events.sort(key=lambda e: e.t_sec)

        # chin_y events are counted alongside primary when fallback is active
        primary_sources = {primary_label}
        if self.use_chin_fallback:
            primary_sources.add("chin_y")
        primary_events = [e for e in events if e.source_signal in primary_sources]
        windows = _build_windows(frames, primary_events)
        if self.use_chin_fallback:
            windows = _chin_jaw_crossval(windows, primary_events, primary_label)
        if self.use_temporal_smooth:
            windows = _smooth_labels(windows)
        bouts = segment_bouts(primary_events)
        n_chews = int(len(primary_peaks))

        return Result(
            engine_name=self.engine_name,
            duration_sec=duration,
            fps=fps,
            face_detection_rate=face_rate,
            n_chews=n_chews,
            chews_per_min=60.0 * n_chews / max(duration, 1e-6),
            events=events,
            windows=windows,
            extra={"n_mar_peaks": int(len(mar_peaks))},
            video_path=video_path,
            frame_count=n_frames,
            usable_duration_sec=usable_duration,
            frames=frames,
            bouts=bouts,
            warnings=[],
        )
