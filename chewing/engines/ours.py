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
        # Count jaw_open events only — primary signal, avoids double counting.
        n_events = sum(
            1
            for e in events
            if t_start <= e.t_sec < t_end and e.source_signal == "jaw_open"
        )
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


class OursEngine(EngineBase):
    """In-house chewing detector built on MediaPipe Face Landmarker.

    n_chews semantics: count of jaw_open peaks only (primary signal, SPEC §8.6).
    MAR peaks are stored in Result.events with source_signal='mar' for inspection
    but never contribute to n_chews — avoids double counting on signals that
    typically co-fire.
    """

    @property
    def engine_name(self) -> str:
        return "ours"

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

        jaw_peaks = find_chew_peaks(jaw_smoothed, fps)
        mar_peaks = find_chew_peaks(mar_smoothed, fps)

        events: List[ChewEvent] = []
        for i in jaw_peaks:
            events.append(
                ChewEvent(
                    t_sec=float(frames[i].t_sec),
                    signal_value=float(jaw_smoothed[i]),
                    source_signal="jaw_open",
                    frame_index=int(frames[i].frame_index),
                )
            )
        for i in mar_peaks:
            events.append(
                ChewEvent(
                    t_sec=float(frames[i].t_sec),
                    signal_value=float(mar_smoothed[i]),
                    source_signal="mar",
                    frame_index=int(frames[i].frame_index),
                )
            )
        events.sort(key=lambda e: e.t_sec)

        windows = _build_windows(frames, events)
        bouts = segment_bouts([e for e in events if e.source_signal == "jaw_open"])
        n_chews = int(len(jaw_peaks))

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
