"""OrofacEngine: thin wrapper around orofacIAnalysis.ChewAnnotator (US-003, SPEC §0).

orofacIAnalysis 0.1.2 is MIT-licensed (Cameron Maloney) — attributed in
ATTRIBUTION.md and README. Internally it uses pixel-absolute peak prominence
(prominence=10 in Cycle.fit), so orofac's outputs are *informational only*
and must not be treated as v1 ground-truth labels (per SPEC §0).

This wrapper:
  * delegates frame-by-frame analysis to ChewAnnotator,
  * flattens per-cycle peaks into Result.events,
  * stores side counts (left/right/middle) in Result.extra,
  * records its informational-only status and the t_sec-approximation
    caveat in Result.warnings.

Compatibility note: orofacIAnalysis 0.1.2 default detectors reference
``mediapipe.solutions``, which was removed in mediapipe >= 0.10.30. To run on
modern mediapipe we inject Tasks-API adapters that satisfy orofac's
FaceDetector / HandDetector interfaces. The adapters are also harmless on
older mediapipe where ``mp.solutions`` still works — orofac just never
instantiates its built-in defaults because we pass non-None detectors.

Result.frames / windows / bouts are intentionally empty: orofac does not
expose per-frame signal data through ChewAnnotator's public API. The OursEngine
covers that surface.
"""

from __future__ import annotations

import urllib.request
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from orofacIAnalysis.chew_annotator import ChewAnnotator
from orofacIAnalysis.detectors.base import (
    FaceDetector as OrofacFaceDetector,
    HandDetector as OrofacHandDetector,
    LandmarkPoint,
    LandmarkResult,
)

from chewing.engines.base import EngineBase
from chewing.types import ChewEvent, Result


# Reuse the shared model cache; OursEngine writes face_landmarker.task here.
MODEL_CACHE_DIR = Path.home() / ".cache" / "chewing-vision"
FACE_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)
HAND_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)
FACE_MODEL_PATH = MODEL_CACHE_DIR / "face_landmarker.task"
HAND_MODEL_PATH = MODEL_CACHE_DIR / "hand_landmarker.task"


OROFAC_INFORMATIONAL_WARNING = (
    "orofac engine uses pixel-absolute peak prominence (prominence=10 in "
    "orofacIAnalysis.Cycle.fit); results are informational only and not v1 "
    "ground-truth labels. left/right/middle side counts are weak meta."
)

# Cycle.jaw_movements only appends on frames where (no hand) AND (face found),
# so peak indices into jaw_movements are not 1:1 with absolute frame indices.
# On clean fixtures the drift is small (most frames append), but confounder
# clips with frequent hand-in-frame can drift by seconds. Flag for US-010.
T_SEC_APPROX_WARNING = (
    "orofac event t_sec values are approximated as (cycle.start_frame + "
    "local_peak_index) / fps; orofac's internal jaw_movements buffer is "
    "sparse (skips hand-in-frame and face-missing frames), so timestamps may "
    "drift on confounder videos. Do not trust ±300ms event-F1 against these."
)


def _ensure(url: str, path: Path) -> Path:
    """Download model file on first call (inline, not extracted as helper yet)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        urllib.request.urlretrieve(url, str(path))
    return path


class TasksApiFaceDetector(OrofacFaceDetector):
    """Adapter: MediaPipe Tasks FaceLandmarker → orofac FaceDetector interface.

    orofac only reads landmark indices 4 (nose) and 152 (gnathion / chin); we
    return the full 478-landmark list so those indices resolve directly.
    """

    def __init__(self) -> None:
        model = _ensure(FACE_MODEL_URL, FACE_MODEL_PATH)
        options = mp_vision.FaceLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(model)),
            running_mode=mp_vision.RunningMode.IMAGE,
            num_faces=1,
        )
        self._landmarker = mp_vision.FaceLandmarker.create_from_options(options)

    def process_image(self, image: np.ndarray) -> List[LandmarkResult]:
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
        result = self._landmarker.detect(mp_image)
        out: List[LandmarkResult] = []
        for face in result.face_landmarks:
            pts = [LandmarkPoint(x=p.x, y=p.y, z=getattr(p, "z", 0.0)) for p in face]
            out.append(LandmarkResult(landmarks=pts, confidence=1.0))
        return out

    def get_landmark_indices(self) -> Dict[str, int]:
        # orofac defaults to {nose:4, gnathion:152} when keys are absent, but
        # being explicit avoids the get-with-default branch.
        return {"nose": 4, "gnathion": 152}

    def visualize(self, image: np.ndarray, results: List[LandmarkResult]) -> np.ndarray:
        # ChewAnnotator never calls visualize; satisfy the ABC and no-op.
        return image


class TasksApiHandDetector(OrofacHandDetector):
    """Adapter: MediaPipe Tasks HandLandmarker → orofac HandDetector interface.

    ChewAnnotator.detect_hand only checks ``len(results) > 0`` — presence,
    not landmarks. We return one empty-landmark LandmarkResult per detected
    hand so the truthy check works without paying the cost of marshalling
    21 hand landmarks orofac will discard anyway.
    """

    def __init__(self) -> None:
        model = _ensure(HAND_MODEL_URL, HAND_MODEL_PATH)
        options = mp_vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(model)),
            running_mode=mp_vision.RunningMode.IMAGE,
            num_hands=2,
        )
        self._landmarker = mp_vision.HandLandmarker.create_from_options(options)

    def process_image(self, image: np.ndarray) -> List[LandmarkResult]:
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
        result = self._landmarker.detect(mp_image)
        return [LandmarkResult(landmarks=[], confidence=1.0) for _ in result.hand_landmarks]

    def get_landmark_indices(self) -> Dict[str, int]:
        return {}

    def visualize(self, image: np.ndarray, results: List[LandmarkResult]) -> np.ndarray:
        return image


class OrofacEngine(EngineBase):
    """Wrapper around orofacIAnalysis.ChewAnnotator emitting a unified Result."""

    @property
    def engine_name(self) -> str:
        return "orofac"

    def analyze(
        self,
        video_path: str,
        start: Optional[float] = None,
        end: Optional[float] = None,
    ) -> Result:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")
        raw_fps = cap.get(cv2.CAP_PROP_FPS)
        fps = float(raw_fps) if raw_fps and raw_fps > 0 else 30.0
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        warnings: List[str] = [OROFAC_INFORMATIONAL_WARNING, T_SEC_APPROX_WARNING]
        if start is not None or end is not None:
            warnings.append(
                "orofac engine does not support start/end trimming; full video processed."
            )

        annotator = ChewAnnotator(
            video_path=video_path,
            face_detector=TasksApiFaceDetector(),
            hand_detector=TasksApiHandDetector(),
        )
        cycle_dicts = annotator.analyze_chewing()

        events: List[ChewEvent] = []
        n_peaks = 0
        left = right = middle = 0
        for cycle in cycle_dicts:
            cycle_start = int(cycle["start_frame"])
            peaks = cycle.get("peaks") or []
            jaw_mvts = cycle.get("jaw_movements") or []
            for local_idx in peaks:
                idx = int(local_idx)
                signal_value = float(jaw_mvts[idx]) if 0 <= idx < len(jaw_mvts) else 0.0
                events.append(
                    ChewEvent(
                        t_sec=(cycle_start + idx) / fps,
                        signal_value=signal_value,
                        source_signal="chin_y",
                        frame_index=cycle_start + idx,
                    )
                )
            n_peaks += len(peaks)
            # Per orofac chew_annotator.py:225-233, directions are only appended
            # when a peak has a following valley AND vertical_motion < 0; thus
            # len(directions) ≤ len(peaks) per cycle, so summed:
            # left+right+middle ≤ n_chews (AC3 invariant, structurally guaranteed).
            left += int(cycle.get("left", 0))
            right += int(cycle.get("right", 0))
            middle += int(cycle.get("middle", 0))

        duration_sec = frame_count / fps if fps > 0 else 0.0
        n_chews = n_peaks

        return Result(
            engine_name=self.engine_name,
            duration_sec=duration_sec,
            fps=fps,
            face_detection_rate=0.0,
            n_chews=n_chews,
            chews_per_min=60.0 * n_chews / max(duration_sec, 1e-6),
            events=events,
            windows=[],
            extra={
                "left": left,
                "right": right,
                "middle": middle,
                "n_cycles": len(cycle_dicts),
            },
            video_path=video_path,
            frame_count=frame_count,
            usable_duration_sec=duration_sec,
            frames=[],
            bouts=[],
            warnings=warnings,
        )
