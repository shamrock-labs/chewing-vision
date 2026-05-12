"""Per-frame and per-window quality scoring (US-014, SPEC §8.7-8.8).

Both functions return values in [0, 1]. compute_frame_quality is what
OursEngine uses to populate FrameSignal.quality; compute_window_quality is
exposed for downstream consumers (US-006 will wire it into WindowLabel.quality
once that field is added).
"""

from __future__ import annotations

from typing import Sequence


def compute_frame_quality(
    face_found: bool,
    landmark_confidence: float,
    head_motion: float,
) -> float:
    """Per-frame quality in [0, 1].

    Returns 0.0 when no face is detected. Otherwise:
        ``landmark_confidence * max(0, 1 - min(1, head_motion))``

    OursEngine currently passes ``landmark_confidence=1.0`` because the Tasks
    API doesn't expose a per-frame landmark confidence; this collapses to the
    pre-US-014 formula. If a future story starts extracting a real confidence
    value, the OursEngine output may shift and US-002's byte-identical
    regression will need updating.
    """
    if not face_found:
        return 0.0
    motion_penalty = max(0.0, 1.0 - min(1.0, head_motion))
    return max(0.0, min(1.0, landmark_confidence * motion_penalty))


def compute_window_quality(
    frame_qualities: Sequence[float],
    face_found_rate: float,
) -> float:
    """Per-window quality in [0, 1].

    Formula: ``face_found_rate * mean(frame_qualities)``. Empty
    ``frame_qualities`` returns 0.0 (degenerate input → degenerate output).
    """
    if not frame_qualities:
        return 0.0
    mean_q = sum(frame_qualities) / len(frame_qualities)
    return max(0.0, min(1.0, face_found_rate * mean_q))
