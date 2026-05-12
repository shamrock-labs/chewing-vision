"""Unit tests for chewing.signals (US-004).

All inputs are synthetic — no video, no MediaPipe — so these run in <1s.
"""

from __future__ import annotations

import pytest

from chewing.signals import (
    DEFAULT_CHIN_IDX,
    DEFAULT_NOSE_IDX,
    LEFT_CORNER_IDX,
    LOWER_LIP_IDX,
    RIGHT_CORNER_IDX,
    UPPER_LIP_IDX,
    compute_chin_y,
    compute_head_motion,
    compute_jaw_open,
    compute_mar,
    landmarks_to_bbox,
)


class _LM:
    """Minimal landmark stub matching MediaPipe's .x / .y / .z protocol."""

    __slots__ = ("x", "y", "z")

    def __init__(self, x: float, y: float, z: float = 0.0) -> None:
        self.x, self.y, self.z = x, y, z


class _BS:
    """Minimal blendshape stub matching MediaPipe's .category_name / .score protocol."""

    def __init__(self, name: str, score: float) -> None:
        self.category_name = name
        self.score = score


def _make_landmarks(overrides: dict) -> list:
    """Build a 478-landmark list with all points at (0.5, 0.5) except overrides.

    Caller passes ``{idx: (x, y)}`` to set specific landmark coordinates. Keeps
    tests focused on the indices each function actually reads.
    """
    landmarks = [_LM(0.5, 0.5) for _ in range(478)]
    for idx, (x, y) in overrides.items():
        landmarks[idx] = _LM(x, y)
    return landmarks


# ---------- compute_mar ----------


def test_compute_mar_zero_horizontal_returns_zero():
    """AC4: when horizontal distance is 0, return 0.0 (no division by zero)."""
    landmarks = _make_landmarks(
        {
            UPPER_LIP_IDX: (0.5, 0.4),
            LOWER_LIP_IDX: (0.5, 0.6),
            LEFT_CORNER_IDX: (0.5, 0.5),
            RIGHT_CORNER_IDX: (0.5, 0.5),
        }
    )
    assert compute_mar(landmarks, w=100, h=100) == 0.0


def test_compute_mar_open_mouth_positive_ratio():
    """Open mouth (vertical>0, horizontal>0) yields a positive ratio."""
    landmarks = _make_landmarks(
        {
            UPPER_LIP_IDX: (0.5, 0.4),
            LOWER_LIP_IDX: (0.5, 0.6),
            LEFT_CORNER_IDX: (0.4, 0.5),
            RIGHT_CORNER_IDX: (0.6, 0.5),
        }
    )
    mar = compute_mar(landmarks, w=100, h=100)
    # vertical = 20px, horizontal = 20px → MAR = 1.0
    assert mar == pytest.approx(1.0, rel=1e-6)


def test_compute_mar_scale_invariant():
    """MAR ratio is invariant to image size when proportions are fixed."""
    landmarks = _make_landmarks(
        {
            UPPER_LIP_IDX: (0.5, 0.45),
            LOWER_LIP_IDX: (0.5, 0.55),
            LEFT_CORNER_IDX: (0.4, 0.5),
            RIGHT_CORNER_IDX: (0.6, 0.5),
        }
    )
    mar_small = compute_mar(landmarks, w=100, h=100)
    mar_big = compute_mar(landmarks, w=1000, h=1000)
    assert mar_small == pytest.approx(mar_big, rel=1e-6)


# ---------- compute_jaw_open ----------


def test_compute_jaw_open_present_returns_score():
    blendshapes = [_BS("eyeBlinkLeft", 0.1), _BS("jawOpen", 0.42), _BS("mouthSmile", 0.05)]
    assert compute_jaw_open(blendshapes) == pytest.approx(0.42)


def test_compute_jaw_open_missing_returns_zero():
    blendshapes = [_BS("eyeBlinkLeft", 0.1), _BS("mouthSmile", 0.05)]
    assert compute_jaw_open(blendshapes) == 0.0


def test_compute_jaw_open_empty_returns_zero():
    assert compute_jaw_open([]) == 0.0


# ---------- compute_chin_y ----------


def test_compute_chin_y_normalized_by_bbox_height():
    """AC5: chin_y is normalized by face bbox height, not raw pixels.

    Construct a face spanning y ∈ [0.2, 0.8] (bbox_h_norm = 0.6, bbox_h_px=60 on h=100),
    with nose at y=0.5 and chin at y=0.7. Expected ratio = (70-50)/60 = 1/3.
    """
    landmarks = _make_landmarks(
        {
            # Two extreme y points define the bbox.
            0: (0.5, 0.2),
            1: (0.5, 0.8),
            DEFAULT_NOSE_IDX: (0.5, 0.5),
            DEFAULT_CHIN_IDX: (0.5, 0.7),
        }
    )
    chin_y = compute_chin_y(landmarks, w=100, h=100)
    assert chin_y == pytest.approx(1.0 / 3.0, rel=1e-6)


def test_compute_chin_y_custom_indices():
    """nose_idx / chin_idx defaults can be overridden."""
    landmarks = _make_landmarks(
        {
            0: (0.5, 0.0),
            1: (0.5, 1.0),  # bbox_h_norm = 1.0 → bbox_h_px = 100
            10: (0.5, 0.4),
            20: (0.5, 0.9),
        }
    )
    chin_y = compute_chin_y(landmarks, w=100, h=100, nose_idx=10, chin_idx=20)
    # (90 - 40) / 100 = 0.5
    assert chin_y == pytest.approx(0.5, rel=1e-6)


# ---------- compute_head_motion ----------


def test_compute_head_motion_no_prev_returns_zero():
    """Frame-0 edge case (or post-gap reset): None prev → 0.0 (not NaN)."""
    curr = _make_landmarks({DEFAULT_NOSE_IDX: (0.5, 0.5)})
    bbox = (0.2, 0.2, 0.8, 0.8)
    assert compute_head_motion(None, curr, bbox) == 0.0


def test_compute_head_motion_zero_displacement_returns_zero():
    """Identical frames produce no motion."""
    landmarks = _make_landmarks({DEFAULT_NOSE_IDX: (0.5, 0.5)})
    bbox = (0.2, 0.2, 0.8, 0.8)
    assert compute_head_motion(landmarks, landmarks, bbox) == 0.0


def test_compute_head_motion_nonzero_displacement_positive():
    """Nose moves by (0.1, 0.0); bbox_h=0.6 → motion = 0.1 / 0.6 ≈ 0.1667."""
    prev = _make_landmarks({DEFAULT_NOSE_IDX: (0.5, 0.5)})
    curr = _make_landmarks({DEFAULT_NOSE_IDX: (0.6, 0.5)})
    bbox = (0.2, 0.2, 0.8, 0.8)
    motion = compute_head_motion(prev, curr, bbox)
    assert motion == pytest.approx(0.1 / 0.6, rel=1e-6)


# ---------- landmarks_to_bbox helper ----------


def test_landmarks_to_bbox_returns_min_max():
    landmarks = [_LM(0.1, 0.2), _LM(0.4, 0.3), _LM(0.7, 0.9)]
    assert landmarks_to_bbox(landmarks) == (0.1, 0.2, 0.7, 0.9)
