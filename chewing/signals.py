"""Per-frame signal extraction primitives (US-004, SPEC §8.3-8.4).

Pure functions consumed by OursEngine. Each function is face-relative — no
pixel-absolute thresholds escape this module (SPEC §18).

Landmark inputs follow MediaPipe Face Landmarker convention: a list of objects
exposing ``.x``, ``.y`` (and optionally ``.z``) as floats in [0, 1] relative to
the source image. Blendshape inputs expose ``.category_name`` and ``.score``.
"""

from __future__ import annotations

import math
from typing import Iterable, Optional, Sequence, Tuple

import numpy as np


# Default 478-landmark FaceMesh indices used by MediaPipe Face Landmarker.
UPPER_LIP_IDX = 13
LOWER_LIP_IDX = 14
LEFT_CORNER_IDX = 78
RIGHT_CORNER_IDX = 308
DEFAULT_NOSE_IDX = 4
DEFAULT_CHIN_IDX = 152


def compute_mar(landmarks, w: int, h: int) -> float:
    """Mouth Aspect Ratio = vertical lip distance / horizontal lip distance.

    Numerator and denominator share pixel units, so the ratio is already
    face-scale-invariant; no separate normalization needed. Returns 0.0 when
    horizontal distance is zero (US-004 AC4).
    """
    p_up = np.array([landmarks[UPPER_LIP_IDX].x * w, landmarks[UPPER_LIP_IDX].y * h])
    p_dn = np.array([landmarks[LOWER_LIP_IDX].x * w, landmarks[LOWER_LIP_IDX].y * h])
    p_l = np.array([landmarks[LEFT_CORNER_IDX].x * w, landmarks[LEFT_CORNER_IDX].y * h])
    p_r = np.array([landmarks[RIGHT_CORNER_IDX].x * w, landmarks[RIGHT_CORNER_IDX].y * h])
    horizontal = float(np.linalg.norm(p_l - p_r))
    if horizontal <= 0:
        return 0.0
    return float(np.linalg.norm(p_up - p_dn) / horizontal)


YAW_CLIP_DEG = 45.0


def compute_mar_yaw_corrected(
    landmarks,
    w: int,
    h: int,
    transformation_matrix,
    yaw_clip_deg: float = YAW_CLIP_DEG,
) -> Optional[float]:
    """MAR with cos(yaw) correction for camera-yaw-induced horizontal foreshortening.

    For pure yaw about Y: observed_horizontal = true_horizontal * cos(yaw),
    so corrected_MAR = vertical / true_horizontal = observed_MAR * cos(yaw).

    Returns None when |yaw| > yaw_clip_deg — caller should fall back to jaw_open.
    """
    data = transformation_matrix.data
    R = np.array(data, dtype=float).reshape(4, 4)
    yaw_rad = math.atan2(R[0, 2], R[2, 2])
    if abs(math.degrees(yaw_rad)) > yaw_clip_deg:
        return None
    cos_yaw = math.cos(yaw_rad)
    if cos_yaw <= 0:
        return None

    p_up = np.array([landmarks[UPPER_LIP_IDX].x * w, landmarks[UPPER_LIP_IDX].y * h])
    p_dn = np.array([landmarks[LOWER_LIP_IDX].x * w, landmarks[LOWER_LIP_IDX].y * h])
    p_l = np.array([landmarks[LEFT_CORNER_IDX].x * w, landmarks[LEFT_CORNER_IDX].y * h])
    p_r = np.array([landmarks[RIGHT_CORNER_IDX].x * w, landmarks[RIGHT_CORNER_IDX].y * h])
    vertical = float(np.linalg.norm(p_up - p_dn))
    observed_horizontal = float(np.linalg.norm(p_l - p_r))
    true_horizontal = observed_horizontal / cos_yaw
    if true_horizontal <= 0:
        return 0.0
    return vertical / true_horizontal


def compute_jaw_open(blendshapes: Iterable) -> float:
    """MediaPipe blendshape ``jawOpen`` score in [0, 1] (0.0 if absent)."""
    for b in blendshapes:
        if b.category_name == "jawOpen":
            return float(b.score)
    return 0.0


def compute_chin_y(
    landmarks,
    w: int,
    h: int,
    nose_idx: int = DEFAULT_NOSE_IDX,
    chin_idx: int = DEFAULT_CHIN_IDX,
) -> float:
    """Chin-to-nose Y displacement, normalized by face bbox height (SPEC §8.4).

    Returns a dimensionless ratio: (chin_y_px - nose_y_px) / bbox_h_px, where
    bbox_h_px is computed from the y-extent of the supplied landmarks. Never
    pixel-absolute (US-004 AC5).
    """
    ys = [lm.y * h for lm in landmarks]
    bbox_h = max(1e-6, max(ys) - min(ys))
    chin_y = landmarks[chin_idx].y * h
    nose_y = landmarks[nose_idx].y * h
    return float((chin_y - nose_y) / bbox_h)


def compute_head_motion(
    prev_landmarks,
    curr_landmarks,
    face_bbox: Sequence[float],
) -> float:
    """Per-frame nose-center displacement normalized by face bbox height.

    Args:
        prev_landmarks: Previous-frame landmark list, or None on frame 0 /
            after a face-missing gap. Returning 0.0 in that case keeps the
            quality formula in [0, 1] (US-002 frame-0 edge case).
        curr_landmarks: Current-frame landmark list.
        face_bbox: ``(xmin, ymin, xmax, ymax)`` in normalized image coordinates
            ([0, 1]). Caller is responsible for computing this — usually
            ``(min/max of landmark.x, min/max of landmark.y)``.

    Returns:
        ``hypot(dx, dy) / bbox_height`` with both terms in normalized image
        space. Anisotropic with respect to image aspect ratio — acceptable as
        a weak head-motion proxy. US-014 may swap this for a tighter formula.
    """
    if prev_landmarks is None:
        return 0.0
    dx = curr_landmarks[DEFAULT_NOSE_IDX].x - prev_landmarks[DEFAULT_NOSE_IDX].x
    dy = curr_landmarks[DEFAULT_NOSE_IDX].y - prev_landmarks[DEFAULT_NOSE_IDX].y
    bbox_h = max(1e-6, float(face_bbox[3]) - float(face_bbox[1]))
    return float(np.hypot(dx, dy) / bbox_h)


def landmarks_to_bbox(landmarks) -> Tuple[float, float, float, float]:
    """Helper for callers — bounding box of a normalized-coord landmark list."""
    xs = [lm.x for lm in landmarks]
    ys = [lm.y for lm in landmarks]
    return (min(xs), min(ys), max(xs), max(ys))
