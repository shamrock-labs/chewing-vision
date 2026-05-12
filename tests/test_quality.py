"""Unit tests for chewing.quality (US-014)."""

from __future__ import annotations

import pytest

from chewing.quality import compute_frame_quality, compute_window_quality


# ---------- compute_frame_quality ----------


def test_compute_frame_quality_no_face_returns_zero():
    assert compute_frame_quality(False, landmark_confidence=1.0, head_motion=0.0) == 0.0


def test_compute_frame_quality_full_quality_with_zero_motion():
    """face_found + confidence=1 + no motion → quality=1."""
    assert compute_frame_quality(True, landmark_confidence=1.0, head_motion=0.0) == 1.0


def test_compute_frame_quality_high_motion_clamps_to_zero():
    """head_motion ≥ 1 fully cancels quality."""
    assert compute_frame_quality(True, landmark_confidence=1.0, head_motion=1.5) == 0.0


def test_compute_frame_quality_partial_motion_partial_penalty():
    """head_motion=0.5 yields quality 0.5 (when confidence=1)."""
    q = compute_frame_quality(True, landmark_confidence=1.0, head_motion=0.5)
    assert q == pytest.approx(0.5, abs=1e-9)


def test_compute_frame_quality_confidence_scales_output():
    """landmark_confidence multiplies the motion-penalty."""
    q = compute_frame_quality(True, landmark_confidence=0.4, head_motion=0.5)
    # 0.4 * (1 - 0.5) = 0.2
    assert q == pytest.approx(0.2, abs=1e-9)


def test_compute_frame_quality_bounds():
    """Always in [0, 1] across reasonable inputs."""
    for lc in [0.0, 0.5, 1.0]:
        for hm in [0.0, 0.3, 1.0, 5.0]:
            q = compute_frame_quality(True, landmark_confidence=lc, head_motion=hm)
            assert 0.0 <= q <= 1.0, (lc, hm, q)


# ---------- compute_window_quality ----------


def test_compute_window_quality_empty_returns_zero():
    """Pin the empty-input contract (degenerate input → degenerate output)."""
    assert compute_window_quality([], face_found_rate=1.0) == 0.0


def test_compute_window_quality_full_quality():
    q = compute_window_quality([1.0, 1.0, 1.0], face_found_rate=1.0)
    assert q == pytest.approx(1.0, abs=1e-9)


def test_compute_window_quality_low_face_rate_drops_below_half():
    """AC: face_found_rate < 0.6 drops window quality below 0.5.

    Concrete case: face_found_rate=0.3, qualities=[1.0]*5 → 0.3 * 1.0 = 0.3 < 0.5.
    """
    q = compute_window_quality([1.0] * 5, face_found_rate=0.3)
    assert q < 0.5
    assert q == pytest.approx(0.3, abs=1e-9)


def test_compute_window_quality_bounds():
    """Always in [0, 1]."""
    for rate in [0.0, 0.5, 1.0]:
        for mean_q in [0.0, 0.5, 1.0]:
            q = compute_window_quality([mean_q] * 4, face_found_rate=rate)
            assert 0.0 <= q <= 1.0, (rate, mean_q, q)
