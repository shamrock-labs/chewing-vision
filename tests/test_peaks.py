"""Unit tests for chewing.peaks (US-005)."""

from __future__ import annotations

import numpy as np
import pytest

from chewing.peaks import find_chew_peaks


def _make_sine(freq_hz: float, fps: float, duration_sec: float) -> np.ndarray:
    """Synthetic chewing signal — pure sine at the given frequency."""
    t = np.arange(0, duration_sec, 1.0 / fps)
    return np.sin(2 * np.pi * freq_hz * t)


def test_find_chew_peaks_returns_ndarray():
    peaks = find_chew_peaks(_make_sine(1.2, 30.0, 5.0), fps=30.0)
    assert isinstance(peaks, np.ndarray)
    assert peaks.dtype.kind == "i"


def test_find_chew_peaks_empty_signal_returns_empty():
    assert find_chew_peaks(np.array([]), fps=30.0).size == 0


def test_find_chew_peaks_flat_signal_returns_empty():
    """Zero-std signal yields no peaks (std-relative prominence guard)."""
    assert find_chew_peaks(np.zeros(300), fps=30.0).size == 0


def test_find_chew_peaks_all_nan_returns_empty():
    assert find_chew_peaks(np.full(300, np.nan), fps=30.0).size == 0


def test_find_chew_peaks_fps_invariance():
    """AC5: a 1.2 Hz sine sampled at 30 fps vs 60 fps yields peak counts within 10%.

    Also pins the absolute count near the true value (30s × 1.2Hz = 36) to
    catch the "both broken in the same way" failure where ratio comparison
    trivially holds (e.g. both return zero).
    """
    sine_30 = _make_sine(1.2, 30.0, 30.0)
    sine_60 = _make_sine(1.2, 60.0, 30.0)

    peaks_30 = find_chew_peaks(sine_30, fps=30.0)
    peaks_60 = find_chew_peaks(sine_60, fps=60.0)

    n30, n60 = len(peaks_30), len(peaks_60)
    assert n30 > 0 and n60 > 0
    relative_diff = abs(n30 - n60) / max(n30, n60)
    assert relative_diff < 0.10, f"30fps={n30}, 60fps={n60}, diff={relative_diff:.3f}"
    # 30 sec × 1.2 Hz = 36 true peaks; allow ±2 for boundary effects.
    assert abs(n30 - 36) <= 2, f"30fps count {n30} far from expected 36"
    assert abs(n60 - 36) <= 2, f"60fps count {n60} far from expected 36"


def test_find_chew_peaks_prominence_filters_noise():
    """Tiny-amplitude noise on a flat baseline should produce few peaks."""
    rng = np.random.default_rng(0)
    noise = rng.normal(0, 1e-6, 300)
    peaks = find_chew_peaks(noise, fps=30.0, prominence_std=0.5)
    # Pure noise with std ≈ 1e-6 still has nonzero std, but prominence threshold
    # = 0.5 * 1e-6 leaves room for spurious peaks. We don't assert zero; we
    # assert "not chewing-scale" — count is well below the true 36 of a 1.2Hz signal.
    assert len(peaks) < 36
