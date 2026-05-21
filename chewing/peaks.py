"""Chewing peak detector (US-005, SPEC §8.6).

Public ``find_chew_peaks`` extracted from OursEngine. fps-relative distance
and signal-std-relative prominence — never pixel-absolute (SPEC §18).
"""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, filtfilt, find_peaks

_BANDPASS_ORDER = 4
# filtfilt requires at least 3*padlen samples; padlen defaults to 3*(max filter order)
_FILTFILT_MIN_SAMPLES = 3 * _BANDPASS_ORDER * 3 + 1


def _bandpass(arr: np.ndarray, fps: float, lo: float, hi: float) -> np.ndarray:
    """Zero-phase Butterworth bandpass [lo, hi] Hz. Returns arr unchanged when
    the signal is too short or the cutoffs exceed the Nyquist frequency."""
    nyq = fps / 2.0
    if lo <= 0 or hi >= nyq or lo >= hi or len(arr) < _FILTFILT_MIN_SAMPLES:
        return arr
    b, a = butter(_BANDPASS_ORDER, [lo / nyq, hi / nyq], btype="band")
    return filtfilt(b, a, arr)


def find_chew_peaks(
    signal: np.ndarray,
    fps: float,
    min_freq: float = 0.0,
    max_freq: float = 2.5,
    prominence_std: float = 0.5,
) -> np.ndarray:
    """Locate chewing peaks in a 1-D signal.

    When ``min_freq > 0``, applies a zero-phase Butterworth bandpass
    [min_freq, max_freq] Hz before peak picking. This removes low-frequency
    head-motion drift and high-frequency noise, leaving only chewing-band
    oscillations. ``distance`` and ``prominence`` are computed on the
    filtered signal so thresholds remain signal-relative.
    """
    arr = np.asarray(signal, dtype=float)
    if len(arr) < 5 or np.all(np.isnan(arr)):
        return np.array([], dtype=int)

    if min_freq > 0:
        arr = _bandpass(arr, fps, min_freq, max_freq)

    std = float(np.nanstd(arr))
    if std <= 0:
        return np.array([], dtype=int)
    distance = max(1, int(fps / max_freq))
    prominence = prominence_std * std
    peaks, _props = find_peaks(arr, distance=distance, prominence=prominence)
    return peaks
