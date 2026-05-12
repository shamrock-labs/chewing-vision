"""Chewing peak detector (US-005, SPEC §8.6).

Public ``find_chew_peaks`` extracted from OursEngine. fps-relative distance
and signal-std-relative prominence — never pixel-absolute (SPEC §18).
"""

from __future__ import annotations

import numpy as np
from scipy.signal import find_peaks


def find_chew_peaks(
    signal: np.ndarray,
    fps: float,
    min_freq: float = 0.8,
    max_freq: float = 2.5,
    prominence_std: float = 0.5,
) -> np.ndarray:
    """Locate chewing peaks in a 1-D signal.

    Uses ``distance = int(fps / max_freq)`` and ``prominence = prominence_std *
    np.nanstd(signal)`` per SPEC §8.6. Returns peak indices into ``signal``.

    ``min_freq`` is reserved for future spectral filtering (e.g. bandpass
    around 0.8-2.5 Hz before peak picking); current impl is time-domain only.
    """
    _ = min_freq  # reserved; documented in docstring above.
    arr = np.asarray(signal)
    if len(arr) < 5 or np.all(np.isnan(arr)):
        return np.array([], dtype=int)
    std = float(np.nanstd(arr))
    if std <= 0:
        return np.array([], dtype=int)
    distance = max(1, int(fps / max_freq))
    prominence = prominence_std * std
    peaks, _props = find_peaks(arr, distance=distance, prominence=prominence)
    return peaks
