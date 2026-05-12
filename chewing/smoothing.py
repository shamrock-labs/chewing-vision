"""Signal smoothing dispatcher (US-005, SPEC §8.5).

``apply_smoothing(signal, method, **kwargs)`` routes to one of six smoothing
backends. fps-aware ``"default"`` reproduces the previous OursEngine inline
behavior verbatim so US-002 regression remains byte-identical after extraction.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import butter, filtfilt, medfilt, savgol_filter


# fps-aware default smoothing target (~0.4s window).
DEFAULT_WINDOW_SEC = 0.4


def _default_savgol(signal: np.ndarray, fps: float) -> np.ndarray:
    """Original OursEngine `_smooth` body — moved verbatim for regression safety.

    Window length is forced odd and clamped to signal length; falls through to
    the raw signal when too short to smooth.
    """
    if len(signal) < 5:
        return signal
    target = max(5, int(round(fps * DEFAULT_WINDOW_SEC)))
    if target % 2 == 0:
        target += 1
    if target > len(signal):
        target = len(signal) if len(signal) % 2 == 1 else len(signal) - 1
    if target < 5:
        return signal
    polyorder = min(3, target - 1)
    return savgol_filter(signal, window_length=target, polyorder=polyorder)


def apply_smoothing(signal: np.ndarray, method: str, **kwargs) -> np.ndarray:
    """Smooth ``signal`` using the named ``method``.

    Supported methods (SPEC §8.5 list + 'default'):

    * ``savgol`` — kwargs: ``window_length`` (odd int), ``polyorder`` (int)
    * ``gaussian`` — kwargs: ``sigma`` (float)
    * ``moving_average`` — kwargs: ``window`` (int)
    * ``butterworth`` — kwargs: ``cutoff_hz`` (float), ``fs`` (float),
      ``order`` (int, default 4)
    * ``median`` — kwargs: ``kernel_size`` (odd int)
    * ``default`` — kwargs: ``fps`` (float); fps-aware Savitzky-Golay used by
      OursEngine. Behavior frozen to match pre-extraction `_smooth` exactly.

    Raises ``ValueError`` for unknown method names.
    """
    arr = np.asarray(signal)
    if method == "savgol":
        return savgol_filter(
            arr,
            window_length=int(kwargs["window_length"]),
            polyorder=int(kwargs["polyorder"]),
        )
    if method == "gaussian":
        return gaussian_filter1d(arr, sigma=float(kwargs["sigma"]))
    if method == "moving_average":
        w = int(kwargs["window"])
        return np.convolve(arr, np.ones(w) / w, mode="same")
    if method == "butterworth":
        fs = float(kwargs["fs"])
        cutoff = float(kwargs["cutoff_hz"])
        order = int(kwargs.get("order", 4))
        b, a = butter(order, cutoff, btype="low", fs=fs)
        return filtfilt(b, a, arr)
    if method == "median":
        return medfilt(arr, kernel_size=int(kwargs["kernel_size"]))
    if method == "default":
        return _default_savgol(arr, fps=float(kwargs["fps"]))
    raise ValueError(f"Unknown smoothing method: {method!r}")
