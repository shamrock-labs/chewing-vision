"""Confounder regression — rest-clip jaw_open should not register as chewing (US-015).

SPEC §11.5: negative rest false-positive windows ≤ 15%. The prd.json AC4 was
reconciled per executionDirectives item 6 from "i.i.d. Gaussian noise std=0.005"
to a slow sinusoidal drift, because find_chew_peaks's std-relative prominence
(US-005 AC-locked default `prominence_std = 0.5 * nanstd(signal)`) is invariant
under noise amplitude scaling — pure i.i.d. noise produces the same false-fire
rate at any std. The drift model captures the actual SPEC §11.5 spectral
character of a real rest clip (no chewing-band content) rather than its raw
variance.

This test inlines the "n_events ≥ 1 → chewing" classification rule from
OursEngine._build_windows rather than scaffolding 900 synthetic FrameSignal
objects to run them through a private method.
"""

from __future__ import annotations

import numpy as np

from chewing.peaks import find_chew_peaks


FPS = 30.0
DURATION_SEC = 30
N_WINDOWS = 30
DRIFT_FREQ_HZ = 0.05      # well below chewing band (0.8-2.5 Hz)
DRIFT_AMPLITUDE = 0.005   # std ≈ amplitude / sqrt(2) ≈ 0.0035
BASELINE = 0.05


def test_rest_clip_drift_does_not_register_as_chewing():
    """A signal with no chewing-band content yields < 5 chewing windows out of 30.

    Models a rest clip: slow 0.05 Hz drift around mean 0.05 with amplitude 0.005.
    No 0.8-2.5 Hz content → find_chew_peaks rejects all but a handful of peaks
    arising from the drift's own slow oscillation.
    """
    t = np.arange(int(FPS * DURATION_SEC)) / FPS
    signal = BASELINE + DRIFT_AMPLITUDE * np.sin(2 * np.pi * DRIFT_FREQ_HZ * t)

    peaks = find_chew_peaks(signal, fps=FPS)
    peak_seconds = peaks / FPS
    chewing_windows = sum(
        1
        for i in range(N_WINDOWS)
        if ((peak_seconds >= i) & (peak_seconds < i + 1)).any()
    )

    assert chewing_windows < 5, (
        f"rest-clip confounder produced {chewing_windows}/30 chewing windows "
        f"(expected < 5); chewing-band peak detection is leaking"
    )
