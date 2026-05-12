"""Static signal visualization for chewing analysis results."""

from __future__ import annotations

import math
import os
from typing import Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from chewing.types import Result


def _nearest_value(t_sec: float, times: Sequence[float], values: Sequence[float]) -> float:
    best_idx = -1
    best_dt = math.inf
    for i, t in enumerate(times):
        if np.isnan(values[i]):
            continue
        dt = abs(t - t_sec)
        if dt < best_dt:
            best_dt = dt
            best_idx = i
    return values[best_idx] if best_idx >= 0 else np.nan


def plot_signals(result: Result, output_path: str) -> None:
    """Write a two-panel PNG showing MAR, jawOpen, peaks, and chewing windows."""

    times = [f.t_sec for f in result.frames]
    mar = [f.mar if f.mar is not None else np.nan for f in result.frames]
    jaw_open = [
        f.jaw_open if f.jaw_open is not None else np.nan for f in result.frames
    ]

    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    ax_mar, ax_jaw = axes

    for window in result.windows:
        if window.label != "chewing":
            continue
        for ax in axes:
            ax.axvspan(
                window.t_start,
                window.t_end,
                color="#22c55e",
                alpha=0.2,
            )

    ax_mar.plot(times, mar)
    ax_mar.set_ylabel("MAR")
    ax_jaw.plot(times, jaw_open)
    ax_jaw.set_ylabel("jawOpen")
    ax_jaw.set_xlabel("time (sec)")

    for event in result.events:
        if event.source_signal == "mar":
            y = _nearest_value(event.t_sec, times, mar)
            if not np.isnan(y):
                ax_mar.plot(event.t_sec, y, "ro")
        elif event.source_signal == "jaw_open":
            y = _nearest_value(event.t_sec, times, jaw_open)
            if not np.isnan(y):
                ax_jaw.plot(event.t_sec, y, "ro")

    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=120)
    plt.close(fig)
