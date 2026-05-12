"""Evaluation helpers for auto-vs-human chewing labels (US-010)."""

from __future__ import annotations

import csv
from typing import Iterable

from chewing.labels import read_window_csv
from chewing.types import WindowLabel


EXCLUDED_QUALITY_LABELS = {"bad_face", "occluded", "unknown"}


def _align_windows(
    auto: list[WindowLabel], human: list[WindowLabel]
) -> list[tuple[WindowLabel, WindowLabel]]:
    if len(auto) != len(human):
        raise ValueError("auto and human CSVs must have the same window count")
    pairs = list(zip(auto, human))
    for a, h in pairs:
        if a.t_start != h.t_start or a.t_end != h.t_end:
            raise ValueError("auto and human CSVs must use the same window grid")
    return pairs


def _binary_metrics(pairs: Iterable[tuple[WindowLabel, WindowLabel]]) -> dict:
    tp = fp = fn = tn = 0
    for auto, human in pairs:
        auto_pos = auto.label == "chewing"
        human_pos = human.label == "chewing"
        if auto_pos and human_pos:
            tp += 1
        elif auto_pos and not human_pos:
            fp += 1
        elif not auto_pos and human_pos:
            fn += 1
        else:
            tn += 1

    precision = tp / (tp + fp) if (tp + fp) else 1.0
    recall = tp / (tp + fn) if (tp + fn) else 1.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall)
        else 0.0
    )
    return {
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "confusion": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
    }


def _read_event_times(path: str) -> list[float]:
    times: list[float] = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            times.append(float(row["t_sec"]))
    return sorted(times)


def _event_f1(auto_times: list[float], human_times: list[float], tolerance_sec: float) -> float:
    matched_human: set[int] = set()
    tp = 0
    for auto_t in auto_times:
        best_idx = None
        best_dt = tolerance_sec
        for i, human_t in enumerate(human_times):
            if i in matched_human:
                continue
            dt = abs(auto_t - human_t)
            if dt <= best_dt:
                best_dt = dt
                best_idx = i
        if best_idx is not None:
            matched_human.add(best_idx)
            tp += 1
    fp = len(auto_times) - tp
    fn = len(human_times) - tp
    precision = tp / (tp + fp) if (tp + fp) else 1.0
    recall = tp / (tp + fn) if (tp + fn) else 1.0
    return (
        2 * precision * recall / (precision + recall)
        if (precision + recall)
        else 0.0
    )


def compare_window_labels(
    auto_csv: str,
    human_csv: str,
    *,
    auto_events_csv: str | None = None,
    human_events_csv: str | None = None,
    tolerance_sec: float = 0.3,
) -> dict:
    """Compare auto labels against human labels.

    The top-level f1/precision/recall/confusion keys mirror the ``all``
    metrics to satisfy US-010 AC1. The nested ``all`` and
    ``quality_filtered`` blocks satisfy AC6; filtered metrics exclude any
    pair where either side is bad_face, occluded, or unknown.
    """

    pairs = _align_windows(read_window_csv(auto_csv), read_window_csv(human_csv))
    all_metrics = _binary_metrics(pairs)
    filtered_pairs = [
        (auto, human)
        for auto, human in pairs
        if auto.label not in EXCLUDED_QUALITY_LABELS
        and human.label not in EXCLUDED_QUALITY_LABELS
    ]
    filtered_metrics = _binary_metrics(filtered_pairs)

    event_f1_300ms = None
    if auto_events_csv and human_events_csv:
        event_f1_300ms = _event_f1(
            _read_event_times(auto_events_csv),
            _read_event_times(human_events_csv),
            tolerance_sec,
        )

    return {
        "f1": all_metrics["f1"],
        "precision": all_metrics["precision"],
        "recall": all_metrics["recall"],
        "confusion": all_metrics["confusion"],
        "event_f1_300ms": event_f1_300ms,
        "all": all_metrics,
        "quality_filtered": filtered_metrics,
    }
