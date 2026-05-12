"""Cross-engine agreement helpers (US-010)."""

from __future__ import annotations

from chewing.labels import read_window_csv
from chewing.types import Bout, Result, WindowLabel


def bucket_events_into_window_labels(events, n_windows: int, window_sec: float) -> list[str]:
    labels = ["rest"] * n_windows
    for event in events:
        idx = int(event.t_sec // window_sec)
        if 0 <= idx < n_windows:
            labels[idx] = "chewing"
    return labels


def _window_f1_labels(labels_a: list[str], labels_b: list[str]) -> float:
    n = min(len(labels_a), len(labels_b))
    labels_a = labels_a[:n]
    labels_b = labels_b[:n]
    tp = sum(1 for a, b in zip(labels_a, labels_b) if a == "chewing" and b == "chewing")
    fp = sum(1 for a, b in zip(labels_a, labels_b) if a != "chewing" and b == "chewing")
    fn = sum(1 for a, b in zip(labels_a, labels_b) if a == "chewing" and b != "chewing")
    if tp == 0:
        return 1.0 if (fp + fn) == 0 else 0.0
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    return 2 * precision * recall / (precision + recall)


def _bout_iou(a_bouts: list[Bout], b_bouts: list[Bout]) -> float:
    if not a_bouts and not b_bouts:
        return 1.0
    if not a_bouts or not b_bouts:
        return 0.0
    intersection = 0.0
    for a in a_bouts:
        for b in b_bouts:
            intersection += max(0.0, min(a.t_end, b.t_end) - max(a.t_start, b.t_start))
    a_total = sum(max(0.0, b.t_end - b.t_start) for b in a_bouts)
    b_total = sum(max(0.0, b.t_end - b.t_start) for b in b_bouts)
    union = a_total + b_total - intersection
    return intersection / union if union > 0 else 0.0


def _labels_from_result(result: Result, n_windows: int, window_sec: float) -> list[str]:
    if result.windows:
        return [w.label for w in result.windows[:n_windows]]
    return bucket_events_into_window_labels(result.events, n_windows, window_sec)


def cross_engine_agreement(
    result_a: Result,
    result_b: Result,
    window_sec: float = 1.0,
) -> dict:
    n_windows = max(len(result_a.windows), len(result_b.windows), 0)
    labels_a = _labels_from_result(result_a, n_windows, window_sec)
    labels_b = _labels_from_result(result_b, n_windows, window_sec)
    count_diff_pct = (
        abs(result_a.n_chews - result_b.n_chews)
        / max(result_a.n_chews, result_b.n_chews, 1)
        * 100.0
    )
    return {
        "window_f1": _window_f1_labels(labels_a, labels_b),
        "count_diff_pct": count_diff_pct,
        "bout_iou": _bout_iou(result_a.bouts, result_b.bouts),
    }


def _result_from_windows(windows: list[WindowLabel], engine_name: str) -> Result:
    n_chews = sum(1 for w in windows if w.label == "chewing")
    return Result(
        engine_name=engine_name,
        duration_sec=windows[-1].t_end if windows else 0.0,
        fps=1.0,
        face_detection_rate=0.0,
        n_chews=n_chews,
        chews_per_min=0.0,
        windows=windows,
    )


def cross_engine_agreement_from_csv(a_csv: str, b_csv: str) -> dict:
    return cross_engine_agreement(
        _result_from_windows(read_window_csv(a_csv), "a"),
        _result_from_windows(read_window_csv(b_csv), "b"),
    )
