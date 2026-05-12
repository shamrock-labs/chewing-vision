import csv

import pytest

from chewing.compare import cross_engine_agreement
from chewing.eval import compare_window_labels
from chewing.labels import write_window_csv
from chewing.types import Bout, Result, WindowLabel


def _windows(labels):
    return [
        WindowLabel(float(i), float(i + 1), label, confidence=1.0, quality=1.0)
        for i, label in enumerate(labels)
    ]


def _write_windows(path, labels):
    write_window_csv(str(path), _windows(labels), "test")


def _write_events(path, times):
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["t_sec", "frame_index", "signal", "value", "confidence", "engine", "side"])
        for i, t in enumerate(times):
            writer.writerow([t, i, "jaw_open", 1.0, 1.0, "test", ""])


def test_eval_self_compare_perfect(tmp_path):
    labels = ["chewing", "rest", "chewing", "rest"]
    csv_path = tmp_path / "labels.csv"
    _write_windows(csv_path, labels)

    result = compare_window_labels(str(csv_path), str(csv_path))

    assert result["f1"] == 1.0
    assert result["precision"] == 1.0
    assert result["recall"] == 1.0
    assert result["confusion"] == {"tp": 2, "fp": 0, "fn": 0, "tn": 2}


def test_eval_k_disagreements(tmp_path):
    human = ["chewing"] * 5 + ["rest"] * 5
    auto = ["rest", "rest", "chewing", "chewing", "chewing", "chewing", "rest", "rest", "rest", "rest"]
    auto_csv = tmp_path / "auto.csv"
    human_csv = tmp_path / "human.csv"
    _write_windows(auto_csv, auto)
    _write_windows(human_csv, human)

    result = compare_window_labels(str(auto_csv), str(human_csv))

    assert result["confusion"] == {"tp": 3, "fp": 1, "fn": 2, "tn": 4}
    assert result["precision"] == pytest.approx(0.75)
    assert result["recall"] == pytest.approx(0.6)
    assert result["f1"] == pytest.approx(2 * 0.75 * 0.6 / (0.75 + 0.6))


def test_eval_quality_filtered_excludes_bad_face(tmp_path):
    auto_csv = tmp_path / "auto.csv"
    human_csv = tmp_path / "human.csv"
    _write_windows(auto_csv, ["chewing", "rest", "bad_face", "occluded"])
    _write_windows(human_csv, ["chewing", "rest", "bad_face", "occluded"])

    result = compare_window_labels(str(auto_csv), str(human_csv))

    assert result["all"]["confusion"]["tn"] == 3
    assert result["quality_filtered"]["confusion"] == {"tp": 1, "fp": 0, "fn": 0, "tn": 1}
    assert result["quality_filtered"]["f1"] == 1.0


def test_event_f1_300ms_basic(tmp_path):
    labels = tmp_path / "labels.csv"
    _write_windows(labels, ["chewing"])
    auto_events = tmp_path / "auto_events.csv"
    human_events = tmp_path / "human_events.csv"
    _write_events(auto_events, [1.0, 2.0])
    _write_events(human_events, [1.2, 3.0])

    result = compare_window_labels(
        str(labels),
        str(labels),
        auto_events_csv=str(auto_events),
        human_events_csv=str(human_events),
    )

    assert result["event_f1_300ms"] == pytest.approx(0.5)


def test_cross_engine_agreement_count_diff():
    a = Result("a", 10.0, 1.0, 0.0, 80, 0.0)
    b = Result("b", 10.0, 1.0, 0.0, 100, 0.0)

    result = cross_engine_agreement(a, b)

    assert result["count_diff_pct"] == pytest.approx(20.0)


def test_cross_engine_agreement_bout_iou():
    a = Result(
        "a",
        10.0,
        1.0,
        0.0,
        0,
        0.0,
        bouts=[Bout(0.0, 4.0, 4, 60.0, 1.0)],
    )
    b = Result(
        "b",
        10.0,
        1.0,
        0.0,
        0,
        0.0,
        bouts=[Bout(2.0, 6.0, 4, 60.0, 1.0)],
    )

    result = cross_engine_agreement(a, b)

    assert result["bout_iou"] == pytest.approx(2.0 / 6.0)
