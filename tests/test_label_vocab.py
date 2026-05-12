"""Unit tests for chewing.labels (US-015)."""

from __future__ import annotations

import csv
import pytest

from chewing.labels import LABEL_VOCAB, write_window_csv
from chewing.types import WindowLabel


EXPECTED_VOCAB = [
    "chewing",
    "rest",
    "speaking",
    "drinking",
    "occluded",
    "bad_face",
    "unknown",
]


def test_label_vocab_matches_spec_exactly():
    """SPEC §6.2 + prd.json AC1: exact 7 values, exact order."""
    assert LABEL_VOCAB == EXPECTED_VOCAB


def test_write_window_csv_accepts_all_seven_labels(tmp_path):
    """AC2: write_window_csv accepts any LABEL_VOCAB value."""
    windows = [
        WindowLabel(t_start=float(i), t_end=float(i + 1), label=lab,
                    mar_mean=0.1, jaw_open_mean=0.2, confidence=0.5)
        for i, lab in enumerate(LABEL_VOCAB)
    ]
    out = tmp_path / "windows.csv"
    write_window_csv(str(out), windows, engine="ours")

    with open(out, newline="") as f:
        rows = list(csv.reader(f))
    header = rows[0]
    data = rows[1:]
    assert header[2] == "label"
    assert [r[2] for r in data] == EXPECTED_VOCAB


def test_write_window_csv_rejects_unknown_label(tmp_path):
    """AC2: unknown label raises ValueError."""
    windows = [
        WindowLabel(t_start=0.0, t_end=1.0, label="totally_invalid",
                    mar_mean=0.1, jaw_open_mean=0.2, confidence=0.5)
    ]
    out = tmp_path / "windows.csv"
    with pytest.raises(ValueError, match="Unknown window label"):
        write_window_csv(str(out), windows, engine="ours")


def test_write_window_csv_rejects_mixed_known_and_unknown(tmp_path):
    """ValueError fires even when only one row is invalid."""
    windows = [
        WindowLabel(t_start=0.0, t_end=1.0, label="chewing",
                    mar_mean=0.1, jaw_open_mean=0.2, confidence=0.5),
        WindowLabel(t_start=1.0, t_end=2.0, label="bogus",
                    mar_mean=0.1, jaw_open_mean=0.2, confidence=0.5),
    ]
    out = tmp_path / "windows.csv"
    with pytest.raises(ValueError):
        write_window_csv(str(out), windows, engine="ours")
