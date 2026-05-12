"""I/O tests for chewing.labels (US-006).

Covers AC7 round-trip explicitly; the rest are cheap header/contract sanity
checks that catch silent schema drift in future stories.
"""

from __future__ import annotations

import csv
import json
import pytest

from chewing.labels import (
    BOUT_HEADER,
    EVENT_HEADER,
    FRAME_HEADER,
    LABEL_VOCAB,
    WINDOW_HEADER,
    read_window_csv,
    write_bouts_csv,
    write_event_csv,
    write_frame_signals_csv,
    write_summary_json,
    write_window_csv,
)
from chewing.types import Bout, ChewEvent, FrameSignal, Result, WindowLabel


# ---------- AC7 round-trip ----------


def test_window_csv_round_trip_preserves_values(tmp_path):
    """write_window_csv → read_window_csv yields the same list within float tolerance."""
    windows = [
        WindowLabel(
            t_start=float(i),
            t_end=float(i + 1),
            label=lab,
            mar_mean=0.10 + 0.01 * i,
            jaw_open_mean=0.20 + 0.01 * i,
            confidence=0.5 + 0.05 * i,
            quality=0.7 + 0.02 * i,
            n_events=i,
        )
        for i, lab in enumerate(LABEL_VOCAB)
    ]
    out = tmp_path / "windows.csv"
    write_window_csv(str(out), windows, engine="ours")
    recovered = read_window_csv(str(out))

    assert len(recovered) == len(windows)
    for orig, got in zip(windows, recovered):
        assert orig.label == got.label
        assert orig.n_events == got.n_events
        assert got.t_start == pytest.approx(orig.t_start, abs=1e-9)
        assert got.t_end == pytest.approx(orig.t_end, abs=1e-9)
        assert got.mar_mean == pytest.approx(orig.mar_mean, abs=1e-9)
        assert got.jaw_open_mean == pytest.approx(orig.jaw_open_mean, abs=1e-9)
        assert got.confidence == pytest.approx(orig.confidence, abs=1e-9)
        assert got.quality == pytest.approx(orig.quality, abs=1e-9)


def test_write_window_csv_rejects_unknown_label_still(tmp_path):
    """US-015 carryover: unknown labels still raise ValueError after US-006 rewrite."""
    windows = [
        WindowLabel(t_start=0.0, t_end=1.0, label="bogus",
                    mar_mean=0.1, jaw_open_mean=0.2, confidence=0.5)
    ]
    with pytest.raises(ValueError, match="Unknown window label"):
        write_window_csv(str(tmp_path / "w.csv"), windows, engine="ours")


# ---------- header sanity (cheap contract regression) ----------


def _read_header(path) -> list:
    with open(path, newline="") as f:
        return next(csv.reader(f))


def test_frame_csv_header_matches_spec_7_1(tmp_path):
    out = tmp_path / "frames.csv"
    write_frame_signals_csv(str(out), [], engine="ours")
    assert _read_header(out) == FRAME_HEADER == [
        "t_sec", "frame_index", "face_found", "mar", "jaw_open", "chin_y", "head_motion", "quality"
    ]


def test_event_csv_header_matches_spec_7_2(tmp_path):
    out = tmp_path / "events.csv"
    write_event_csv(str(out), [], engine="ours")
    assert _read_header(out) == EVENT_HEADER == [
        "t_sec", "frame_index", "signal", "value", "confidence", "engine", "side"
    ]


def test_window_csv_header_matches_spec_7_3(tmp_path):
    out = tmp_path / "windows.csv"
    write_window_csv(str(out), [], engine="ours")
    assert _read_header(out) == WINDOW_HEADER == [
        "t_start", "t_end", "label", "confidence", "quality", "n_events", "engine",
        "mar_mean", "jaw_open_mean"
    ]


def test_bouts_csv_header_matches_spec_7_4(tmp_path):
    out = tmp_path / "bouts.csv"
    write_bouts_csv(str(out), [], engine="ours")
    assert _read_header(out) == BOUT_HEADER == [
        "t_start", "t_end", "n_events", "chews_per_min", "confidence", "engine"
    ]


# ---------- frame None / NaN handling ----------


def test_frame_csv_blank_cells_for_no_face(tmp_path):
    """face_found=False FrameSignal yields empty cells for optional signal fields."""
    frames = [
        FrameSignal(t_sec=0.0, frame_index=0, face_found=False,
                    mar=None, jaw_open=None, chin_y=None, head_motion=None, quality=0.0),
        FrameSignal(t_sec=0.03, frame_index=1, face_found=True,
                    mar=0.15, jaw_open=0.20, chin_y=0.30, head_motion=0.0, quality=1.0),
    ]
    out = tmp_path / "frames.csv"
    write_frame_signals_csv(str(out), frames, engine="ours")
    with open(out, newline="") as f:
        rows = list(csv.reader(f))
    # Row 1 (header), Row 2 (no face — empty signal cells), Row 3 (face — populated).
    assert rows[1][2] == "false"
    assert rows[1][3:7] == ["", "", "", ""]
    assert rows[2][2] == "true"
    assert rows[2][3] == "0.15"


# ---------- event side handling ----------


def test_event_csv_side_blank_when_none(tmp_path):
    events = [
        ChewEvent(t_sec=0.5, signal_value=0.3, source_signal="jaw_open", frame_index=15),
        ChewEvent(t_sec=1.0, signal_value=0.4, source_signal="chin_y", frame_index=30,
                  confidence=0.8, side="left"),
    ]
    out = tmp_path / "events.csv"
    write_event_csv(str(out), events, engine="ours")
    with open(out, newline="") as f:
        rows = list(csv.reader(f))
    assert rows[1][6] == ""        # side None → ""
    assert rows[2][6] == "left"


# ---------- summary JSON (SPEC §7.5) ----------


def _make_ours_result() -> Result:
    return Result(
        engine_name="ours",
        duration_sec=10.0,
        fps=30.0,
        face_detection_rate=0.95,
        n_chews=20,
        chews_per_min=120.0,
        video_path="/tmp/clip.mp4",
        frame_count=300,
        usable_duration_sec=9.5,
        frames=[
            FrameSignal(t_sec=0.0, frame_index=0, face_found=True,
                        mar=0.1, jaw_open=0.2, chin_y=0.3, head_motion=0.0, quality=0.9),
            FrameSignal(t_sec=0.03, frame_index=1, face_found=True,
                        mar=0.1, jaw_open=0.2, chin_y=0.3, head_motion=0.0, quality=0.7),
        ],
        bouts=[Bout(t_start=0.0, t_end=5.0, n_events=10, chews_per_min=120.0, confidence=0.9)],
        warnings=["sample warning"],
    )


def _make_orofac_result() -> Result:
    return Result(
        engine_name="orofac",
        duration_sec=10.0,
        fps=30.0,
        face_detection_rate=0.0,
        n_chews=18,
        chews_per_min=108.0,
        video_path="/tmp/clip.mp4",
        frame_count=300,
        extra={"left": 8, "right": 7, "middle": 3, "n_cycles": 2},
    )


def test_summary_json_has_top_level_keys(tmp_path):
    out = tmp_path / "summary.json"
    write_summary_json(str(out), [_make_ours_result(), _make_orofac_result()])
    with open(out) as f:
        data = json.load(f)
    assert set(data.keys()) == {"video", "duration_sec", "fps", "frame_count", "engines", "agreement"}
    assert data["video"] == "clip.mp4"
    assert data["agreement"] == {}
    assert set(data["engines"].keys()) == {"ours", "orofac"}
    # ours block has its shape-specific fields.
    ours = data["engines"]["ours"]
    assert ours["n_bouts"] == 1
    assert ours["face_detection_rate"] == 0.95
    assert ours["quality_mean"] == pytest.approx(0.8, abs=1e-9)
    # orofac block has side counts.
    orof = data["engines"]["orofac"]
    assert orof["left"] == 8 and orof["right"] == 7 and orof["middle"] == 3


def test_summary_json_accepts_single_result(tmp_path):
    out = tmp_path / "summary.json"
    write_summary_json(str(out), _make_ours_result())
    with open(out) as f:
        data = json.load(f)
    assert "ours" in data["engines"]
    assert "orofac" not in data["engines"]


def test_summary_json_empty_results_raises(tmp_path):
    with pytest.raises(ValueError):
        write_summary_json(str(tmp_path / "x.json"), [])
