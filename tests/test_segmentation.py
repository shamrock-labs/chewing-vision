"""Unit tests for chewing.segmentation (US-014)."""

from __future__ import annotations

from chewing.segmentation import segment_bouts
from chewing.types import ChewEvent


def _ev(t: float, src: str = "jaw_open") -> ChewEvent:
    return ChewEvent(t_sec=t, signal_value=0.5, source_signal=src)


def test_segment_bouts_empty_input_returns_empty():
    assert segment_bouts([]) == []


def test_segment_bouts_close_events_merge_into_one_bout():
    """AC: events spaced 0.5s apart merge into one bout."""
    events = [_ev(t) for t in [0.0, 0.5, 1.0, 1.5, 2.0]]
    bouts = segment_bouts(events, max_gap_sec=1.2)
    assert len(bouts) == 1
    assert bouts[0].n_events == 5
    assert bouts[0].t_start == 0.0
    assert bouts[0].t_end == 2.0


def test_segment_bouts_wide_gap_splits_into_two():
    """AC: events spaced 2.0s apart split into two bouts.

    Events at [0.0, 0.5, 1.0, 3.0, 3.5]:
      - gap 0.0→0.5 = 0.5 ≤ 1.2 → same bout
      - gap 0.5→1.0 = 0.5 ≤ 1.2 → same bout
      - gap 1.0→3.0 = 2.0  > 1.2 → split here
      - gap 3.0→3.5 = 0.5 ≤ 1.2 → same bout
    Expected: 2 bouts, sized 3 and 2.
    """
    events = [_ev(t) for t in [0.0, 0.5, 1.0, 3.0, 3.5]]
    bouts = segment_bouts(events, max_gap_sec=1.2)
    assert len(bouts) == 2
    assert bouts[0].n_events == 3
    assert bouts[1].n_events == 2


def test_segment_bouts_does_not_filter_by_source_signal():
    """Contract: caller is responsible for source_signal filtering.

    segment_bouts treats jaw_open and mar events equally — both are included
    in the bout if their timestamps are within max_gap_sec.
    """
    events = [
        _ev(0.0, "mar"),
        _ev(0.5, "jaw_open"),
    ]
    bouts = segment_bouts(events, max_gap_sec=1.2)
    assert len(bouts) == 1
    assert bouts[0].n_events == 2


def test_segment_bouts_max_gap_boundary_is_inclusive():
    """gap == max_gap_sec stays in the same bout (≤ comparison)."""
    events = [_ev(0.0), _ev(1.2)]
    bouts = segment_bouts(events, max_gap_sec=1.2)
    assert len(bouts) == 1
    assert bouts[0].n_events == 2
