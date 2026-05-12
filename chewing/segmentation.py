"""Bout segmentation (US-014, SPEC §7.4).

Groups chewing events into contiguous bouts separated by gaps > max_gap_sec.

Contract change vs OursEngine's pre-US-014 inline `_build_bouts`: segment_bouts
does NOT filter by source_signal. The caller is responsible for selecting which
event subset to segment (e.g., OursEngine passes only jaw_open events to
preserve n_chews semantics).
"""

from __future__ import annotations

from typing import List

from chewing.types import Bout, ChewEvent


def _cluster_to_bout(cluster: List[ChewEvent]) -> Bout:
    t_start = cluster[0].t_sec
    t_end = cluster[-1].t_sec
    # Span floor avoids divide-by-zero on single-event bouts.
    span = max(t_end - t_start, 1e-3)
    return Bout(
        t_start=t_start,
        t_end=t_end,
        n_events=len(cluster),
        chews_per_min=60.0 * len(cluster) / span,
        confidence=min(1.0, 0.5 + 0.05 * len(cluster)),
    )


def segment_bouts(
    events: List[ChewEvent],
    max_gap_sec: float = 1.2,
) -> List[Bout]:
    """Group ``events`` into bouts separated by gaps > ``max_gap_sec``.

    Events must be sorted by t_sec ascending. Returns an empty list when
    ``events`` is empty.
    """
    if not events:
        return []
    bouts: List[Bout] = []
    cluster: List[ChewEvent] = [events[0]]
    for ev in events[1:]:
        if ev.t_sec - cluster[-1].t_sec <= max_gap_sec:
            cluster.append(ev)
        else:
            bouts.append(_cluster_to_bout(cluster))
            cluster = [ev]
    bouts.append(_cluster_to_bout(cluster))
    return bouts
