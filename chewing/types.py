"""Shared dataclasses used across chewing-vision engines and outputs."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ChewEvent:
    """A single detected chew peak.

    SPEC §6.1 declares additional fields (frame_index, confidence, side); they
    are appended below with defaults so existing v1 keyword construction
    (t_sec, signal_value, source_signal) still works unchanged (US-006 v2 schema).
    """

    t_sec: float
    signal_value: float
    source_signal: str  # one of: 'mar' | 'jaw_open' | 'chin_y'
    # v2 expansion (US-006, SPEC §6.1) — appended with defaults for v1 compat.
    frame_index: int = 0
    confidence: float = 1.0
    side: Optional[str] = None  # 'left' | 'right' | 'middle' | None


@dataclass
class WindowLabel:
    """A 1-second (configurable) classification window.

    SPEC §6.1 declares quality + n_events fields; appended with defaults
    so existing v1 callers keep working (US-006 v2 schema).
    """

    t_start: float
    t_end: float
    label: str  # one of LABEL_VOCAB (chewing|rest|speaking|drinking|occluded|bad_face|unknown)
    mar_mean: float = 0.0
    jaw_open_mean: float = 0.0
    confidence: float = 0.0
    # v2 expansion (US-006, SPEC §6.1).
    quality: float = 0.0
    n_events: int = 0


@dataclass
class FrameSignal:
    """Per-frame signal vector emitted by an engine (SPEC §6.1).

    Signal fields (mar, jaw_open, chin_y, head_motion) are Optional because they
    are undefined on frames where the face landmarker fails (face_found=False).
    """

    t_sec: float
    frame_index: int
    face_found: bool
    mar: Optional[float]
    jaw_open: Optional[float]
    chin_y: Optional[float]
    head_motion: Optional[float]
    quality: float


@dataclass
class Bout:
    """Contiguous chewing bout aggregated from events (SPEC §6.1, §7.4)."""

    t_start: float
    t_end: float
    n_events: int
    chews_per_min: float
    confidence: float


@dataclass
class Result:
    """Unified analysis result returned by every engine."""

    # v1 required fields — keep order stable for positional callers.
    engine_name: str
    duration_sec: float
    fps: float
    face_detection_rate: float
    n_chews: int
    chews_per_min: float
    # v1 collections.
    events: List[ChewEvent] = field(default_factory=list)
    windows: List[WindowLabel] = field(default_factory=list)
    extra: Dict[str, Any] = field(default_factory=dict)
    # v2 expansion (US-013, SPEC §6.1). Appended (not interleaved per SPEC order)
    # so existing v1 positional construction in day1/ and US-001 keeps working.
    video_path: str = ""
    frame_count: int = 0
    usable_duration_sec: float = 0.0
    frames: List[FrameSignal] = field(default_factory=list)
    bouts: List[Bout] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
