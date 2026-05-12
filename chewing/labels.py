"""CSV and JSON I/O for chewing-vision outputs (US-006, SPEC §6.2 + §7).

Public surface:
  * LABEL_VOCAB                       — SPEC §6.2 7-value label list
  * write_frame_signals_csv           — SPEC §7.1
  * write_event_csv                   — SPEC §7.2
  * write_window_csv / read_window_csv — SPEC §7.3 (round-trip safe)
  * write_bouts_csv                   — SPEC §7.4
  * write_summary_json                — SPEC §7.5
"""

from __future__ import annotations

import csv
import json
import os
from typing import Iterable, List, Sequence, Union

from chewing.types import Bout, ChewEvent, FrameSignal, Result, WindowLabel


LABEL_VOCAB = [
    "chewing",
    "rest",
    "speaking",
    "drinking",
    "occluded",
    "bad_face",
    "unknown",
]


# ---------- helpers ----------


def _csv_value(v) -> str:
    """Render a CSV cell — None/NaN become empty strings."""
    if v is None:
        return ""
    if isinstance(v, float) and v != v:  # NaN
        return ""
    if isinstance(v, bool):
        return "true" if v else "false"
    return str(v)


def _optional_float(s: str):
    """Parse a CSV cell back to float-or-None ('' → None)."""
    if s == "":
        return None
    return float(s)


def _optional_str(s: str):
    return s if s else None


# ---------- frame signals (SPEC §7.1) ----------


FRAME_HEADER = [
    "t_sec",
    "frame_index",
    "face_found",
    "mar",
    "jaw_open",
    "chin_y",
    "head_motion",
    "quality",
]


def write_frame_signals_csv(
    path: str, frames: Iterable[FrameSignal], engine: str
) -> None:
    """Write per-frame signals to CSV. ``engine`` is recorded in the filename
    upstream (SPEC §7.1 columns omit engine); included here as a parameter to
    keep the writer signature uniform with the other CSV writers."""
    _ = engine  # consumed by callers for filename construction; not a column.
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(FRAME_HEADER)
        for fs in frames:
            w.writerow(
                [
                    _csv_value(fs.t_sec),
                    _csv_value(fs.frame_index),
                    _csv_value(fs.face_found),
                    _csv_value(fs.mar),
                    _csv_value(fs.jaw_open),
                    _csv_value(fs.chin_y),
                    _csv_value(fs.head_motion),
                    _csv_value(fs.quality),
                ]
            )


# ---------- events (SPEC §7.2) ----------


EVENT_HEADER = [
    "t_sec",
    "frame_index",
    "signal",
    "value",
    "confidence",
    "engine",
    "side",
]


def write_event_csv(
    path: str, events: Iterable[ChewEvent], engine: str
) -> None:
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(EVENT_HEADER)
        for ev in events:
            w.writerow(
                [
                    _csv_value(ev.t_sec),
                    _csv_value(ev.frame_index),
                    _csv_value(ev.source_signal),
                    _csv_value(ev.signal_value),
                    _csv_value(ev.confidence),
                    _csv_value(engine),
                    _csv_value(ev.side),
                ]
            )


# ---------- windows (SPEC §7.3 + US-015 label validation) ----------


WINDOW_HEADER = [
    "t_start",
    "t_end",
    "label",
    "confidence",
    "quality",
    "n_events",
    "engine",
    "mar_mean",
    "jaw_open_mean",
]


def write_window_csv(
    path: str, windows: Iterable[WindowLabel], engine: str
) -> None:
    """Write window labels to CSV (SPEC §7.3).

    Validates every ``window.label`` against ``LABEL_VOCAB`` (US-015 AC2);
    raises ValueError on any unknown label before writing.
    """
    rows = list(windows)
    for w in rows:
        if w.label not in LABEL_VOCAB:
            raise ValueError(
                f"Unknown window label {w.label!r}; expected one of {LABEL_VOCAB}"
            )
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(WINDOW_HEADER)
        for w in rows:
            writer.writerow(
                [
                    _csv_value(w.t_start),
                    _csv_value(w.t_end),
                    _csv_value(w.label),
                    _csv_value(w.confidence),
                    _csv_value(w.quality),
                    _csv_value(w.n_events),
                    _csv_value(engine),
                    _csv_value(w.mar_mean),
                    _csv_value(w.jaw_open_mean),
                ]
            )


def read_window_csv(path: str) -> List[WindowLabel]:
    """Round-trip reader for write_window_csv (US-006 AC7).

    Returns a list of WindowLabel. The ``engine`` column is read but not
    stored back (WindowLabel has no engine field).
    """
    out: List[WindowLabel] = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            out.append(
                WindowLabel(
                    t_start=float(row["t_start"]),
                    t_end=float(row["t_end"]),
                    label=row["label"],
                    confidence=float(row["confidence"]),
                    quality=float(row["quality"]),
                    n_events=int(row["n_events"]),
                    mar_mean=float(row["mar_mean"]),
                    jaw_open_mean=float(row["jaw_open_mean"]),
                )
            )
    return out


# ---------- bouts (SPEC §7.4) ----------


BOUT_HEADER = ["t_start", "t_end", "n_events", "chews_per_min", "confidence", "engine"]


def write_bouts_csv(path: str, bouts: Iterable[Bout], engine: str) -> None:
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(BOUT_HEADER)
        for b in bouts:
            w.writerow(
                [
                    _csv_value(b.t_start),
                    _csv_value(b.t_end),
                    _csv_value(b.n_events),
                    _csv_value(b.chews_per_min),
                    _csv_value(b.confidence),
                    _csv_value(engine),
                ]
            )


# ---------- summary JSON (SPEC §7.5) ----------


def _engine_summary(r: Result) -> dict:
    """Per-engine summary block. Shape varies by engine — orofac doesn't
    expose face / quality, but ours doesn't expose side counts.
    """
    block = {
        "n_chews": r.n_chews,
        "chews_per_min": r.chews_per_min,
    }
    if r.engine_name == "ours":
        face_qualities = [f.quality for f in r.frames if f.face_found]
        quality_mean = sum(face_qualities) / len(face_qualities) if face_qualities else 0.0
        block.update(
            {
                "usable_duration_sec": r.usable_duration_sec,
                "face_detection_rate": r.face_detection_rate,
                "n_bouts": len(r.bouts),
                "quality_mean": quality_mean,
                "warnings": list(r.warnings),
            }
        )
    elif r.engine_name == "orofac":
        block.update(
            {
                "left": int(r.extra.get("left", 0)),
                "right": int(r.extra.get("right", 0)),
                "middle": int(r.extra.get("middle", 0)),
                "warnings": list(r.warnings),
            }
        )
    else:
        block["warnings"] = list(r.warnings)
    return block


def write_summary_json(
    path: str,
    result_or_results: Union[Result, Sequence[Result]],
    agreement: dict = None,
) -> None:
    """Write SPEC §7.5 summary JSON.

    Accepts a single Result or a sequence of Results. Top-level video/fps/
    frame_count/duration_sec are taken from the first Result (engines should
    share the same source video). ``agreement`` defaults to ``{}`` — US-007
    (CLI analyze) and US-010 (eval) pass a computed dict.
    """
    if isinstance(result_or_results, Result):
        results = [result_or_results]
    else:
        results = list(result_or_results)
    if not results:
        raise ValueError("write_summary_json requires at least one Result")
    head = results[0]
    summary = {
        "video": os.path.basename(head.video_path) if head.video_path else "",
        "duration_sec": head.duration_sec,
        "fps": head.fps,
        "frame_count": head.frame_count,
        "engines": {r.engine_name: _engine_summary(r) for r in results},
        "agreement": dict(agreement) if agreement else {},
    }
    with open(path, "w") as f:
        json.dump(summary, f, indent=2)
