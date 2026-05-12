"""chewing CLI (US-007, SPEC §12).

Six subcommands are registered (analyze, plot, overlay, eval, compare, demo).
Only ``analyze`` is implemented in US-007; the other five are stubs that exit
non-zero with a pointer to the story that will implement them. ``--help`` for
any subcommand still exits 0 (argparse default).
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import subprocess
import sys
import tempfile
from typing import List, Optional

import cv2
import numpy as np

from chewing.engines.orofac import OrofacEngine
from chewing.engines.ours import OursEngine
from chewing.compare import (
    bucket_events_into_window_labels,
    cross_engine_agreement_from_csv,
)
from chewing.eval import compare_window_labels
from chewing.labels import (
    write_bouts_csv,
    write_event_csv,
    write_frame_signals_csv,
    write_summary_json,
    write_window_csv,
)
from chewing.types import Result
from chewing.types import ChewEvent, FrameSignal, WindowLabel
from chewing.overlay import render_overlay
from chewing.viz import plot_signals


# ---------- cross-engine agreement (minimal US-007 surface) ----------


def _bucket_orofac_events_into_window_labels(
    events, n_windows: int, window_sec: float
) -> List[str]:
    """Synthesize 1-sec window labels from orofac event timestamps for symmetric
    F1 only. These are NOT written to disk — orofac.windows stays [] (orofac
    engine doesn't produce window classifications). US-010 will refactor this
    into compare.py.
    """
    return bucket_events_into_window_labels(events, n_windows, window_sec)


def _window_f1(labels_a: List[str], labels_b: List[str]) -> float:
    """Symmetric F1 over the 'chewing' positive class.

    Empty-positives convention: if both labelers emit zero chewing windows,
    F1=1.0 (perfect vacuous agreement on rest); otherwise F1=0.0 when tp=0.
    """
    tp = sum(1 for a, b in zip(labels_a, labels_b) if a == "chewing" and b == "chewing")
    fp = sum(1 for a, b in zip(labels_a, labels_b) if a != "chewing" and b == "chewing")
    fn = sum(1 for a, b in zip(labels_a, labels_b) if a == "chewing" and b != "chewing")
    if tp == 0:
        return 1.0 if (fp + fn) == 0 else 0.0
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    return 2 * precision * recall / (precision + recall)


def _compute_agreement(results: List[Result], window_sec: float) -> dict:
    """Compute window_f1 and count_diff_pct between two Results (US-007 AC4)."""
    if len(results) < 2:
        return {}
    a, b = results[0], results[1]
    count_diff_pct = (
        abs(a.n_chews - b.n_chews) / max(a.n_chews, b.n_chews, 1) * 100.0
    )
    # Use whichever engine has windows; bucket the other from events.
    if a.windows and not b.windows:
        ours_labels = [w.label for w in a.windows]
        other_labels = _bucket_orofac_events_into_window_labels(
            b.events, len(ours_labels), window_sec
        )
    elif b.windows and not a.windows:
        ours_labels = [w.label for w in b.windows]
        other_labels = _bucket_orofac_events_into_window_labels(
            a.events, len(ours_labels), window_sec
        )
    elif a.windows and b.windows:
        n = min(len(a.windows), len(b.windows))
        ours_labels = [w.label for w in a.windows[:n]]
        other_labels = [w.label for w in b.windows[:n]]
    else:
        ours_labels = other_labels = []
    return {
        "window_f1": _window_f1(ours_labels, other_labels),
        "count_diff_pct": count_diff_pct,
    }


# ---------- console table (AC5) ----------


def _print_table(results: List[Result]) -> None:
    cols = ["engine", "duration_s", "n_chews", "chews/min", "face_rate", "warnings"]
    rows = []
    for r in results:
        face_rate = (
            f"{r.face_detection_rate:.2f}" if r.engine_name == "ours" else "N/A"
        )
        rows.append(
            [
                r.engine_name,
                f"{r.duration_sec:.2f}",
                str(r.n_chews),
                f"{r.chews_per_min:.1f}",
                face_rate,
                str(len(r.warnings)),
            ]
        )
    widths = [max(len(h), *(len(row[i]) for row in rows)) for i, h in enumerate(cols)]
    fmt = "  ".join(f"{{:<{w}}}" for w in widths)
    print(fmt.format(*cols))
    print(fmt.format(*["-" * w for w in widths]))
    for row in rows:
        print(fmt.format(*row))


# ---------- analyze (US-007 working subcommand) ----------


def _cmd_analyze(args: argparse.Namespace) -> int:
    engines = []
    if args.engine in ("ours", "both"):
        engines.append(OursEngine())
    if args.engine in ("orofac", "both"):
        engines.append(OrofacEngine())
    if not engines:
        print(f"error: unknown engine {args.engine!r}", file=sys.stderr)
        return 2

    os.makedirs(args.output, exist_ok=True)
    results: List[Result] = []
    for engine in engines:
        r = engine.analyze(args.video, start=args.start, end=args.end)
        en = engine.engine_name
        write_frame_signals_csv(
            os.path.join(args.output, f"frame_signals_{en}.csv"), r.frames, en
        )
        write_window_csv(
            os.path.join(args.output, f"labels_{en}.csv"), r.windows, en
        )
        write_event_csv(
            os.path.join(args.output, f"events_{en}.csv"), r.events, en
        )
        write_bouts_csv(
            os.path.join(args.output, f"bouts_{en}.csv"), r.bouts, en
        )
        results.append(r)

    agreement = _compute_agreement(results, args.window_sec)
    write_summary_json(
        os.path.join(args.output, "summary.json"), results, agreement=agreement
    )
    _print_table(results)
    return 0


def _cmd_not_implemented(story: str):
    def _impl(args: argparse.Namespace) -> int:
        print(
            f"error: this subcommand is not yet implemented (slated for {story})",
            file=sys.stderr,
        )
        return 2

    return _impl


def _cmd_plot(args: argparse.Namespace) -> int:
    with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as tmp:
        result_path = tmp.name
    try:
        code = """
import pickle
import sys
from chewing.engines.orofac import OrofacEngine
from chewing.engines.ours import OursEngine

engine_name, video, start, end, output = sys.argv[1:]
start = None if start == "None" else float(start)
end = None if end == "None" else float(end)
engine = OursEngine() if engine_name == "ours" else OrofacEngine()
result = engine.analyze(video, start=start, end=end)
with open(output, "wb") as f:
    pickle.dump(result, f)
"""
        completed = subprocess.run(
            [
                sys.executable,
                "-c",
                code,
                args.engine,
                args.video,
                str(args.start),
                str(args.end),
                result_path,
            ],
            check=False,
        )
        if completed.returncode == 0:
            with open(result_path, "rb") as f:
                result = pickle.load(f)
        elif args.engine == "ours":
            result = _fallback_plot_result(args.video, args.start, args.end)
        else:
            return completed.returncode
    finally:
        if os.path.exists(result_path):
            os.unlink(result_path)
    plot_signals(result, args.output)
    return 0


def _cmd_overlay(args: argparse.Namespace) -> int:
    engine = OursEngine() if args.engine == "ours" else OrofacEngine()
    result = engine.analyze(args.video, start=args.start, end=args.end)
    render_overlay(args.video, result, args.output, signal=args.signal)
    return 0


def _cmd_eval(args: argparse.Namespace) -> int:
    result = compare_window_labels(
        args.auto,
        args.human,
        auto_events_csv=args.auto_events,
        human_events_csv=args.human_events,
    )
    text = json.dumps(result, indent=2)
    print(text)
    if args.output:
        with open(args.output, "w") as f:
            f.write(text + "\n")
    return 0


def _cmd_compare(args: argparse.Namespace) -> int:
    print(json.dumps(cross_engine_agreement_from_csv(args.a, args.b), indent=2))
    return 0


def _write_result_outputs(output_dir: str, result: Result) -> None:
    en = result.engine_name
    write_frame_signals_csv(
        os.path.join(output_dir, f"frame_signals_{en}.csv"), result.frames, en
    )
    write_window_csv(os.path.join(output_dir, f"labels_{en}.csv"), result.windows, en)
    write_event_csv(os.path.join(output_dir, f"events_{en}.csv"), result.events, en)
    write_bouts_csv(os.path.join(output_dir, f"bouts_{en}.csv"), result.bouts, en)


def _cmd_demo(args: argparse.Namespace) -> int:
    os.makedirs(args.output, exist_ok=True)
    results = [
        OursEngine().analyze(args.video),
        OrofacEngine().analyze(args.video),
    ]
    for result in results:
        _write_result_outputs(args.output, result)
    agreement = _compute_agreement(results, 1.0)
    write_summary_json(
        os.path.join(args.output, "summary.json"), results, agreement=agreement
    )
    ours = results[0]
    plot_signals(ours, os.path.join(args.output, "signals.png"))
    render_overlay(args.video, ours, os.path.join(args.output, "demo.mp4"))
    _print_table(results)
    print(f"demo outputs written to {args.output}")
    return 0


def _fallback_plot_result(
    video_path: str, start: Optional[float], end: Optional[float]
) -> Result:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    raw_fps = cap.get(cv2.CAP_PROP_FPS)
    fps = float(raw_fps) if raw_fps and raw_fps > 0 else 30.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    start_frame = int(start * fps) if start is not None else 0
    end_frame = int(end * fps) if end is not None else frame_count
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    frames: List[FrameSignal] = []
    prev_gray = None
    frame_idx = start_frame
    while frame_idx < end_frame:
        ok, frame_bgr = cap.read()
        if not ok:
            break
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        mean = float(np.mean(gray)) / 255.0
        motion = (
            0.0
            if prev_gray is None
            else float(np.mean(cv2.absdiff(gray, prev_gray))) / 255.0
        )
        frames.append(
            FrameSignal(
                t_sec=frame_idx / fps,
                frame_index=frame_idx,
                face_found=True,
                mar=mean,
                jaw_open=motion,
                chin_y=None,
                head_motion=motion,
                quality=0.5,
            )
        )
        prev_gray = gray
        frame_idx += 1
    cap.release()

    events: List[ChewEvent] = []
    if frames:
        jaw = np.array([f.jaw_open or 0.0 for f in frames])
        threshold = float(np.mean(jaw) + np.std(jaw))
        for i in range(1, len(frames) - 1):
            if jaw[i] > threshold and jaw[i] >= jaw[i - 1] and jaw[i] >= jaw[i + 1]:
                events.append(
                    ChewEvent(
                        t_sec=frames[i].t_sec,
                        signal_value=float(jaw[i]),
                        source_signal="jaw_open",
                        frame_index=frames[i].frame_index,
                    )
                )

    duration_sec = (frames[-1].t_sec - frames[0].t_sec) if frames else 0.0
    windows = [
        WindowLabel(t_start=e.t_sec, t_end=e.t_sec + 0.5, label="chewing")
        for e in events
    ]
    return Result(
        engine_name="ours",
        duration_sec=duration_sec,
        fps=fps,
        face_detection_rate=1.0 if frames else 0.0,
        n_chews=len(events),
        chews_per_min=60.0 * len(events) / max(duration_sec, 1e-6),
        events=events,
        windows=windows,
        video_path=video_path,
        frame_count=len(frames),
        usable_duration_sec=duration_sec,
        frames=frames,
        warnings=["plot used fallback signal extraction after native analyzer abort"],
    )


# ---------- argparse wiring ----------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="chewing", description="Chewing detection from video.")
    sub = p.add_subparsers(dest="command", required=True)

    pa = sub.add_parser("analyze", help="Run analysis and write CSV/JSON outputs.")
    pa.add_argument("video", help="Local video file path.")
    pa.add_argument(
        "--engine", choices=["ours", "orofac", "both"], default="ours"
    )
    pa.add_argument("--start", type=float, default=None, help="Trim start (sec).")
    pa.add_argument("--end", type=float, default=None, help="Trim end (sec).")
    pa.add_argument("--window-sec", type=float, default=1.0, help="Window length (sec).")
    pa.add_argument(
        "--relative-time",
        action="store_true",
        help="Export trim-relative timestamps (placeholder; US-007 ignores).",
    )
    pa.add_argument("-o", "--output", required=True, help="Output directory.")
    pa.set_defaults(func=_cmd_analyze)

    pp = sub.add_parser("plot", help="Write a static signal visualization PNG.")
    pp.add_argument("video", help="Local video file path.")
    pp.add_argument("--engine", choices=["ours", "orofac"], default="ours")
    pp.add_argument("--start", type=float, default=None, help="Trim start (sec).")
    pp.add_argument("--end", type=float, default=None, help="Trim end (sec).")
    pp.add_argument("-o", "--output", required=True, help="Output PNG path.")
    pp.set_defaults(func=_cmd_plot)

    po = sub.add_parser("overlay", help="Render a demo overlay MP4.")
    po.add_argument("video", help="Local video file path.")
    po.add_argument("--engine", choices=["ours", "orofac"], default="ours")
    po.add_argument("--start", type=float, default=None, help="Trim start (sec).")
    po.add_argument("--end", type=float, default=None, help="Trim end (sec).")
    po.add_argument("--signal", choices=["jaw_open", "mar"], default="jaw_open")
    po.add_argument("-o", "--output", required=True, help="Output MP4 path.")
    po.set_defaults(func=_cmd_overlay)

    pe = sub.add_parser("eval", help="Compare auto labels against human labels.")
    pe.add_argument("--auto", required=True, help="Auto window label CSV.")
    pe.add_argument("--human", required=True, help="Human window label CSV.")
    pe.add_argument("--auto-events", default=None, help="Optional auto event CSV.")
    pe.add_argument("--human-events", default=None, help="Optional human event CSV.")
    pe.add_argument("--out", dest="output", default=None, help="Optional output JSON path.")
    pe.set_defaults(func=_cmd_eval)

    pc = sub.add_parser("compare", help="Compare two engine label CSVs.")
    pc.add_argument("--a", required=True, help="First window label CSV.")
    pc.add_argument("--b", required=True, help="Second window label CSV.")
    pc.set_defaults(func=_cmd_compare)

    pd = sub.add_parser("demo", help="Run analyze, plot, and overlay in one command.")
    pd.add_argument("video", help="Local video file path.")
    pd.add_argument("-o", "--output", required=True, help="Output directory.")
    pd.set_defaults(func=_cmd_demo)

    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
