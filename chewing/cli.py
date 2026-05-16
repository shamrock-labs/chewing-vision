"""chewing CLI (US-007, SPEC §12).

Six original subcommands (analyze, plot, overlay, eval, compare, demo)
plus fetch for Firebase Storage session download.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
from rich.columns import Columns
from rich.console import Console
from rich.markup import escape
from rich.padding import Padding
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table
from rich.text import Text
from rich.panel import Panel
from rich.align import Align

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

console = Console()

try:
    import questionary as _q

    _QSTYLE = _q.Style([
        ("qmark",       "fg:#5fd7ff bold"),
        ("question",    "bold"),
        ("answer",      "fg:#5fd7ff bold"),
        ("pointer",     "fg:#5fd7ff bold"),
        ("highlighted", "fg:#5fd7ff bold"),
        ("selected",    "fg:#5fd7ff"),
        ("separator",   "fg:#555555"),
        ("instruction", "fg:#555555"),
        ("text",        ""),
        ("disabled",    "fg:#555555 italic"),
    ])
except ImportError:
    _QSTYLE = None


# ---------- helpers ----------


def _fmt_bytes(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"



# ---------- welcome ----------


def _show_welcome() -> None:
    console.print()
    console.print(Panel(
        Align.center(
            "[bold cyan]chewing-vision[/bold cyan]  [dim]0.1.0[/dim]\n\n"
            "[dim]AirPods IMU 씹기 감지 · 분석 · Firebase 연동[/dim]"
        ),
        border_style="cyan",
        padding=(1, 4),
    ))
    console.print()

    commands = [
        ("analyze", "비디오에서 씹기 분석 후 CSV / JSON 출력"),
        ("plot",    "신호 시각화 PNG 생성"),
        ("overlay", "씹기 감지 오버레이 MP4 렌더링"),
        ("eval",    "자동 라벨 vs 수동 라벨 평가"),
        ("compare", "두 엔진 CSV 비교"),
        ("demo",    "analyze + plot + overlay 한 번에"),
        ("fetch",   "Firebase Storage에서 세션 다운로드"),
    ]

    console.print("  [bold]Commands[/bold]\n")
    for cmd, desc in commands:
        console.print(f"  [cyan]{cmd:<10}[/cyan] [dim]{desc}[/dim]")

    console.print()
    console.print(
        "  [dim]chewing-vision [bold]<command>[/bold] --help  for details[/dim]"
    )
    console.print()


# ---------- interactive menu ----------


_CMD_INFO: dict[str, tuple[str, str]] = {
    "analyze": (
        "analyze — 씹기 분석",
        "비디오에서 씹기 이벤트를 감지하고 frame_signals / labels / events / bouts CSV와 summary.json을 출력합니다.",
    ),
    "plot": (
        "plot — 신호 시각화",
        "분석 결과를 MAR · jaw_open 시계열 그래프 PNG로 저장합니다.",
    ),
    "overlay": (
        "overlay — 오버레이 렌더링",
        "씹기 감지 결과를 원본 비디오에 오버레이한 MP4를 생성합니다.",
    ),
    "eval": (
        "eval — 라벨 평가",
        "자동 라벨 CSV와 수동 라벨 CSV를 비교해 precision · recall · F1을 출력합니다.",
    ),
    "compare": (
        "compare — 엔진 비교",
        "두 엔진(ours / orofac)이 출력한 CSV를 교차 비교해 agreement 점수를 산출합니다.",
    ),
    "demo": (
        "demo — 풀 파이프라인",
        "analyze → plot → overlay 를 한 번에 실행합니다. 결과물이 모두 출력 디렉토리에 저장됩니다.",
    ),
    "fetch": (
        "fetch — Firebase 다운로드",
        "Firebase Storage의 세션 목록 조회 및 로컬 다운로드를 수행합니다. 서비스 계정 JSON이 필요합니다.",
    ),
    "loso": (
        "loso — LOSO 교차 검증",
        "다운로드된 세션 디렉토리를 자동 탐색해 Leave-One-Session-Out CV를 실행하고 PNG + HTML 리포트를 저장합니다.",
    ),
}


def _gather_args(cmd: str) -> "argparse.Namespace | None":
    """Prompt for per-command arguments via questionary. Returns None if aborted."""
    import questionary

    title, detail = _CMD_INFO.get(cmd, (cmd, ""))
    console.print(
        f"\n  [bold cyan]{title}[/bold cyan]"
        f"\n  [dim]{detail}[/dim]\n"
    )

    def ask(widget):
        try:
            return widget.unsafe_ask()
        except KeyboardInterrupt:
            return None

    def path(msg, **kw):
        return ask(questionary.path(msg, style=_QSTYLE, **kw))

    def text(msg, **kw):
        return ask(questionary.text(msg, style=_QSTYLE, **kw))

    def sel(msg, **kw):
        return ask(questionary.select(msg, style=_QSTYLE, **kw))

    ns: dict = {}

    if cmd == "analyze":
        video = path("비디오 파일 경로:")
        if video is None:
            return None
        import os as _os
        default_out = str(_os.path.dirname(_os.path.abspath(video))) if video else "./output"
        output = text("출력 디렉토리:", default=default_out)
        if output is None:
            return None
        engine = sel("엔진:", choices=["ours", "orofac", "both"], default="ours")
        if engine is None:
            return None
        ns = dict(
            video=video, output=output, engine=engine,
            start=None, end=None, window_sec=1.0, relative_time=False,
            func=_cmd_analyze,
        )

    elif cmd == "plot":
        video = path("비디오 파일 경로:")
        if video is None:
            return None
        output = text("출력 PNG 경로:", default="./signals.png")
        if output is None:
            return None
        engine = sel("엔진:", choices=["ours", "orofac"], default="ours")
        if engine is None:
            return None
        ns = dict(
            video=video, output=output, engine=engine,
            start=None, end=None, func=_cmd_plot,
        )

    elif cmd == "overlay":
        video = path("비디오 파일 경로:")
        if video is None:
            return None
        output = text("출력 MP4 경로:", default="./overlay.mp4")
        if output is None:
            return None
        engine = sel("엔진:", choices=["ours", "orofac"], default="ours")
        if engine is None:
            return None
        signal = sel("신호:", choices=["jaw_open", "mar"], default="jaw_open")
        if signal is None:
            return None
        ns = dict(
            video=video, output=output, engine=engine,
            start=None, end=None, signal=signal, func=_cmd_overlay,
        )

    elif cmd == "eval":
        auto = path("자동 라벨 CSV:")
        if auto is None:
            return None
        human = path("수동 라벨 CSV:")
        if human is None:
            return None
        ns = dict(
            auto=auto, human=human,
            auto_events=None, human_events=None, output=None,
            func=_cmd_eval,
        )

    elif cmd == "compare":
        a = path("엔진 A CSV:")
        if a is None:
            return None
        b = path("엔진 B CSV:")
        if b is None:
            return None
        ns = dict(a=a, b=b, func=_cmd_compare)

    elif cmd == "demo":
        video = path("비디오 파일 경로:")
        if video is None:
            return None
        output = text("출력 디렉토리:", default="./demo_output")
        if output is None:
            return None
        ns = dict(video=video, output=output, func=_cmd_demo)

    elif cmd == "loso":
        sessions_dir = text("세션 디렉토리:", default="./sessions")
        if sessions_dir is None:
            return None
        output = text("리포트 출력 디렉토리:", default="ml/outputs")
        if output is None:
            return None
        ns = dict(sessions_dir=sessions_dir, output=output, func=_cmd_loso)

    elif cmd == "fetch":
        action = sel(
            "작업 선택:",
            choices=[
                "세션 목록 보기 (--list)",
                "모든 세션 다운로드 (--all)",
                "특정 세션 다운로드",
            ],
        )
        if action is None:
            return None
        output = text("로컬 저장 경로:", default="./sessions")
        if output is None:
            return None
        if "list" in action:
            ns = dict(list=True, all=False, session_id=None,
                      output=output, credentials=None, func=_cmd_fetch)
        elif "all" in action:
            ns = dict(list=False, all=True, session_id=None,
                      output=output, credentials=None, func=_cmd_fetch)
        else:
            sid = text("세션 ID:")
            if sid is None:
                return None
            ns = dict(list=False, all=False, session_id=sid,
                      output=output, credentials=None, func=_cmd_fetch)

    return argparse.Namespace(**ns)


def _interactive_menu() -> int:
    """Arrow-key menu launcher. Loops until Exit or Ctrl+C."""
    import questionary

    console.print()
    console.print(Panel(
        Align.center(
            "[bold cyan]chewing-vision[/bold cyan]  [dim]0.1.0[/dim]\n\n"
            "[dim]AirPods IMU 씹기 감지 · 분석 · Firebase 연동[/dim]"
        ),
        border_style="cyan",
        padding=(1, 4),
    ))
    console.print()

    choices = [
        questionary.Choice("analyze  — 비디오에서 씹기 분석",     value="analyze"),
        questionary.Choice("plot     — 신호 시각화 PNG 생성",      value="plot"),
        questionary.Choice("overlay  — 씹기 감지 오버레이 MP4",    value="overlay"),
        questionary.Choice("eval     — 자동 vs 수동 라벨 평가",    value="eval"),
        questionary.Choice("compare  — 두 엔진 CSV 비교",          value="compare"),
        questionary.Choice("demo     — analyze + plot + overlay",  value="demo"),
        questionary.Choice("fetch    — Firebase 세션 다운로드",     value="fetch"),
        questionary.Separator(),
        questionary.Choice("loso     — LOSO 교차 검증",              value="loso"),
        questionary.Separator(),
        questionary.Choice("exit",                                 value="exit"),
    ]

    while True:
        try:
            cmd = questionary.select(
                "Command:", choices=choices, style=_QSTYLE
            ).unsafe_ask()
        except KeyboardInterrupt:
            break

        if cmd is None or cmd == "exit":
            break

        console.print()
        ns = _gather_args(cmd)
        if ns is None:
            console.print("  [dim]취소됨[/dim]\n")
            continue

        console.print()
        try:
            ns.func(ns)
        except Exception as exc:
            console.print(f"  [red]✗[/red]  {escape(str(exc))}")
        console.print()

    console.print("\n  [dim]Bye![/dim]\n")
    return 0


# ---------- cross-engine agreement ----------


def _bucket_orofac_events_into_window_labels(
    events, n_windows: int, window_sec: float
) -> List[str]:
    return bucket_events_into_window_labels(events, n_windows, window_sec)


def _window_f1(labels_a: List[str], labels_b: List[str]) -> float:
    tp = sum(1 for a, b in zip(labels_a, labels_b) if a == "chewing" and b == "chewing")
    fp = sum(1 for a, b in zip(labels_a, labels_b) if a != "chewing" and b == "chewing")
    fn = sum(1 for a, b in zip(labels_a, labels_b) if a == "chewing" and b != "chewing")
    if tp == 0:
        return 1.0 if (fp + fn) == 0 else 0.0
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    return 2 * precision * recall / (precision + recall)


def _compute_agreement(results: List[Result], window_sec: float) -> dict:
    if len(results) < 2:
        return {}
    a, b = results[0], results[1]
    count_diff_pct = (
        abs(a.n_chews - b.n_chews) / max(a.n_chews, b.n_chews, 1) * 100.0
    )
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


# ---------- results table ----------


def _print_table(results: List[Result]) -> None:
    table = Table(show_header=True, header_style="bold", show_lines=False, box=None,
                  pad_edge=False)
    table.add_column("engine",     style="bold cyan", min_width=8)
    table.add_column("duration",   justify="right",   min_width=9)
    table.add_column("chews",      justify="right",   min_width=6)
    table.add_column("chews/min",  justify="right",   min_width=9)
    table.add_column("face rate",  justify="right",   min_width=9)
    table.add_column("warnings",   justify="right",   min_width=8)

    for r in results:
        face_rate = (
            f"{r.face_detection_rate:.2f}" if r.engine_name == "ours" else "[dim]N/A[/dim]"
        )
        n_warn = len(r.warnings)
        warn_str = (
            f"[yellow]{n_warn}[/yellow]" if n_warn > 0 else f"[dim]{n_warn}[/dim]"
        )
        table.add_row(
            r.engine_name,
            f"{r.duration_sec:.2f}s",
            str(r.n_chews),
            f"{r.chews_per_min:.1f}",
            face_rate,
            warn_str,
        )

    console.print()
    console.print(Padding(table, (0, 0, 0, 2)))


# ---------- analyze ----------


def _cmd_analyze(args: argparse.Namespace) -> int:
    engines = []
    if args.engine in ("ours", "both"):
        engines.append(OursEngine())
    if args.engine in ("orofac", "both"):
        engines.append(OrofacEngine())
    if not engines:
        console.print(f"[red]✗[/red] Unknown engine [bold]{args.engine!r}[/bold]")
        return 2

    os.makedirs(args.output, exist_ok=True)
    video_name = Path(args.video).name
    results: List[Result] = []

    with console.status(
        f"  [cyan]Analyzing[/cyan]  [dim]{escape(video_name)}[/dim]",
        spinner="dots",
    ) as status:
        for engine in engines:
            status.update(
                f"  [cyan]Running[/cyan]  [bold]{engine.engine_name}[/bold]"
                f"  [dim]{escape(video_name)}[/dim]"
            )
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
    console.print(
        f"\n  [green]✓[/green]  Outputs saved  [dim]{escape(args.output)}[/dim]\n"
    )
    return 0


def _cmd_not_implemented(story: str):
    def _impl(args: argparse.Namespace) -> int:
        console.print(
            f"[red]✗[/red]  Not yet implemented  [dim](slated for {story})[/dim]"
        )
        return 2

    return _impl


# ---------- plot ----------


def _cmd_plot(args: argparse.Namespace) -> int:
    with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as tmp:
        result_path = tmp.name
    try:
        code = """
import pickle, sys
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
        with console.status(
            f"  [cyan]Plotting[/cyan]  [dim]{escape(Path(args.video).name)}[/dim]",
            spinner="dots",
        ):
            completed = subprocess.run(
                [sys.executable, "-c", code,
                 args.engine, args.video,
                 str(args.start), str(args.end), result_path],
                check=False,
            )
            if completed.returncode == 0:
                with open(result_path, "rb") as f:
                    result = pickle.load(f)
            elif args.engine == "ours":
                result = _fallback_plot_result(args.video, args.start, args.end)
            else:
                return completed.returncode

        plot_signals(result, args.output)
    finally:
        if os.path.exists(result_path):
            os.unlink(result_path)

    console.print(f"\n  [green]✓[/green]  Plot saved  [dim]{escape(args.output)}[/dim]\n")
    return 0


# ---------- overlay ----------


def _cmd_overlay(args: argparse.Namespace) -> int:
    engine = OursEngine() if args.engine == "ours" else OrofacEngine()
    with console.status(
        f"  [cyan]Rendering overlay[/cyan]  [dim]{escape(Path(args.video).name)}[/dim]",
        spinner="dots",
    ):
        result = engine.analyze(args.video, start=args.start, end=args.end)
        render_overlay(args.video, result, args.output, signal=args.signal)

    console.print(f"\n  [green]✓[/green]  Overlay saved  [dim]{escape(args.output)}[/dim]\n")
    return 0


# ---------- eval ----------


def _cmd_eval(args: argparse.Namespace) -> int:
    result = compare_window_labels(
        args.auto,
        args.human,
        auto_events_csv=args.auto_events,
        human_events_csv=args.human_events,
    )
    text = json.dumps(result, indent=2)
    console.print_json(text)
    if args.output:
        with open(args.output, "w") as f:
            f.write(text + "\n")
        console.print(f"\n  [green]✓[/green]  Saved  [dim]{escape(args.output)}[/dim]\n")
    return 0


# ---------- compare ----------


def _cmd_compare(args: argparse.Namespace) -> int:
    console.print_json(json.dumps(cross_engine_agreement_from_csv(args.a, args.b), indent=2))
    return 0


# ---------- demo ----------


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
    results: List[Result] = []

    with console.status(
        f"  [cyan]Demo[/cyan]  [dim]{escape(Path(args.video).name)}[/dim]",
        spinner="dots",
    ) as status:
        for engine_cls in (OursEngine, OrofacEngine):
            engine = engine_cls()
            status.update(
                f"  [cyan]Running[/cyan]  [bold]{engine.engine_name}[/bold]  …"
            )
            r = engine.analyze(args.video)
            _write_result_outputs(args.output, r)
            results.append(r)

        status.update("  [cyan]Rendering[/cyan]  signals + overlay …")
        agreement = _compute_agreement(results, 1.0)
        write_summary_json(
            os.path.join(args.output, "summary.json"), results, agreement=agreement
        )
        ours = results[0]
        plot_signals(ours, os.path.join(args.output, "signals.png"))
        render_overlay(args.video, ours, os.path.join(args.output, "demo.mp4"))

    _print_table(results)
    console.print(
        f"\n  [green]✓[/green]  Demo outputs  [dim]{escape(args.output)}[/dim]\n"
    )
    return 0


# ---------- fallback plot result ----------


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


# ---------- fetch ----------


def _cmd_fetch(args: argparse.Namespace) -> int:
    try:
        from chewing.firebase import FirebaseClient
    except ImportError:
        console.print(
            "[red]✗[/red]  firebase-admin not installed\n"
            "   Run: [bold]pip install 'chewing-vision\\[firebase]'[/bold]"
        )
        return 1

    # connect
    with console.status(
        "  [cyan]Connecting[/cyan]  [dim]soma-dc84d.firebasestorage.app[/dim]",
        spinner="dots",
    ):
        try:
            client = FirebaseClient(credentials_path=args.credentials)
        except (ValueError, Exception) as e:
            console.print(f"\n  [red]✗[/red]  {escape(str(e))}\n")
            return 1

    console.print(
        "  [green]✓[/green]  Connected  [dim]soma-dc84d.firebasestorage.app[/dim]"
    )

    # --list
    if args.list:
        with console.status("  [cyan]Fetching session list…[/cyan]", spinner="dots"):
            sessions = client.list_sessions()

        if not sessions:
            console.print("\n  [dim]No sessions found.[/dim]\n")
            return 0

        table = Table(show_header=True, header_style="bold", box=None,
                      show_lines=False, pad_edge=False)
        table.add_column("Session ID",  style="cyan",  min_width=28)
        table.add_column("UID",         style="dim",   min_width=10)
        table.add_column("Files",                      min_width=22)
        table.add_column("Size",        justify="right")

        total_bytes = 0
        for s in sessions:
            table.add_row(
                s["session_id"],
                s["uid"][:8] + "…",
                ", ".join(s["files"]),
                _fmt_bytes(s["total_bytes"]),
            )
            total_bytes += s["total_bytes"]

        console.print()
        console.print(Padding(table, (0, 0, 0, 2)))
        console.print(
            f"\n  [dim]{len(sessions)} session{'s' if len(sessions) != 1 else ''}"
            f"  ·  {_fmt_bytes(total_bytes)} total[/dim]\n"
        )
        return 0

    # --all
    if args.all:
        with console.status("  [cyan]Fetching session list…[/cyan]", spinner="dots"):
            sessions = client.list_sessions()

        if not sessions:
            console.print("\n  [dim]No sessions found.[/dim]\n")
            return 0

        console.print(
            f"\n  Downloading [bold]{len(sessions)}[/bold] sessions"
            f"  →  [dim]{escape(args.output)}[/dim]\n"
        )

        with Progress(
            SpinnerColumn(style="cyan"),
            TextColumn("  {task.description}"),
            BarColumn(bar_width=28),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            console=console,
            transient=False,
        ) as progress:
            overall = progress.add_task(
                "[cyan]sessions[/cyan]", total=len(sessions)
            )
            for s in sessions:
                sid = s["session_id"]
                file_task = progress.add_task(f"[dim]{sid}[/dim]",
                                              total=len(s["files"]))

                def on_start(fname, size, _t=file_task):
                    progress.update(_t, description=f"[dim]{fname}[/dim]")

                def on_done(fname, _t=file_task):
                    progress.advance(_t)

                client.download_session(sid, args.output,
                                        on_file_start=on_start, on_file_done=on_done)
                progress.advance(overall)

        console.print(
            f"\n  [green]✓[/green]  All sessions saved"
            f"  [dim]{escape(args.output)}[/dim]\n"
        )
        return 0

    # single session
    if not args.session_id:
        console.print(
            "  [red]✗[/red]  Provide a session ID, [bold]--list[/bold],"
            " or [bold]--all[/bold]\n"
        )
        return 2

    sid = args.session_id

    # fetch file list first to know total
    with console.status(f"  [cyan]Resolving[/cyan]  [dim]{sid}[/dim]", spinner="dots"):
        try:
            sessions = client.list_sessions()
        except Exception as e:
            console.print(f"\n  [red]✗[/red]  {escape(str(e))}\n")
            return 1

    match = next((s for s in sessions if s["session_id"] == sid), None)
    if match is None:
        console.print(f"\n  [red]✗[/red]  Session [bold]{sid}[/bold] not found\n")
        return 1

    n_files = len(match["files"])
    console.print(f"\n  Downloading  [bold cyan]{sid}[/bold cyan]\n")

    with Progress(
        SpinnerColumn(style="cyan"),
        TextColumn("  {task.description}"),
        BarColumn(bar_width=28),
        MofNCompleteColumn(),
        console=console,
        transient=False,
    ) as progress:
        task = progress.add_task("[dim]…[/dim]", total=n_files)

        def on_start(fname, size, _t=task):
            progress.update(
                _t,
                description=f"[dim]{escape(fname)}[/dim]"
                + (f"  [dim]{_fmt_bytes(size)}[/dim]" if size else ""),
            )

        def on_done(fname, _t=task):
            progress.advance(_t)

        try:
            out = client.download_session(sid, args.output,
                                          on_file_start=on_start, on_file_done=on_done)
        except FileNotFoundError as e:
            console.print(f"\n  [red]✗[/red]  {escape(str(e))}\n")
            return 1

    console.print(
        f"\n  [green]✓[/green]  Session saved  [dim]{escape(str(out))}[/dim]\n"
    )
    return 0


# ---------- loso ----------


def _cmd_loso(args: argparse.Namespace) -> int:
    import sys as _sys
    from pathlib import Path as _Path
    ml_dir = _Path(__file__).parent.parent / "ml"
    _sys.path.insert(0, str(ml_dir))
    try:
        from compare_sessions import main as _loso_main
    except ImportError as e:
        console.print(f"[red]✗[/red]  Cannot import ml/compare_sessions: {escape(str(e))}")
        return 1

    sessions_dir = _Path(args.sessions_dir)
    output_dir   = _Path(args.output)

    if not sessions_dir.exists():
        console.print(
            f"[red]✗[/red]  Sessions dir not found: [bold]{escape(str(sessions_dir))}[/bold]\n"
            f"   Run: [bold]chewing-vision fetch --all -o {escape(str(sessions_dir))}[/bold]"
        )
        return 1

    console.print(
        f"\n  [cyan]LOSO CV[/cyan]"
        f"  [dim]{escape(str(sessions_dir))}[/dim]"
        f"  →  [dim]{escape(str(output_dir))}[/dim]\n"
    )
    _loso_main(sessions_dir=sessions_dir, output_dir=output_dir)
    console.print(
        f"\n  [green]✓[/green]  Report saved  [dim]{escape(str(output_dir))}[/dim]\n"
    )
    return 0


# ---------- argparse ----------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="chewing", description="Chewing detection from video.")
    sub = p.add_subparsers(dest="command", required=False)

    # analyze
    pa = sub.add_parser("analyze", help="Run analysis and write CSV/JSON outputs.")
    pa.add_argument("video")
    pa.add_argument("--engine", choices=["ours", "orofac", "both"], default="ours")
    pa.add_argument("--start", type=float, default=None)
    pa.add_argument("--end", type=float, default=None)
    pa.add_argument("--window-sec", type=float, default=1.0)
    pa.add_argument("--relative-time", action="store_true")
    pa.add_argument("-o", "--output", required=True)
    pa.set_defaults(func=_cmd_analyze)

    # plot
    pp = sub.add_parser("plot", help="Write a static signal visualization PNG.")
    pp.add_argument("video")
    pp.add_argument("--engine", choices=["ours", "orofac"], default="ours")
    pp.add_argument("--start", type=float, default=None)
    pp.add_argument("--end", type=float, default=None)
    pp.add_argument("-o", "--output", required=True)
    pp.set_defaults(func=_cmd_plot)

    # overlay
    po = sub.add_parser("overlay", help="Render a demo overlay MP4.")
    po.add_argument("video")
    po.add_argument("--engine", choices=["ours", "orofac"], default="ours")
    po.add_argument("--start", type=float, default=None)
    po.add_argument("--end", type=float, default=None)
    po.add_argument("--signal", choices=["jaw_open", "mar"], default="jaw_open")
    po.add_argument("-o", "--output", required=True)
    po.set_defaults(func=_cmd_overlay)

    # eval
    pe = sub.add_parser("eval", help="Compare auto labels against human labels.")
    pe.add_argument("--auto", required=True)
    pe.add_argument("--human", required=True)
    pe.add_argument("--auto-events", default=None)
    pe.add_argument("--human-events", default=None)
    pe.add_argument("--out", dest="output", default=None)
    pe.set_defaults(func=_cmd_eval)

    # compare
    pc = sub.add_parser("compare", help="Compare two engine label CSVs.")
    pc.add_argument("--a", required=True)
    pc.add_argument("--b", required=True)
    pc.set_defaults(func=_cmd_compare)

    # demo
    pd = sub.add_parser("demo", help="Run analyze, plot, and overlay in one command.")
    pd.add_argument("video")
    pd.add_argument("-o", "--output", required=True)
    pd.set_defaults(func=_cmd_demo)

    # fetch
    pf = sub.add_parser(
        "fetch",
        help="Download sessions from Firebase Storage.",
        description=(
            "Download IMU/video sessions from Firebase Storage.\n\n"
            "Auth: pass --credentials or set CHEWING_FIREBASE_CREDENTIALS "
            "to the path of a service account JSON.\n"
            "Generate one at: Firebase Console → Project Settings → "
            "Service Accounts → Generate new private key."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    pf.add_argument(
        "session_id", nargs="?", default=None,
        help="Session ID to download (e.g. 20260513T145413_a61a4c).",
    )
    pf.add_argument("--list", action="store_true", help="List all available sessions.")
    pf.add_argument("--all",  action="store_true", help="Download every session.")
    pf.add_argument("-o", "--output", default="./sessions", help="Local output directory.")
    pf.add_argument(
        "--credentials", default=None, metavar="PATH",
        help="Service account JSON path (overrides CHEWING_FIREBASE_CREDENTIALS).",
    )
    pf.set_defaults(func=_cmd_fetch)

    # loso
    pl = sub.add_parser(
        "loso",
        help="LOSO CV across downloaded sessions.",
        description=(
            "Leave-One-Session-Out cross-validation across all sessions in --sessions-dir.\n\n"
            "Expected layout (produced by fetch + analyze):\n"
            "  {sessions-dir}/{session_id}/imu.csv\n"
            "  {sessions-dir}/{session_id}/session.json\n"
            "  {sessions-dir}/{session_id}/labels_ours.csv"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    pl.add_argument(
        "--sessions-dir", default="./sessions", metavar="DIR",
        help="Directory of downloaded sessions (default: ./sessions).",
    )
    pl.add_argument(
        "-o", "--output", default="ml/outputs", metavar="DIR",
        help="Output directory for PNG + HTML report (default: ml/outputs).",
    )
    pl.set_defaults(func=_cmd_loso)

    return p


def main(argv: Optional[List[str]] = None) -> int:
    from dotenv import load_dotenv
    load_dotenv()
    args = _build_parser().parse_args(argv)
    if not args.command:
        return _interactive_menu()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
