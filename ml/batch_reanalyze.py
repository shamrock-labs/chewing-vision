"""Batch re-analyze all sessions with composite signal mode.

Overwrites sessions/*/labels_ours.csv with composite-mode windows so that
compare_sessions.py (LOSO) reads the new labels.
"""

from __future__ import annotations

import sys
from pathlib import Path

WORKTREE_ROOT = Path(__file__).resolve().parents[1]  # .../compressed-rolling-falcon/
MAIN_REPO_ROOT = Path(__file__).resolve().parents[4]  # .../chewing-vision/
sys.path.insert(0, str(WORKTREE_ROOT))

from chewing.engines.ours import OursEngine
from chewing.labels import write_window_csv

SESSIONS_DIR = MAIN_REPO_ROOT / "sessions"
MAR_WEIGHT = 0.3
ENGINE = OursEngine(signal_mode="composite", mar_weight=MAR_WEIGHT)


def reanalyze_session(session_dir: Path) -> None:
    videos = sorted(session_dir.glob("video_*.mp4"))
    if not videos:
        print(f"  [skip] no video in {session_dir.name}")
        return
    video_path = str(videos[0])
    print(f"  analyzing {session_dir.name} ...", flush=True)
    result = ENGINE.analyze(video_path)
    if not result.windows:
        print(f"  [warn] no windows produced for {session_dir.name}")
        return
    out_path = str(session_dir / "labels_ours.csv")
    write_window_csv(out_path, result.windows, ENGINE.engine_name)
    chew_windows = sum(1 for w in result.windows if w.label == "chewing")
    print(
        f"  done: {result.n_chews} chews, {chew_windows}/{len(result.windows)} chew windows → {out_path}"
    )


def main() -> None:
    session_dirs = sorted(d for d in SESSIONS_DIR.iterdir() if d.is_dir())
    print(f"[batch_reanalyze] {len(session_dirs)} sessions, engine={ENGINE.engine_name}")
    for sd in session_dirs:
        reanalyze_session(sd)
    print("[batch_reanalyze] done.")


if __name__ == "__main__":
    main()
