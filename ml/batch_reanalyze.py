"""Batch re-analyze all sessions with jaw_open signal mode.

Overwrites sessions/*/labels_ours.csv + labels_ours_jaw_open.csv so that
LOSO and import_to_db read consistent labels.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

MAIN_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MAIN_REPO_ROOT))

from chewing.engines.ours import OursEngine
from chewing.labels import write_window_csv

SESSIONS_DIR = MAIN_REPO_ROOT / "sessions"
ENGINE = OursEngine(signal_mode="jaw_open")


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
    out_path = session_dir / "labels_ours.csv"
    write_window_csv(str(out_path), result.windows, ENGINE.engine_name)
    # import_to_db.py requires labels_ours_jaw_open.csv to exist
    shutil.copy2(out_path, session_dir / "labels_ours_jaw_open.csv")
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
