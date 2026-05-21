"""Batch re-analyze all sessions with jaw_open signal mode.

Overwrites sessions/*/labels_ours.csv + labels_ours_jaw_open.csv so that
LOSO and import_to_db read consistent labels.

Flags:
  --chin-fallback   Also run chin_y fallback engine and write labels_ours_chin.csv
                    for comparison. Does NOT overwrite labels_ours.csv.
  --session <id>    Only re-analyze the given session (substring match).
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

MAIN_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MAIN_REPO_ROOT))

from chewing.engines.ours import OursEngine
from chewing.labels import write_window_csv

SESSIONS_DIR = MAIN_REPO_ROOT / "sessions"


def reanalyze_session(session_dir: Path, engine: OursEngine, out_name: str, copy_as_jaw_open: bool = False) -> None:
    videos = sorted(session_dir.glob("video_*.mp4"))
    if not videos:
        print(f"  [skip] no video in {session_dir.name}")
        return
    video_path = str(videos[0])
    print(f"  analyzing {session_dir.name} [{engine.engine_name}] ...", flush=True)
    result = engine.analyze(video_path)
    if not result.windows:
        print(f"  [warn] no windows produced for {session_dir.name}")
        return
    out_path = session_dir / out_name
    write_window_csv(str(out_path), result.windows, engine.engine_name)
    if copy_as_jaw_open:
        shutil.copy2(out_path, session_dir / "labels_ours_jaw_open.csv")
    chew_windows = sum(1 for w in result.windows if w.label == "chewing")
    print(f"  done: {result.n_chews} chews, {chew_windows}/{len(result.windows)} chew windows → {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chin-fallback", action="store_true",
                    help="Also run chin_y fallback and write labels_ours_chin.csv")
    ap.add_argument("--session", default="",
                    help="Only process sessions whose name contains this string")
    args = ap.parse_args()

    base_engine = OursEngine(signal_mode="jaw_open")
    chin_engine = OursEngine(signal_mode="jaw_open", use_chin_fallback=True) if args.chin_fallback else None

    session_dirs = sorted(d for d in SESSIONS_DIR.iterdir() if d.is_dir())
    if args.session:
        session_dirs = [d for d in session_dirs if args.session in d.name]

    print(f"[batch_reanalyze] {len(session_dirs)} sessions")
    for sd in session_dirs:
        reanalyze_session(sd, base_engine, "labels_ours.csv", copy_as_jaw_open=True)
        if chin_engine:
            reanalyze_session(sd, chin_engine, "labels_ours_chin.csv")
    print("[batch_reanalyze] done.")


if __name__ == "__main__":
    main()
