"""One-shot pipeline: Firebase pull → analyze → DB import → InsForge sync → LOSO.

Usage:
    .venv/bin/python ml/sync.py [--notes "설명"] [--skip-loso]
"""
import argparse
import subprocess
import sys
from pathlib import Path

MAIN_ROOT = Path(__file__).resolve().parents[1]
PYTHON    = str(MAIN_ROOT / ".venv/bin/python")
CHEWING   = str(MAIN_ROOT / ".venv/bin/chewing")
SESSIONS  = MAIN_ROOT / "sessions"


def run(cmd: list[str]) -> None:
    print(f"\n>>> {' '.join(str(c) for c in cmd)}")
    result = subprocess.run(cmd, cwd=str(MAIN_ROOT))
    if result.returncode != 0:
        print(f"[sync] failed (exit {result.returncode}). Aborting.")
        sys.exit(result.returncode)


def analyze_new() -> None:
    new = [
        (sdir, sorted(sdir.glob("video_*.mp4"))[0])
        for sdir in sorted(SESSIONS.iterdir())
        if sdir.is_dir()
        and not (sdir / "labels_ours.csv").exists()
        and sorted(sdir.glob("video_*.mp4"))
    ]
    if not new:
        print("[sync] analyze: 신규 세션 없음")
        return
    for sdir, video in new:
        print(f"\n[sync] analyze: {sdir.name}")
        run([CHEWING, "analyze", str(video), "-o", str(sdir), "--engine", "both"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--notes", default="auto sync")
    ap.add_argument("--skip-loso", action="store_true", help="LOSO 재실행 건너뜀")
    args = ap.parse_args()

    run([PYTHON, "ml/download_sessions.py"])
    analyze_new()
    run([PYTHON, "ml/import_to_db.py", "--mode", "all"])
    run([PYTHON, "ml/download_sessions.py", "--sync"])
    if not args.skip_loso:
        run([PYTHON, "ml/save_loso_results.py", "--notes", args.notes])

    print("\n[sync] Done.")


if __name__ == "__main__":
    main()
