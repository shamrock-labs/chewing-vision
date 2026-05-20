"""Download sessions from Firebase Storage → local sessions/ dir + InsForge DB upsert.

Usage:
    .venv/bin/python ml/download_sessions.py [--dry-run] [--no-video] [--sync]

--sync: after local download, also push files to InsForge Storage
        (imu.csv, video.mp4, signals.csv if present)

Idempotent: skips files that already exist locally (and on InsForge if --sync).
After download, run:
    chewing analyze sessions/<session_id>/video_<session_id>.mp4
to generate labels_ours.csv and frame_signals_ours.csv, then re-run with --sync.
"""
import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import firebase_admin
from firebase_admin import credentials, storage

MAIN_ROOT = Path(__file__).resolve().parents[1]
SESSIONS_DIR = MAIN_ROOT / "sessions"
CRED_PATH = MAIN_ROOT / "soma-dc84d-firebase-adminsdk-fbsvc-d4d0f77964.json"
STORAGE_BUCKET = "soma-dc84d.firebasestorage.app"


def init_firebase():
    cred = credentials.Certificate(str(CRED_PATH))
    firebase_admin.initialize_app(cred, {"storageBucket": STORAGE_BUCKET})


def list_remote_sessions(bucket) -> dict[str, list]:
    """Return {session_id: [blob, ...]} grouped from sessions/{uid}/{session_id}/"""
    blobs = bucket.list_blobs(prefix="sessions/")
    sessions: dict[str, list] = {}
    for blob in blobs:
        parts = blob.name.split("/")
        # sessions / {uid} / {session_id} / {filename}
        if len(parts) < 4:
            continue
        session_id = parts[2]
        sessions.setdefault(session_id, []).append(blob)
    return sessions


def download_session(session_id: str, blobs: list, skip_video: bool, dry_run: bool) -> bool:
    """Download blobs for a session. Returns True if any file was downloaded."""
    dest_dir = SESSIONS_DIR / session_id
    dest_dir.mkdir(parents=True, exist_ok=True)
    downloaded = False
    for blob in blobs:
        filename = blob.name.split("/")[-1]
        if skip_video and filename.endswith(".mp4"):
            continue
        dest = dest_dir / filename
        if dest.exists():
            print(f"  skip (exists): {filename}")
            continue
        size_mb = (blob.size or 0) / 1_048_576
        print(f"  download: {filename} ({size_mb:.1f} MB)")
        if not dry_run:
            blob.download_to_filename(str(dest))
        downloaded = True
    return downloaded


def _query(sql: str):
    result = subprocess.run(
        ["npx", "@insforge/cli", "db", "query", sql],
        cwd=str(MAIN_ROOT),
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip())
    return result.stdout


def _import_sql(sql: str):
    with tempfile.NamedTemporaryFile(mode="w", suffix=".sql", delete=False) as f:
        f.write(sql)
        tmp = f.name
    result = subprocess.run(
        ["npx", "@insforge/cli", "db", "import", tmp],
        cwd=str(MAIN_ROOT),
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip())


def _compress_video(src: Path, dst: Path) -> Path | None:
    """Compress src mp4 → dst with ffmpeg. Returns dst on success, None on failure/skip."""
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        return None
    if dst.exists() and dst.stat().st_mtime >= src.stat().st_mtime:
        return dst  # cache is fresh
    print(f"  compressing video ({src.stat().st_size // 1_048_576}MB → ?) ...")
    result = subprocess.run(
        [ffmpeg, "-i", str(src), "-vf", "scale=640:-2",
         "-c:v", "libx264", "-crf", "28", "-preset", "fast",
         "-an", "-y", str(dst)],
        capture_output=True,
    )
    if result.returncode != 0:
        print(f"  ffmpeg failed: {result.stderr.decode()[:120]}")
        return None
    print(f"  compressed: {dst.stat().st_size // 1_048_576}MB")
    return dst


def sync_to_insforge(session_id: str, dry_run: bool):
    """Upload imu.csv, video.mp4 (compressed), signals.csv from local → InsForge Storage."""
    local_dir = SESSIONS_DIR / session_id
    upload_targets = [
        (sorted(local_dir.glob("imu_*.csv")),      f"{session_id}/imu.csv"),
        (sorted(local_dir.glob("video_*.mp4")),    f"{session_id}/video.mp4"),
        (sorted(local_dir.glob("frame_signals_ours.csv")) + [local_dir / "frame_signals_ours.csv"],
         f"{session_id}/signals.csv"),
    ]
    for candidates, key in upload_targets:
        seen = set()
        files = [p for p in candidates if p.exists() and str(p) not in seen and not seen.add(str(p))]
        if not files:
            continue
        local_file = files[0]

        # auto-compress mp4 before upload
        upload_file = local_file
        if key.endswith(".mp4"):
            compressed = local_dir / "video_compressed.mp4"
            if dry_run:
                ffmpeg = shutil.which("ffmpeg")
                if ffmpeg:
                    print(f"  [dry-run] would compress {local_file.name} → {compressed.name}")
                print(f"  [dry-run] would upload {compressed.name if ffmpeg else local_file.name} → InsForge {key}")
                continue
            result = _compress_video(local_file, compressed)
            if result:
                upload_file = result
        elif dry_run:
            print(f"  [dry-run] would upload {local_file.name} → InsForge {key}")
            continue

        result = subprocess.run(
            ["npx", "@insforge/cli", "storage", "upload", str(upload_file),
             "--bucket", "sessions", "--key", key],
            cwd=str(MAIN_ROOT), capture_output=True, text=True,
        )
        if result.returncode == 0:
            print(f"  insforge upload: {key}")
        else:
            print(f"  insforge upload FAILED ({key}): {result.stderr.strip()[:80]}")


def upsert_session_db(session_id: str, dry_run: bool):
    sql = f"INSERT INTO sessions (id) VALUES ('{session_id}') ON CONFLICT (id) DO NOTHING;"
    if dry_run:
        print(f"  [dry-run] would upsert session: {session_id}")
        return
    try:
        _import_sql(sql)
        print(f"  db upsert: {session_id}")
    except Exception as e:
        print(f"  db upsert failed (may already exist): {e}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="List without downloading")
    parser.add_argument("--no-video", action="store_true", help="Skip mp4 download")
    parser.add_argument("--sync", action="store_true", help="Also push to InsForge Storage after download")
    args = parser.parse_args()

    print("Initializing Firebase...")
    init_firebase()
    bucket = storage.bucket()

    print("Listing remote sessions...")
    remote = list_remote_sessions(bucket)
    if not remote:
        print("No sessions found in Firebase Storage.")
        return

    print(f"Found {len(remote)} remote session(s):")
    for sid in sorted(remote):
        filenames = [b.name.split("/")[-1] for b in remote[sid]]
        local_exists = (SESSIONS_DIR / sid).exists()
        status = "local" if local_exists else "new"
        print(f"  [{status}] {sid}: {', '.join(sorted(filenames))}")

    print()
    new_count = 0
    for session_id in sorted(remote):
        print(f"Session: {session_id}")
        downloaded = download_session(session_id, remote[session_id], args.no_video, args.dry_run)
        upsert_session_db(session_id, args.dry_run)
        if args.sync:
            sync_to_insforge(session_id, args.dry_run)
        if downloaded:
            new_count += 1

    print(f"\nDone. {new_count} session(s) had new files downloaded.")
    if new_count > 0 and not args.dry_run:
        print("\nNext: run chewing analyze on each new session's video:")
        for session_id in sorted(remote):
            video_files = [b for b in remote[session_id] if b.name.endswith(".mp4")]
            if video_files:
                print(f"  .venv/bin/chewing analyze sessions/{session_id}/{video_files[0].name.split('/')[-1]}")


if __name__ == "__main__":
    main()
