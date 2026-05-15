"""Firebase Storage client for listing and downloading sessions."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Callable

try:
    import firebase_admin
    from firebase_admin import credentials, storage as fb_storage

    _HAS_FIREBASE = True
except ImportError:
    _HAS_FIREBASE = False

BUCKET = "soma-dc84d.firebasestorage.app"


def _require_firebase() -> None:
    if not _HAS_FIREBASE:
        raise ImportError(
            "firebase-admin is not installed.\n"
            "Run: pip install 'chewing-vision[firebase]'"
        )


class FirebaseClient:
    """Read-only access to Firebase Storage sessions.

    Auth: service account JSON via --credentials flag or
    CHEWING_FIREBASE_CREDENTIALS environment variable.
    """

    def __init__(self, credentials_path: str | None = None) -> None:
        _require_firebase()
        creds_path = credentials_path or os.environ.get("CHEWING_FIREBASE_CREDENTIALS")
        if not creds_path:
            raise ValueError(
                "Firebase credentials required.\n"
                "Pass --credentials <path> or set CHEWING_FIREBASE_CREDENTIALS."
            )
        if not firebase_admin._apps:
            cred = credentials.Certificate(creds_path)
            firebase_admin.initialize_app(cred, {"storageBucket": BUCKET})
        self._bucket = fb_storage.bucket()

    def list_sessions(self) -> list[dict]:
        """Return sessions grouped from Storage path sessions/{uid}/{session_id}/."""
        blobs = list(self._bucket.list_blobs(prefix="sessions/"))
        sessions: dict[tuple[str, str], dict] = {}
        for blob in blobs:
            parts = blob.name.split("/")
            if len(parts) < 4:
                continue
            uid, session_id, filename = parts[1], parts[2], "/".join(parts[3:])
            if not filename:
                continue
            key = (uid, session_id)
            if key not in sessions:
                sessions[key] = {
                    "uid": uid,
                    "session_id": session_id,
                    "files": [],
                    "total_bytes": 0,
                }
            sessions[key]["files"].append(filename)
            sessions[key]["total_bytes"] += blob.size or 0
        return sorted(sessions.values(), key=lambda s: s["session_id"])

    def download_session(
        self,
        session_id: str,
        output_dir: str | Path,
        on_file_start: Callable[[str, int], None] | None = None,
        on_file_done: Callable[[str], None] | None = None,
    ) -> Path:
        """Download all files for session_id to output_dir/session_id/.

        Calls on_file_start(filename, size_bytes) before each file and
        on_file_done(filename) after.  Returns the local session directory.
        """
        blobs = [
            b for b in self._bucket.list_blobs(prefix=f"sessions/")
            if len(b.name.split("/")) >= 4 and b.name.split("/")[2] == session_id
        ]
        if not blobs:
            raise FileNotFoundError(
                f"Session {session_id!r} not found in Firebase Storage."
            )

        out = Path(output_dir) / session_id
        out.mkdir(parents=True, exist_ok=True)

        for blob in blobs:
            filename = blob.name.split("/")[-1]
            if not filename:
                continue
            local_path = out / filename
            if on_file_start:
                on_file_start(filename, blob.size or 0)
            blob.download_to_filename(str(local_path))
            if on_file_done:
                on_file_done(filename)

        return out

    def download_all(
        self,
        output_dir: str | Path,
        on_session_start: Callable[[str], None] | None = None,
        on_session_done: Callable[[str, Path], None] | None = None,
    ) -> list[Path]:
        """Download every session returned by list_sessions()."""
        sessions = self.list_sessions()
        results: list[Path] = []
        for s in sessions:
            if on_session_start:
                on_session_start(s["session_id"])
            path = self.download_session(s["session_id"], output_dir)
            if on_session_done:
                on_session_done(s["session_id"], path)
            results.append(path)
        return results
