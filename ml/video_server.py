"""Local video + signal CSV server for the annotation web app.

Serves with Range request support (required for HTML5 video seeking):
  GET /sessions/{id}/video   -> mp4
  GET /sessions/{id}/signals -> frame_signals_ours.csv

Usage:
    .venv/bin/python ml/video_server.py [--port 8765]
    # expose to teammates:
    ngrok http 8765
"""
import argparse
import re
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path

SESSIONS_DIR = Path(__file__).resolve().parents[1] / "sessions"


class RangeHandler(BaseHTTPRequestHandler):
    def do_OPTIONS(self):
        self.send_response(200)
        self._cors()
        self.end_headers()

    def do_GET(self):
        parts = self.path.split("?")[0].strip("/").split("/")
        if len(parts) == 3 and parts[0] == "sessions":
            session_dir = SESSIONS_DIR / parts[1]
            if parts[2] == "video":
                videos = sorted(session_dir.glob("video_*.mp4"))
                if not videos:
                    self.send_error(404)
                    return
                self._serve(videos[0], "video/mp4")
            elif parts[2] == "signals":
                sig = session_dir / "frame_signals_ours.csv"
                if not sig.exists():
                    self.send_error(404)
                    return
                self._serve(sig, "text/csv; charset=utf-8")
            else:
                self.send_error(404)
        else:
            self.send_error(404)

    def _cors(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Range")
        self.send_header("Access-Control-Expose-Headers", "Content-Range, Accept-Ranges, Content-Length")

    def _serve(self, path: Path, content_type: str):
        file_size = path.stat().st_size
        rng = self.headers.get("Range", "")
        m = re.match(r"bytes=(\d+)-(\d*)", rng)
        if m:
            start = int(m.group(1))
            end = int(m.group(2)) if m.group(2) else file_size - 1
            end = min(end, file_size - 1)
            length = end - start + 1
            self.send_response(206)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Range", f"bytes {start}-{end}/{file_size}")
            self.send_header("Content-Length", str(length))
            self.send_header("Accept-Ranges", "bytes")
            self._cors()
            self.end_headers()
            with open(path, "rb") as f:
                f.seek(start)
                self.wfile.write(f.read(length))
        else:
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(file_size))
            self.send_header("Accept-Ranges", "bytes")
            self._cors()
            self.end_headers()
            with open(path, "rb") as f:
                self.wfile.write(f.read())

    def log_message(self, fmt, *args):
        print(f"  {self.address_string()} {fmt % args}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8765)
    args = ap.parse_args()
    print(f"[video_server] http://localhost:{args.port}")
    print(f"[video_server] sessions dir: {SESSIONS_DIR}")
    HTTPServer(("", args.port), RangeHandler).serve_forever()
