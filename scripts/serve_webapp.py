"""Serve the pre-built web app locally and open it in a browser.

A static file server is needed (not file://) so the page can fetch
recommendations.json. Run scripts/build_webapp.py first to populate the dir.

    python scripts/serve_webapp.py            # serves out/webapp on :8000
    python scripts/serve_webapp.py --dir out/webapp_test --port 8010
"""
from __future__ import annotations

import argparse
import http.server
import json
import os
import re
import socketserver
import sys
import webbrowser
from functools import partial
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


class RangeRequestHandler(http.server.SimpleHTTPRequestHandler):
    """SimpleHTTPRequestHandler + HTTP Range support, so the <video> can seek.

    Browsers require 206 Partial Content responses to scrub an mp4; the stdlib
    handler serves the whole file on every request, which breaks seeking.
    """

    def send_head(self):
        rng = self.headers.get("Range")
        if not rng:
            return super().send_head()
        path = self.translate_path(self.path)
        if not os.path.isfile(path):
            return super().send_head()
        m = re.match(r"bytes=(\d+)-(\d*)", rng)
        if not m:
            return super().send_head()
        size = os.path.getsize(path)
        start = int(m.group(1))
        end = int(m.group(2)) if m.group(2) else size - 1
        end = min(end, size - 1)
        if start > end:
            self.send_error(416); return None
        length = end - start + 1
        f = open(path, "rb"); f.seek(start)
        self.send_response(206)
        self.send_header("Content-Type", self.guess_type(path))
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.send_header("Content-Length", str(length))
        self.end_headers()
        self._range_remaining = length
        return _LimitedReader(f, length)

    def do_POST(self):
        if self.path.rstrip("/") != "/chat":
            self.send_error(404); return
        try:
            n = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(n) or b"{}")
            from src.app.chat_backend import answer
            out = answer(body.get("message", ""), body.get("rec") or {})
            data = json.dumps(out).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
        except Exception as e:
            # No key / SDK / model error -> 503 so the page uses its offline assistant.
            msg = json.dumps({"error": f"{type(e).__name__}: {e}"}).encode()
            self.send_response(503)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(msg)))
            self.end_headers()
            self.wfile.write(msg)

    def copyfile(self, source, outputfile):
        # Honour the range window when streaming the body.
        if isinstance(source, _LimitedReader):
            while True:
                chunk = source.read(64 * 1024)
                if not chunk:
                    break
                outputfile.write(chunk)
        else:
            super().copyfile(source, outputfile)


class _LimitedReader:
    """A file wrapper that yields at most ``remaining`` bytes then EOFs."""

    def __init__(self, f, remaining):
        self.f = f
        self.remaining = remaining

    def read(self, n=-1):
        if self.remaining <= 0:
            return b""
        if n < 0 or n > self.remaining:
            n = self.remaining
        data = self.f.read(n)
        self.remaining -= len(data)
        return data

    def close(self):
        self.f.close()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Serve the NBA play recommender web app")
    ap.add_argument("--dir", default="out/webapp")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--no-open", action="store_true")
    args = ap.parse_args(argv)

    root = Path(args.dir).resolve()
    if not (root / "index.html").exists():
        print(f"no index.html in {root} — run: python scripts/build_webapp.py --video <clip> --calib <calib.json>")
        return 2

    handler = partial(RangeRequestHandler, directory=str(root))
    socketserver.TCPServer.allow_reuse_address = True
    url = f"http://localhost:{args.port}/index.html"
    if os.environ.get("GROQ_API_KEY"):
        chat = f"LLM assistant: ON (Groq {os.environ.get('GROQ_MODEL', 'llama-3.3-70b-versatile')})"
    elif os.environ.get("ANTHROPIC_API_KEY"):
        chat = "LLM assistant: ON (Claude claude-opus-4-8)"
    else:
        chat = "LLM assistant: OFF (set GROQ_API_KEY or ANTHROPIC_API_KEY; the offline assistant still works)"
    with socketserver.TCPServer(("", args.port), handler) as httpd:
        print(f"serving {root}\n  {url}\n  {chat}\nCtrl-C to stop.")
        if not args.no_open:
            webbrowser.open(url)
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nstopped.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
