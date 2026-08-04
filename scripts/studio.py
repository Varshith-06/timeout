"""Browser studio: the whole intake wizard as a web front-end.

Open it and everything happens in the browser — paste a YouTube link (or point at
a local file), calibrate the court and the ball by clicking on a frame, and it
builds the pause-to-overlay app and shows the player. No command line, no desktop
calibration window.

    python scripts/studio.py            # http://localhost:8000

Endpoints (used by src/app/studio/index.html):
  GET  /                     the studio page
  GET  /player               the pause-to-overlay player (after a build)
  GET  /api/landmarks        court landmark names + coords + descriptions
  POST /api/source           {kind:'url'|'path', value, start, end} -> load a clip
  GET  /api/frame?t=SEC      a JPEG frame for calibrating
  GET  /api/suggest          auto-suggested calibration timestamps
  POST /api/calibrate        {time, clicks:{name:[x,y]}, ball:[x,y]} -> solve + save a shot
  POST /api/build            {detector} -> build the app from the saved shots
  GET  /recommendations.json,/clip.mp4   served from the workspace
  POST /chat                 LLM assistant (see chat_backend)
"""
from __future__ import annotations

import base64
import io
import json
import os
import re
import subprocess
import sys
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import http.server  # noqa: E402
import socketserver  # noqa: E402

WORK = ROOT / "out" / "studio"
WORK.mkdir(parents=True, exist_ok=True)
STATE = {"video": None, "meta": None, "loading": False, "load_error": None,
         "shots": [], "building": False, "built": False, "log": ""}


def _json(handler, obj, code=200):
    data = json.dumps(obj).encode()
    handler.send_response(code)
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Content-Length", str(len(data)))
    handler.end_headers()
    handler.wfile.write(data)


def _ffmpeg():
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


def _load_source(kind, value, start, end):
    if kind == "url":
        out = WORK; cmd = [sys.executable, "-m", "yt_dlp",
                           "-f", "bestvideo[height<=720][ext=mp4]+bestaudio/best[height<=720]",
                           "--merge-output-format", "mp4", "-o", str(out / "clip.%(ext)s")]
        ff = _ffmpeg()
        if ff:
            cmd += ["--ffmpeg-location", ff]
        if start or end:
            cmd += ["--download-sections", f"*{start or '0:00'}-{end or '99:59'}", "--force-keyframes-at-cuts"]
        cmd.append(value)
        (WORK / "clip.mp4").unlink(missing_ok=True)
        subprocess.run(cmd, check=True)
        return WORK / "clip.mp4"
    p = Path(value).expanduser()
    if not p.exists():
        raise FileNotFoundError(value)
    return p


def _video_meta(path):
    from src.perception.video import VideoSource
    v = VideoSource(str(path))
    m = {"duration": round(v.frame_count / v.fps, 1), "fps": v.fps, "w": v.width, "h": v.height}
    v.release()
    return m


class Handler(http.server.SimpleHTTPRequestHandler):
    def _frame_jpeg(self, t):
        import cv2
        from src.perception.video import VideoSource
        v = VideoSource(str(STATE["video"])); rgb = v.frame_at_time(float(t)); v.release()
        ok, buf = cv2.imencode(".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        return buf.tobytes()

    # ---- GET ----
    def do_GET(self):
        path = self.path.split("?")[0]
        if path in ("/", "/index.html"):
            return self._send_file(ROOT / "src/app/studio/index.html", "text/html")
        if path == "/player":
            return self._send_file(ROOT / "src/app/webapp/index.html", "text/html")
        if path == "/api/landmarks":
            from src.perception.calibrate import CLICK_LANDMARKS
            return _json(self, [{"name": n, "xy": list(xy), "desc": d} for n, xy, d in CLICK_LANDMARKS])
        if path == "/api/status":
            return _json(self, {"loading": STATE["loading"], "load_error": STATE["load_error"],
                                "video_ready": STATE["video"] is not None, "meta": STATE["meta"],
                                "building": STATE["building"], "built": STATE["built"],
                                "log": STATE["log"], "shots": STATE["shots"]})
        if path == "/api/frame":
            t = re.search(r"t=([\d.]+)", self.path)
            try:
                data = self._frame_jpeg(t.group(1) if t else 0)
                self.send_response(200); self.send_header("Content-Type", "image/jpeg")
                self.send_header("Content-Length", str(len(data))); self.end_headers()
                self.wfile.write(data); return
            except Exception as e:
                return _json(self, {"error": str(e)}, 500)
        if path == "/api/suggest":
            return self._suggest()
        if path in ("/recommendations.json", "/clip.mp4"):
            return self._send_range(WORK / path.lstrip("/"))
        self.send_error(404)

    def _suggest(self):
        try:
            r = subprocess.run([sys.executable, str(ROOT / "scripts/suggest_calibration.py"),
                                "--video", str(STATE["video"]), "--n", "4"],
                               capture_output=True, text=True, timeout=180)
            times = [float(t) for t in dict.fromkeys(re.findall(r"--time\s+([\d.]+)", r.stdout))]
            return _json(self, {"times": times or [2.0]})
        except Exception:
            return _json(self, {"times": [2.0]})

    # ---- POST ----
    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(n) if n else b"{}"
        if self.path == "/chat":
            try:
                from src.app.chat_backend import answer
                body = json.loads(raw or b"{}")
                return _json(self, answer(body.get("message", ""), body.get("rec") or {}))
            except Exception as e:
                return _json(self, {"error": str(e)}, 503)
        try:
            body = json.loads(raw or b"{}")
        except Exception:
            return _json(self, {"error": "bad json"}, 400)
        if self.path == "/api/source":
            # Load asynchronously (a URL download can take a while) — the page polls
            # /api/status for a spinner instead of blocking on this request.
            STATE.update(video=None, meta=None, loading=True, load_error=None, shots=[], built=False)

            def load():
                try:
                    p = _load_source(body["kind"], body["value"], body.get("start"), body.get("end"))
                    STATE["meta"] = _video_meta(p); STATE["video"] = str(p)
                except Exception as e:
                    STATE["load_error"] = f"{type(e).__name__}: {e}"
                STATE["loading"] = False

            threading.Thread(target=load, daemon=True).start()
            return _json(self, {"ok": True, "loading": True})
        if self.path == "/api/calibrate":
            return self._calibrate(body)
        if self.path == "/api/build":
            return self._build(body)
        self.send_error(404)

    def _calibrate(self, body):
        from src.perception.calibrate import solve_calibration
        clicks = {k: tuple(v) for k, v in body.get("clicks", {}).items()}
        meta = _video_meta(STATE["video"])
        calib = solve_calibration(clicks, img_size=(meta["w"], meta["h"]))
        if calib is None:
            return _json(self, {"ok": False, "error": "need 4+ well-spread points"}, 200)
        calib.time = float(body.get("time", 0))
        if body.get("ball"):
            calib.ball_px = tuple(body["ball"])
        i = len(STATE["shots"]) + 1
        f = WORK / f"shot{i}.json"; calib.save(f)
        STATE["shots"].append({"file": str(f), "time": calib.time, "reproj": calib.reproj_error_ft})
        return _json(self, {"ok": True, "reproj": round(calib.reproj_error_ft, 2), "n_shots": i})

    def _build(self, body):
        if STATE["building"] or not STATE["shots"]:
            return _json(self, {"ok": False, "error": "no shots or already building"}, 400)
        shots = [s["file"] for s in STATE["shots"]]
        det = body.get("detector", "yolo")
        cmd = [sys.executable, str(ROOT / "scripts/build_webapp.py"), "--video", str(STATE["video"]),
               "--shots", *shots, "--detector", det, "--detect-workers",
               "16" if det == "roboflow" else "1", "--live-ball", "4", "--out", str(WORK)]

        def run():
            STATE["building"] = True; STATE["log"] = "Detecting, tracking, scoring..."
            try:
                r = subprocess.run(cmd, capture_output=True, text=True, cwd=str(ROOT))
                STATE["log"] = (r.stdout or "")[-400:] + (r.stderr or "")[-200:]
                STATE["built"] = (WORK / "recommendations.json").exists()
            except Exception as e:
                STATE["log"] = str(e)
            STATE["building"] = False

        threading.Thread(target=run, daemon=True).start()
        return _json(self, {"ok": True})

    # ---- file/range helpers ----
    def _send_file(self, path, ctype):
        data = Path(path).read_bytes()
        self.send_response(200); self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data))); self.end_headers()
        self.wfile.write(data)

    def _send_range(self, path):
        if not path.is_file():
            return self.send_error(404)
        size = path.stat().st_size
        rng = self.headers.get("Range")
        m = re.match(r"bytes=(\d+)-(\d*)", rng or "")
        with open(path, "rb") as fh:
            if m:
                a = int(m.group(1)); b = int(m.group(2)) if m.group(2) else size - 1
                fh.seek(a); data = fh.read(b - a + 1)
                self.send_response(206)
                self.send_header("Content-Range", f"bytes {a}-{b}/{size}")
            else:
                data = fh.read(); self.send_response(200)
        ctype = "application/json" if path.suffix == ".json" else "video/mp4"
        self.send_header("Content-Type", ctype); self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(len(data))); self.end_headers()
        self.wfile.write(data)

    def log_message(self, *a):
        pass


def main():
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
    http.server.ThreadingHTTPServer.allow_reuse_address = True
    chat = "Groq" if os.environ.get("GROQ_API_KEY") else ("Claude" if os.environ.get("ANTHROPIC_API_KEY") else "offline")
    with http.server.ThreadingHTTPServer(("", port), Handler) as httpd:
        print(f"NBA Play Recommender studio:  http://localhost:{port}\n  assistant: {chat}\nCtrl-C to stop.")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nstopped.")


if __name__ == "__main__":
    main()
