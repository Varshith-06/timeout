"""One-command setup for ANY clip: source -> calibrate -> build -> serve.

Give it a local video file (your own upload) or a YouTube URL, and it walks the
whole pipeline: download (if a URL), suggest calibration timestamps, open the
guided calibration window(s) (click the court landmarks, then the ball), build the
pause-to-overlay app, and serve it in your browser.

    python scripts/new_clip.py my_clip.mp4
    python scripts/new_clip.py "https://youtube.com/watch?v=..." --start 8:00 --end 12:00
    python scripts/new_clip.py game.mp4 --shots 4 --detector roboflow

No predefined clips — this is the front door for anything.
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable


def _ffmpeg() -> str | None:
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


def _download(url: str, start: str | None, end: str | None) -> Path:
    out_dir = ROOT / "data/video/clips/youtube"; out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [PY, "-m", "yt_dlp", "-f", "bestvideo[height<=720][ext=mp4]+bestaudio/best[height<=720]",
           "--merge-output-format", "mp4", "-o", str(out_dir / "clip_%(id)s.%(ext)s")]
    ff = _ffmpeg()
    if ff:
        cmd += ["--ffmpeg-location", ff]
    if start or end:
        cmd += ["--download-sections", f"*{start or '0:00'}-{end or '99:59'}", "--force-keyframes-at-cuts"]
    cmd.append(url)
    print("Downloading (this can take a minute)...")
    subprocess.run(cmd, check=True)
    mp4s = sorted(out_dir.glob("clip_*.mp4"), key=lambda p: p.stat().st_mtime)
    if not mp4s:
        raise SystemExit("download produced no mp4")
    return mp4s[-1]


def _suggest_times(video: Path, n: int) -> list[float]:
    r = subprocess.run([PY, str(ROOT / "scripts/suggest_calibration.py"), "--video", str(video),
                        "--n", str(n)], capture_output=True, text=True)
    times = re.findall(r"--time\s+([\d.]+)", r.stdout)
    return [float(t) for t in dict.fromkeys(times)]      # unique, ordered


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Set up the recommender on any clip")
    ap.add_argument("source", help="local video file OR a YouTube URL")
    ap.add_argument("--start", help="URL only: clip start, e.g. 8:00")
    ap.add_argument("--end", help="URL only: clip end, e.g. 12:00")
    ap.add_argument("--shots", type=int, default=3, help="how many camera shots to calibrate")
    ap.add_argument("--detector", choices=["yolo", "roboflow"], default="yolo")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args(argv)

    # 1. Resolve the source.
    if re.match(r"https?://", args.source):
        video = _download(args.source, args.start, args.end)
    else:
        video = Path(args.source).resolve()
        if not video.exists():
            raise SystemExit(f"no such file: {video}")
    name = video.stem
    out = ROOT / "out" / f"webapp_{name}"
    print(f"clip: {video}")

    # 2. Suggest calibration timestamps.
    times = _suggest_times(video, args.shots)
    if not times:
        print("Couldn't auto-find clean court frames; calibrating at 2s."); times = [2.0]
    print(f"Calibrating {len(times)} shot(s) at: {', '.join(f'{t:.1f}s' for t in times)}")

    # 3. Guided calibration (court landmarks + ball) — opens windows to click.
    shot_files = [f"{name}_shot{i+1}.json" for i in range(len(times))]
    subprocess.run([PY, str(ROOT / "scripts/calibrate_shots.py"), "--video", str(video),
                    "--times", *[str(t) for t in times], "--prefix", f"{name}_shot"], check=False)
    shots = [f for f in shot_files if (ROOT / f).exists()]
    if not shots:
        raise SystemExit("no calibrations were saved — nothing to build.")

    # 4. Build the app.
    print(f"Building the app from {len(shots)} calibrated shot(s)...")
    subprocess.run([PY, str(ROOT / "scripts/build_webapp.py"), "--video", str(video),
                    "--shots", *shots, "--detector", args.detector, "--detect-workers",
                    "16" if args.detector == "roboflow" else "1", "--live-ball", "4",
                    "--out", str(out)], check=True)

    # 5. Serve it.
    print(f"\nReady. Serving {out} ...")
    subprocess.run([PY, str(ROOT / "scripts/serve_webapp.py"), "--dir", str(out),
                    "--port", str(args.port)], check=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
