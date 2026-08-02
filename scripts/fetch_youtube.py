"""Download a short segment of an NBA video from YouTube (works outside the US).

The official NBA clip API (fetch_nba_clips.py) is geo-locked — from many regions
it returns a "VIDEO NOT AVAILABLE" placeholder. YouTube isn't geo-locked, so this
grabs real broadcast footage for the pause-to-overlay demo. You lose the paired
play-by-play labels, but you get real pixels to run the pipeline on.

Keep clips short (personal/research fair use, per the roadmap).

    python scripts/fetch_youtube.py "https://youtube.com/watch?v=..." --start 1:05 --end 1:20
    # -> data/video/clips/youtube/<id>.mp4

Prefer a URL that shows continuous main-camera half-court action (a possession),
not a fast-cut highlight reel — the calibration assumes one camera shot.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Download a YouTube clip segment for the demo")
    ap.add_argument("url")
    ap.add_argument("--start", default=None, help="start time, e.g. 1:05 (default: from 0)")
    ap.add_argument("--end", default=None, help="end time, e.g. 1:20 (default: +20s)")
    ap.add_argument("--out-dir", default="data/video/clips/youtube")
    ap.add_argument("--height", type=int, default=720, help="max video height")
    args = ap.parse_args(argv)

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable, "-m", "yt_dlp",
        "-f", f"bestvideo[height<={args.height}][ext=mp4]+bestaudio/best[height<={args.height}]",
        "--merge-output-format", "mp4",
        "-o", str(out / "%(id)s.%(ext)s"),
    ]
    if args.start or args.end:
        s = args.start or "00:00"
        e = args.end or "99:59"
        cmd += ["--download-sections", f"*{s}-{e}", "--force-keyframes-at-cuts"]
    cmd.append(args.url)

    print("running:", " ".join(cmd[2:]))
    rc = subprocess.call(cmd)
    if rc == 0:
        mp4s = sorted(out.glob("*.mp4"))
        if mp4s:
            print(f"\nDownloaded -> {mp4s[-1]}")
            print("Next: calibrate a frame, then run the overlay:")
            print(f"  python scripts/calibrate.py --video {mp4s[-1]} --time 2 --out calib.json")
            print(f"  python scripts/demo_realvideo.py --video {mp4s[-1]} --calib calib.json --pause 8")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
