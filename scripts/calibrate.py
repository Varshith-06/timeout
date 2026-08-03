"""Calibrate a broadcast frame: click court landmarks to solve the homography.

Opens one frame from a video (or an image), guides you to click known court
landmarks, solves the pixel->court homography, shows the projected court lines
for you to verify, and saves the result.

    # from a video at a timestamp (main-camera, half-court frame is best)
    python scripts/calibrate.py --video game.mp4 --time 63.5 --out calib.json

    # or from a saved screenshot
    python scripts/calibrate.py --image frame.png --out calib.json

How to click: a court diagram highlights one landmark at a time — click the SAME
spot in the video frame. Click 6-8 well-spread line intersections; right-click to
skip any that are off-screen. Don't pick points that all lie on one line.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Court homography calibration")
    ap.add_argument("--video", type=str, default=None)
    ap.add_argument("--time", type=float, default=0.0, help="seconds into the video")
    ap.add_argument("--frame", type=int, default=None, help="frame index (overrides --time)")
    ap.add_argument("--image", type=str, default=None, help="calibrate from an image file instead")
    ap.add_argument("--out", type=str, default="calibration.json")
    args = ap.parse_args(argv)

    from src.perception.calibrate import interactive_calibrate

    click_time = None
    if args.image:
        import matplotlib.image as mpimg
        frame = (mpimg.imread(args.image)[..., :3] * (255 if mpimg.imread(args.image).max() <= 1 else 1)).astype("uint8")
    elif args.video:
        from src.perception.video import VideoSource
        vid = VideoSource(args.video)
        print(f"video: {vid.width}x{vid.height}, {vid.fps:.1f} fps, {vid.frame_count} frames")
        frame = vid.frame_at_index(args.frame) if args.frame is not None else vid.frame_at_time(args.time)
        # The click-time anchors this calibration's camera shot (for multi-shot builds).
        click_time = (args.frame / vid.fps) if args.frame is not None else args.time
        vid.release()
        if frame is None:
            print("could not read that frame"); return 1
    else:
        print("provide --video or --image"); return 1

    calib = interactive_calibrate(np.asarray(frame), out_path=args.out, time=click_time)
    return 0 if calib is not None else 1


if __name__ == "__main__":
    raise SystemExit(main())
