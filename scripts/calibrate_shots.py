"""Calibrate several camera shots in one guided sequence (for multi-shot builds).

Opens the calibration window for each timestamp one after another — click the
guided landmarks, close the verify window, and it advances to the next shot —
saving shot1.json, shot2.json, ... Then build with --shots (see the README).

    python scripts/calibrate_shots.py --video game.mp4 --times 1.5 42 86 162
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Guided multi-shot calibration")
    ap.add_argument("--video", required=True)
    ap.add_argument("--times", nargs="+", type=float, required=True, help="video times (s), one per shot")
    ap.add_argument("--prefix", default="shot", help="output prefix -> shot1.json, shot2.json, ...")
    args = ap.parse_args(argv)

    from src.perception.calibrate import interactive_calibrate
    from src.perception.video import VideoSource

    vid = VideoSource(args.video)
    print(f"video: {vid.width}x{vid.height}, {vid.fps:.1f} fps — {len(args.times)} shots to calibrate")
    saved = []
    for i, t in enumerate(args.times, 1):
        out = f"{args.prefix}{i}.json"
        print(f"\n=== shot {i}/{len(args.times)} @ {t:.1f}s -> {out} ===")
        print("L-click landmark · R-click skip point · u=undo · Enter=solve, then A=accept / R=redo. "
              "For a close-up/replay with no clean lines, press Esc to skip the whole shot.")
        frame = vid.frame_at_time(t)
        if frame is None:
            print(f"  could not read frame at {t:.1f}s — skipping"); continue
        calib = interactive_calibrate(np.asarray(frame), out_path=out, time=t)
        if calib is not None:
            saved.append(out)
    vid.release()

    if saved:
        print("\nCalibrated:", " ".join(saved))
        print("Now build the app:")
        print(f"  python scripts/build_webapp.py --video {args.video} --shots {' '.join(saved)}")
        print("  python scripts/serve_webapp.py")
    else:
        print("\nNo calibrations saved.")
    return 0 if saved else 1


if __name__ == "__main__":
    raise SystemExit(main())
