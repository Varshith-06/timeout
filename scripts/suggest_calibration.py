"""Suggest good calibration timestamps in a clip (for multi-shot builds).

Calibration wants a clean main-camera frame that shows the court lines. This scans
the video, scores each sampled frame by how much *court* (wood-tone floor) is
visible, segments the video into camera shots (colour-histogram cuts), and prints
the best-scoring frame in each of the most court-heavy shots — well spread in time.
Saves a thumbnail per suggestion so you can eyeball it before clicking.

    python scripts/suggest_calibration.py --video game.mp4 --n 4
    # -> prints timestamps + writes out/calib_suggest/*.png

Then calibrate each printed timestamp and build with --shots (see the README).
"""
from __future__ import annotations

import argparse
import io
import sys
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace", line_buffering=True)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402


def _court_fraction(rgb) -> float:
    """Fraction of pixels that look like a wood court floor (tan/orange, mid-sat)."""
    import cv2
    hsv = cv2.cvtColor(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), cv2.COLOR_BGR2HSV)
    H, S, V = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    mask = (H > 5) & (H < 30) & (S > 60) & (V > 90)
    return float(mask.mean())


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Suggest calibration timestamps for multi-shot")
    ap.add_argument("--video", required=True)
    ap.add_argument("--every", type=float, default=1.0, help="sample interval (seconds)")
    ap.add_argument("--n", type=int, default=4, help="how many timestamps to suggest")
    ap.add_argument("--min-court", type=float, default=0.18, help="min court fraction to consider")
    ap.add_argument("--min-players", type=int, default=6, help="min detected players (rejects close-ups)")
    ap.add_argument("--out", default="out/calib_suggest")
    args = ap.parse_args(argv)

    import cv2
    from src.perception.cuts import CutDetector
    from src.perception.video import VideoSource

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    vid = VideoSource(args.video)
    duration = vid.frame_count / vid.fps
    step = max(1, int(round(args.every * vid.fps)))
    print(f"scanning {duration:.0f}s of {args.video} every {args.every:.1f}s...")

    cutdet = CutDetector(threshold=0.5)
    shots, cur = [], []            # each shot: list of (time, court_frac, frame_idx)
    for f in range(0, vid.frame_count, step):
        rgb = vid.frame_at_index(f)
        if rgb is None:
            continue
        if cutdet.update(cv2.resize(rgb, (64, 36))) and cur:
            shots.append(cur); cur = []
        cur.append((f / vid.fps, _court_fraction(rgb), f))
    if cur:
        shots.append(cur)

    # Best (most court) frame per shot; keep shots that actually show the court.
    best = []
    for shot in shots:
        t, frac, fi = max(shot, key=lambda r: r[1])
        if frac >= args.min_court:
            best.append((t, frac, fi))

    # Reject close-ups / tight shots: a main-camera calibration frame shows many
    # players (wide view). Gate each candidate by player count (cheap: one detect
    # per shot, not per frame).
    from src.perception.video import PretrainedYOLODetector
    det = PretrainedYOLODetector()
    gated = []
    for t, frac, fi in best:
        rgb = vid.frame_at_index(fi)
        n = len(det.detect(rgb, 0).by_class("player"))
        if n >= args.min_players:
            gated.append((t, frac, fi, n))
    # Prefer many players (wide, main-camera) then court visibility.
    gated.sort(key=lambda r: (-min(r[3], 12), -r[1]))
    best = [(t, frac, fi) for t, frac, fi, _ in gated]

    # Greedily pick N, spread out in time (>= 8s apart) so they're distinct plays.
    picks = []
    for t, frac, fi in best:
        if all(abs(t - pt) >= 8.0 for pt, _, _ in picks):
            picks.append((t, frac, fi))
        if len(picks) >= args.n:
            break
    picks.sort(key=lambda r: r[0])

    if not picks:
        print("No clean court frames found — try a different segment or lower --min-court.")
        vid.release(); return 1

    print(f"\n{len(picks)} suggested calibration timestamps (court-visibility in %):")
    for k, (t, frac, fi) in enumerate(picks, 1):
        rgb = vid.frame_at_index(fi)
        cv2.imwrite(str(out / f"shot{k}_t{t:.0f}s.png"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        print(f"  shot {k}: --time {t:.1f}   (court {frac*100:.0f}%)  -> {out}/shot{k}_t{t:.0f}s.png")
    vid.release()

    times = " ".join(f"shot{k}.json" for k in range(1, len(picks) + 1))
    print("\nCalibrate each (a court diagram guides the clicks), then build:")
    for k, (t, _, _) in enumerate(picks, 1):
        print(f"  python scripts/calibrate.py --video {args.video} --time {t:.1f} --out shot{k}.json")
    print(f"  python scripts/build_webapp.py --video {args.video} --shots {times}")
    print("  python scripts/serve_webapp.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
