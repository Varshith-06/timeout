"""Does per-frame calibration actually beat the propagated one? (roadmap 4.6)

    python scripts/eval_keypoints.py --video clip.mp4 --calib calib.json

Walks a camera shot from its click-frame and, at each sampled frame, compares
the two ways of getting a court homography:

* **propagated** — the manual calibration carried forward by optical flow, which
  is what the pipeline does today. Its error is expected to grow with distance
  from the anchor; that growth *is* the drift this model exists to remove.
* **per-frame** — solved from the keypoint model's predictions alone, with no
  knowledge of the calibration. Its error should be roughly flat in time.

Both are reported as median reprojection error in feet, the same quantity
``solve_homography`` gates on at 2 ft. Neither is ground truth — the propagated
homography is scored against its own tracked points, the model against its own
predictions — so a low number means "self-consistent", not "correct". The
overlay image is the check on correctness: run with ``--render`` and look at
whether the lines sit on the real court.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.perception.calibrate import Calibration, HomographyTracker, _LANDMARK_COURT  # noqa: E402
from src.perception.cuts import CutDetector                                # noqa: E402
from src.perception.homography import (plausible_court_homography,         # noqa: E402
                                       solve_homography)
from src.perception.keypoint_model import CourtKeypointDetector, DEFAULT_WEIGHTS  # noqa: E402
from src.perception.video import VideoSource                               # noqa: E402


def _thumb(rgb):
    import cv2
    return cv2.resize(rgb, (64, 36))


def _tracker_reproj_ft(tracker):
    import cv2
    if tracker.H is None:
        return float("nan")
    proj = cv2.perspectiveTransform(
        tracker.pts.reshape(-1, 1, 2).astype(np.float64), tracker.H).reshape(-1, 2)
    err = np.linalg.norm(proj - tracker.court, axis=1)
    return float(np.median(err)) if len(err) else float("nan")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", required=True)
    ap.add_argument("--calib", required=True)
    ap.add_argument("--weights", default=DEFAULT_WEIGHTS)
    ap.add_argument("--stride", type=int, default=10, help="frames between samples")
    ap.add_argument("--max-span", type=float, default=25.0, help="seconds to walk forward")
    ap.add_argument("--render", default=None, metavar="PNG",
                    help="write a side-by-side overlay grid here")
    ap.add_argument("--panels", type=int, default=6, help="frames to draw when rendering")
    args = ap.parse_args(argv)

    video = VideoSource(args.video)
    calib = Calibration.load(args.calib)
    if calib.time is None:
        print("calibration has no --time; cannot locate its shot")
        return 2
    det = CourtKeypointDetector(weights=args.weights)
    if not det.available:
        print(f"no weights at {args.weights}; train with scripts/train_keypoints.py")
        return 2

    anchor = int(round(calib.time * video.fps))
    frame = video.frame_at_index(anchor)
    if frame is None:
        print("could not read the anchor frame")
        return 2

    tracker = HomographyTracker(video.gray(frame), calib)
    cutdet = CutDetector(threshold=0.5)
    cutdet.update(_thumb(frame))

    rows, panels = [], []
    n_steps = int(args.max_span * video.fps / args.stride)
    for n in range(n_steps + 1):
        idx = anchor + n * args.stride
        if idx >= video.frame_count:
            break
        f = video.frame_at_index(idx)
        if f is None:
            break
        if n > 0:
            if cutdet.update(_thumb(f)):
                print(f"camera cut at t={idx / video.fps:.1f}s — end of shot")
                break
            if tracker.update(video.gray(f)) is None:
                print(f"tracker lost at t={idx / video.fps:.1f}s")
                break

        prop_ft = _tracker_reproj_ft(tracker)
        prop_n = int(len(tracker.pts))
        H_prop = tracker.H

        px, ct, cf = det.predict(f)
        res = solve_homography(px, ct, cf) if len(px) >= 4 else None
        model_ft = res.median_reproj_ft if res is not None else float("nan")
        model_n = res.n_points if res is not None else 0
        # Report what the pipeline would actually accept, which is not just
        # "did it solve": a self-consistent but geometrically absurd homography
        # is exactly the failure this model can produce.
        accepted = bool(res is not None and res.ok and res.n_points >= 6
                        and plausible_court_homography(res.H, (video.width, video.height)))
        H_model = res.H if accepted else None

        rows.append((idx / video.fps, idx - anchor, prop_ft, prop_n, model_ft,
                     model_n, len(px), accepted))
        if args.render and len(panels) < args.panels:
            step = max(1, (n_steps + 1) // args.panels)
            if n % step == 0:
                panels.append((idx / video.fps, f, H_prop, H_model))

    video.release()

    # The point counts are not decoration. A homography through 4 points has zero
    # residual by construction, so a small error backed by 4 points means "fewer
    # correspondences survived", not "more accurate" — true of the propagated
    # tracker as it loses points, and of RANSAC when its consensus set collapses.
    print(f"\n{'t(s)':>7} {'+frames':>8} {'propagated':>12} {'n':>3} "
          f"{'per-frame':>11} {'inl':>4} {'kp':>4} {'solved':>7}")
    for t, df, p, pn, m, mn, nk, ok in rows:
        print(f"{t:7.1f} {df:8d} {p:10.2f}ft {pn:3d} {m:9.2f}ft {mn:4d} {nk:4d} "
              f"{str(ok):>7}")

    if rows:
        prop = np.array([r[2] for r in rows], dtype=float)
        mod = np.array([r[4] for r in rows], dtype=float)
        solved = sum(1 for r in rows if r[7])
        print(f"\nframes: {len(rows)}   model solved within the 2 ft gate: "
              f"{solved}/{len(rows)} ({100.0*solved/len(rows):.0f}%)")
        with np.errstate(invalid="ignore"):
            print(f"propagated  median {np.nanmedian(prop):.2f} ft   "
                  f"first {prop[0]:.2f} -> last {prop[-1]:.2f}")
            print(f"per-frame   median {np.nanmedian(mod):.2f} ft   "
                  f"first {mod[0]:.2f} -> last {mod[-1]:.2f}")
        # The signature of drift: propagated error grows with distance from the
        # anchor while the per-frame error does not.
        if len(rows) > 4:
            half = len(rows) // 2
            print(f"drift check — propagated {np.nanmedian(prop[:half]):.2f} -> "
                  f"{np.nanmedian(prop[half:]):.2f} ft, "
                  f"per-frame {np.nanmedian(mod[:half]):.2f} -> "
                  f"{np.nanmedian(mod[half:]):.2f} ft")

    if args.render and panels:
        _render(panels, args.render)
        print(f"\noverlay -> {args.render}   (yellow = propagated, cyan = per-frame)")
    return 0


def _render(panels, out_path):
    import cv2
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from src.perception.video_overlay import court_polylines

    cols = min(3, len(panels))
    rows_n = (len(panels) + cols - 1) // cols
    fig, axes = plt.subplots(rows_n, cols, figsize=(7 * cols, 4 * rows_n), squeeze=False)
    for ax, (t, frame, H_prop, H_model) in zip(axes.ravel(), panels):
        ax.imshow(frame)
        ax.axis("off")
        for H, colour in ((H_prop, "yellow"), (H_model, "cyan")):
            if H is None:
                continue
            H_c2p = np.linalg.inv(H)
            for poly in court_polylines():
                px = cv2.perspectiveTransform(
                    poly.reshape(-1, 1, 2).astype(np.float64), H_c2p).reshape(-1, 2)
                ax.plot(px[:, 0], px[:, 1], color=colour, lw=1.4, alpha=0.85)
        ax.set_xlim(0, frame.shape[1])
        ax.set_ylim(frame.shape[0], 0)
        ax.set_title(f"t = {t:.1f}s", fontsize=10)
    for ax in axes.ravel()[len(panels):]:
        ax.axis("off")
    plt.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=66, bbox_inches="tight")


if __name__ == "__main__":
    raise SystemExit(main())
