"""Pre-compute the pause-to-recommendation data for the web app.

Runs the full trained pipeline over a clip **once** (one detection pass at a
stride), then exports a recommendation at a fixed cadence into a JSON the browser
plays back instantly — no live inference. The web UI (out/webapp/index.html) loads
this JSON and the video and draws the overlay on pause.

    python scripts/build_webapp.py --video clip.mp4 --calib calib.json
    python scripts/serve_webapp.py            # then open the printed URL

Detection is the bottleneck, so it defaults to the local YOLO detector (free,
offline). --detector roboflow uses the hosted workflow (one call per sampled
frame — slow over a whole clip).
"""
from __future__ import annotations

import argparse
import io
import json
import shutil
import sys
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace", line_buffering=True)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Pre-compute web-app recommendations for a clip")
    ap.add_argument("--video", required=True)
    ap.add_argument("--calib", required=True)
    ap.add_argument("--models", default="models/phase2")
    ap.add_argument("--out", default="out/webapp")
    ap.add_argument("--calib-time", type=float, default=39.0,
                    help="video time (s) the calibration was clicked; the homography is "
                         "propagated FORWARD from here (one camera shot), so processing starts here")
    ap.add_argument("--stride", type=int, default=8, help="detection stride (video frames)")
    ap.add_argument("--context", type=float, default=2.0,
                    help="seconds of context each side of a pause for local tracking/roster")
    ap.add_argument("--cadence", type=float, default=1.0, help="seconds between exported recommendations")
    ap.add_argument("--max-seconds", type=float, default=None, help="absolute end time (s); default = clip end")
    ap.add_argument("--detector", choices=["yolo", "roboflow"], default="yolo")
    ap.add_argument("--rf-workspace", default="varshith-ublcu")
    ap.add_argument("--rf-workflow", default="general-segmentation-api")
    args = ap.parse_args(argv)

    from src.app.webexport import build_overlay_spec
    from src.perception.calibrate import Calibration
    from src.perception.realvideo import build_realvideo_clip
    from src.perception.state_from_cv import build_state_from_cv, is_showable, recover_tracking
    from src.perception.video import PretrainedYOLODetector, RoboflowWorkflowDetector, VideoSource
    from src.value.actions import enumerate_actions
    from src.value.scoring import score_actions
    from src.value.state_value import ValueModel
    from src.value.submodels import SubModels

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    vid = VideoSource(args.video)
    duration = vid.frame_count / vid.fps
    start_sec = max(0.0, args.calib_time)
    end = min(duration, args.max_seconds) if args.max_seconds else duration
    print(f"video: {vid.width}x{vid.height}, {vid.fps:.0f} fps, {duration:.1f}s "
          f"(processing {start_sec:.1f}-{end:.1f}s from the calibration frame, stride {args.stride})")

    if args.detector == "roboflow":
        detector = RoboflowWorkflowDetector(workspace=args.rf_workspace, workflow_id=args.rf_workflow)
    else:
        detector = PretrainedYOLODetector()
        print(f"detector: local YOLO ({detector.weights})")

    calib = Calibration.load(args.calib)
    sm = SubModels.load(args.models)
    vm = ValueModel.load(Path(args.models) / "value.pt")

    # One detection + tracking pass over [calib_time, end]: the tracker initialises
    # from the calibration's clicked points at the FIRST processed frame (= the
    # calibration frame), then optical-flow-propagates the homography forward.
    print("Detecting + tracking from the calibration frame forward (one pass)...")
    clip = build_realvideo_clip(vid, detector, calib, pause_sec=end,
                                window_sec=end - start_sec, stride=args.stride)
    vid.release()
    sec_per_frame = args.stride / vid.fps

    # The single calibration + optical-flow homography is valid only within ONE
    # camera shot. Past the first replay/angle cut the homography is meaningless,
    # so truncate there — that's the coherent segment the calibration covers.
    cut_idx = next((i for i, f in enumerate(clip.frames) if i > 0 and f.cut), None)
    if cut_idx is not None:
        cut_t = start_sec + cut_idx * sec_per_frame
        clip.frames = clip.frames[:cut_idx]
        end = min(end, cut_t)
        print(f"camera cut at ~{cut_t:.1f}s — limiting to the calibrated shot "
              f"({start_sec:.1f}-{cut_t:.1f}s, {len(clip.frames)} frames)")

    from src.perception.synthetic_broadcast import BroadcastClip, BroadcastFrame

    def subclip(lo: int, hi: int) -> BroadcastClip:
        """A re-indexed local window (frame_idx 0..n) reusing the detections/kp."""
        sub = BroadcastClip()
        for j, f in enumerate(clip.frames[lo:hi]):
            sub.frames.append(BroadcastFrame(
                frame_idx=j, camera=f.camera, detections=f.detections, kp_pixels=f.kp_pixels,
                kp_court=f.kp_court, kp_conf=f.kp_conf, clock_read=f.clock_read, cut=f.cut, image=f.image))
        return sub

    # State is built on a LOCAL window per pause, not the whole shot: the roster
    # gate and velocities assume a short window, and the homography is most
    # trustworthy near each moment. Detection already ran once; re-running only the
    # cheap tracking/roster per window is fast.
    half = max(1, int(round(args.context / sec_per_frame)))
    recommendations = []
    n_shown = 0
    t = start_sec
    while t <= end:
        i = int(round((t - start_sec) / sec_per_frame))
        if i >= len(clip.frames):
            break
        lo, hi = max(0, i - half), min(len(clip.frames), i + half + 1)
        sub = subclip(lo, hi)
        lrec = recover_tracking(sub, roster_rows=[], stride=args.stride)
        ci = i - lo
        state, conf = build_state_from_cv(lrec, ci)
        if state is not None and is_showable(state, conf) and state.handler is not None:
            scored = score_actions(state, enumerate_actions(state), sm, vm)
            spec = build_overlay_spec(lrec, sub, ci, state, scored, names={}, video_time=round(t, 2))
            recommendations.append(spec)
            n_shown += 1
        else:
            recommendations.append({"video_time": round(t, 2), "frame_idx": i, "gate": False,
                                    "confidence": round(float(conf), 3)})
        t += args.cadence
    print(f"processed {len(clip.frames)} frames in {half*2+1}-frame windows")

    # Copy the video next to the JSON so the app serves both from one directory.
    dst = out / "clip.mp4"
    if Path(args.video).resolve() != dst.resolve():
        shutil.copyfile(args.video, dst)
    payload = {
        "video": "clip.mp4",
        "video_size": [vid.width, vid.height],
        "fps": vid.fps,
        "duration": round(duration, 2),
        "recommendations": recommendations,
    }
    (out / "recommendations.json").write_text(json.dumps(payload), encoding="utf-8")
    # Ship the (version-controlled) web UI next to the data.
    ui = Path(__file__).resolve().parents[1] / "src" / "app" / "webapp" / "index.html"
    shutil.copyfile(ui, out / "index.html")
    print(f"exported {len(recommendations)} pause points ({n_shown} with a recommendation) "
          f"-> {out/'recommendations.json'}")
    print(f"video -> {dst}\nui -> {out/'index.html'}\nNext: python scripts/serve_webapp.py --dir {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
