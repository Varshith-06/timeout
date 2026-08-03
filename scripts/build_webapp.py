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
    ap.add_argument("--calib", help="single calibration.json (one camera shot)")
    ap.add_argument("--shots", nargs="+",
                    help="multiple calibration files, each carrying its own click-time "
                         "(calibrate.py --time). Covers several camera shots -> more of the game.")
    ap.add_argument("--models", default="models/phase2")
    ap.add_argument("--out", default="out/webapp")
    ap.add_argument("--calib-time", type=float, default=39.0,
                    help="click-time (s) for a single --calib that has none stored; each shot's "
                         "homography propagates FORWARD from its click-time within one camera shot")
    ap.add_argument("--stride", type=int, default=8, help="detection stride (video frames)")
    ap.add_argument("--context", type=float, default=2.0,
                    help="seconds of context each side of a pause for local tracking/roster")
    ap.add_argument("--cadence", type=float, default=1.0, help="seconds between exported recommendations")
    ap.add_argument("--max-seconds", type=float, default=None, help="absolute end time (s); default = clip end")
    ap.add_argument("--no-jersey", action="store_true", help="skip jersey-number OCR (faster)")
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

    from src.perception.synthetic_broadcast import BroadcastClip, BroadcastFrame

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    vid = VideoSource(args.video)
    duration = vid.frame_count / vid.fps
    hard_end = min(duration, args.max_seconds) if args.max_seconds else duration
    sec_per_frame = args.stride / vid.fps
    half = max(1, int(round(args.context / sec_per_frame)))
    print(f"video: {vid.width}x{vid.height}, {vid.fps:.0f} fps, {duration:.1f}s")

    if args.detector == "roboflow":
        detector = RoboflowWorkflowDetector(workspace=args.rf_workspace, workflow_id=args.rf_workflow)
    else:
        detector = PretrainedYOLODetector()
        print(f"detector: local YOLO ({detector.weights})")
    sm = SubModels.load(args.models)
    vm = ValueModel.load(Path(args.models) / "value.pt")

    # Assemble the shot list: each calibration covers one camera shot, anchored at
    # its click-time. More shots -> more of the game is covered.
    shots = []
    if args.shots:
        for f in args.shots:
            c = Calibration.load(f)
            if c.time is None:
                print(f"{f} has no stored click-time — recalibrate with "
                      f"'calibrate.py --time <sec>' so it knows which shot it anchors."); return 2
            shots.append((c, f, float(c.time)))
    elif args.calib:
        c = Calibration.load(args.calib)
        shots.append((c, args.calib, float(c.time if c.time is not None else args.calib_time)))
    else:
        print("provide --calib (one shot) or --shots a.json b.json ... (multiple shots)"); return 2
    shots.sort(key=lambda s: s[2])

    def process_shot(calib, start_sec, end_bound):
        """Detect+track one camera shot from its calibration, export its pauses."""
        clip = build_realvideo_clip(vid, detector, calib, pause_sec=end_bound,
                                    window_sec=end_bound - start_sec, stride=args.stride)
        # The homography is valid only within this shot — truncate at the first cut.
        cut_idx = next((i for i, f in enumerate(clip.frames) if i > 0 and f.cut), None)
        shot_end = end_bound
        if cut_idx is not None:
            clip.frames = clip.frames[:cut_idx]
            shot_end = start_sec + cut_idx * sec_per_frame
        if not args.no_jersey:
            try:
                from src.perception.jersey import assign_jerseys
                assign_jerseys(clip.frames)
            except Exception as e:
                print(f"  jersey OCR unavailable ({type(e).__name__}); players stay unnamed")

        def subclip(lo, hi):
            sub = BroadcastClip()
            for j, f in enumerate(clip.frames[lo:hi]):
                sub.frames.append(BroadcastFrame(
                    frame_idx=j, camera=f.camera, detections=f.detections, kp_pixels=f.kp_pixels,
                    kp_court=f.kp_court, kp_conf=f.kp_conf, clock_read=f.clock_read, cut=f.cut, image=f.image))
            return sub

        recs, t = [], start_sec
        while t <= shot_end:
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
                recs.append(build_overlay_spec(lrec, sub, ci, state, scored, names={}, video_time=round(t, 2)))
            else:
                recs.append({"video_time": round(t, 2), "frame_idx": i, "gate": False,
                             "confidence": round(float(conf), 3)})
            t += args.cadence
        return recs, shot_end

    recommendations = []
    for k, (calib, fname, t0) in enumerate(shots):
        end_bound = min(hard_end, shots[k + 1][2] if k + 1 < len(shots) else hard_end)
        if end_bound - t0 < args.cadence:
            continue
        print(f"shot {k+1}/{len(shots)}: {Path(fname).name} @ {t0:.1f}s (up to {end_bound:.1f}s)...")
        recs, shot_end = process_shot(calib, t0, end_bound)
        recommendations.extend(recs)
        shown = sum(1 for r in recs if r.get("actions"))
        print(f"  covered {t0:.1f}-{shot_end:.1f}s: {shown}/{len(recs)} pauses with a recommendation")
    recommendations.sort(key=lambda r: r["video_time"])
    vid.release()
    n_shown = sum(1 for r in recommendations if r.get("actions"))

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
