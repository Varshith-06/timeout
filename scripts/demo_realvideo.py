"""Real-video pause-to-overlay: video + calibration -> state -> EPV -> overlay.

Ties the whole real-video path together:
  broadcast clip --YOLO--> detections --calibration H--> court positions
  --tracking/teams--> State --value model--> EPV-scored actions --> overlay.

It reuses the exact perception spine and value model validated on synthetic and
SportVU data — the only real-video-specific pieces are the video reader
(`src/perception/video.py`) and the manual court calibration
(`scripts/calibrate.py`). Feed the calibration's clicked landmarks in as
"keypoints" each frame (optical-flow tracked), so `recover_tracking` solves the
same homography and everything downstream runs unchanged.

    python scripts/calibrate.py --video clip.mp4 --time 2.0 --out calib.json
    python scripts/train_phase2.py                      # once, for models/phase2
    python scripts/demo_realvideo.py --video clip.mp4 --calib calib.json --pause 6.0

Note: NBA serves region-locked "VIDEO NOT AVAILABLE" placeholders to some
connections. Fetch clips from a US home connection (`fetch_nba_clips.py`); the
demo detects a placeholder and tells you rather than producing an empty overlay.
"""
from __future__ import annotations

import argparse
import io
import sys
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace", line_buffering=True)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Real-video pause-to-overlay demo")
    ap.add_argument("--video", required=True)
    ap.add_argument("--calib", required=True, help="calibration.json from scripts/calibrate.py")
    ap.add_argument("--pause", type=float, default=5.0, help="pause timestamp (seconds)")
    ap.add_argument("--window", type=float, default=4.0, help="seconds of context before the pause")
    ap.add_argument("--stride", type=int, default=5)
    ap.add_argument("--models", default="models/phase2")
    ap.add_argument("--out", default="out/realvideo")
    ap.add_argument("--detector", choices=["yolo", "roboflow"], default="yolo",
                    help="yolo = local (COCO or models/basketball.pt); "
                         "roboflow = hosted workflow (needs $ROBOFLOW_API_KEY)")
    ap.add_argument("--rf-workspace", default="varshith-ublcu")
    ap.add_argument("--rf-workflow", default="general-segmentation-api")
    ap.add_argument("--rf-classes", default="ball, basket, person")
    args = ap.parse_args(argv)

    from src.perception.calibrate import Calibration
    from src.perception.realvideo import build_realvideo_clip, looks_like_placeholder
    from src.perception.state_from_cv import (build_state_from_cv, pick_showable_frame,
                                              recover_tracking)
    from src.perception.video import (PretrainedYOLODetector, RoboflowWorkflowDetector,
                                      VideoSource)

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    vid = VideoSource(args.video)
    print(f"video: {vid.width}x{vid.height}, {vid.fps:.0f} fps, {vid.frame_count} frames")

    if args.detector == "roboflow":
        detector = RoboflowWorkflowDetector(workspace=args.rf_workspace,
                                            workflow_id=args.rf_workflow, classes=args.rf_classes)
        print(f"detector: Roboflow workflow {args.rf_workspace}/{args.rf_workflow} "
              f"(one network call per frame — larger --stride is faster)")
    else:
        detector = PretrainedYOLODetector()
        print(f"detector: local YOLO ({detector.weights})")
    if looks_like_placeholder(vid, detector):
        print("This clip is an NBA 'VIDEO NOT AVAILABLE' placeholder, not game footage.\n"
              "Fetch real clips from a US home connection (fetch_nba_clips.py).")
        return 2

    calib = Calibration.load(args.calib)
    print("Detecting + tracking the context window (pretrained YOLO, CPU)...")
    clip = build_realvideo_clip(vid, detector, calib, args.pause, args.window, args.stride)
    vid.release()

    rec = recover_tracking(clip, roster_rows=[], stride=args.stride, fps=vid.fps)
    d = rec.diagnostics
    print(f"recovered: homography valid {d['homog_valid_rate']:.0%}, "
          f"mean {d['mean_players']:.1f} players, {d['n_tracklets']} tracklets")

    fi = pick_showable_frame(rec)
    if fi is None:
        print("No frame passed the confidence gate — too few players tracked "
              "(occlusion/replay), so no recommendation is shown. This is the honest gate.")
        return 0
    state, conf = build_state_from_cv(rec, fi)
    print(f"state at frame {fi}: confidence {conf:.2f}, {state.context.n_players_observed} players")

    actions_scored = None
    if (Path(args.models) / "value.pt").exists() and state.handler is not None:
        from src.value.actions import enumerate_actions
        from src.value.scoring import score_actions
        from src.value.state_value import ValueModel
        from src.value.submodels import SubModels
        sm = SubModels.load(args.models); vm = ValueModel.load(Path(args.models) / "value.pt")
        actions_scored = score_actions(state, enumerate_actions(state), sm, vm)
        top = actions_scored[0]
        print(f"recommendation: {top.action.action} at EPV {top.q:.2f}")

        from src.perception.state_from_cv import frame_player_pixels
        from src.perception.video_overlay import render_video_overlay
        H = rec.homographies[fi]
        player_feet, ball_px = frame_player_pixels(rec, clip, fi)
        png = out / "realvideo_overlay.png"
        render_video_overlay(clip.frames[fi], state, actions_scored, H, player_feet, ball_px, png)
        print(f"-> {png}")
    else:
        print("(no value model / no ball handler — run train_phase2.py; recommendation skipped)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
