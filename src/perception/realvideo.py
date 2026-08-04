"""Adapt real broadcast video into the perception spine (roadmap 4.1 / 5.5).

Turns a window of real video frames into the same ``BroadcastClip`` the synthetic
pipeline produces, so :func:`src.perception.state_from_cv.recover_tracking` and
everything downstream run unchanged. The trick: feed the manual calibration's
clicked landmarks in as per-frame "keypoints" (optical-flow tracked), so the
homography solver recovers the propagated court mapping for each frame.
"""
from __future__ import annotations

import numpy as np

from src.perception.calibrate import Calibration, HomographyTracker, _LANDMARK_COURT
from src.perception.camera import BroadcastCamera
from src.perception.cuts import CutDetector
from src.perception.synthetic_broadcast import BroadcastClip, BroadcastFrame


def looks_like_placeholder(video, detector=None) -> bool:
    """True if the clip is the NBA 'VIDEO NOT AVAILABLE' card, not game footage.

    The card is animated (a moving gradient), so frame motion doesn't reveal it —
    but a detector finds ~no people on it, whereas real footage has several
    players on every frame. Falls back to a motion check if no detector is given.
    """
    fracs = [0.3, 0.5, 0.7]
    frames = [video.frame_at_index(int(video.frame_count * f)) for f in fracs]
    frames = [f for f in frames if f is not None]
    if not frames:
        return False
    if detector is not None:
        players = max(len(detector.detect(f, i).by_class("player")) for i, f in enumerate(frames))
        return players <= 1     # real NBA frames show several players
    # No detector: fall back to a near-static-image check.
    if len(frames) < 2:
        return False
    return float(np.mean(np.abs(frames[0].astype(int) - frames[1].astype(int)))) < 3.0


def build_realvideo_clip(video, detector, calibration: Calibration,
                         pause_sec: float, window_sec: float, stride: int,
                         detect_workers: int = 1, exclude_refs: bool = True,
                         ball_seed=None) -> BroadcastClip:
    """Build a BroadcastClip from the ``window_sec`` of video before ``pause_sec``.

    ``detect_workers`` > 1 runs the (independent) per-frame detections concurrently
    — a big win for a network detector (Roboflow) where each call is I/O-bound.
    Tracking stays sequential (it depends on the previous frame). ``exclude_refs``
    drops referee-looking (striped) person detections. ``ball_seed`` is a pixel
    (x, y) the user clicked for the ball at the first frame — it overrides the
    (often wrong) initial ball detection so the tracker starts locked on.
    """
    import cv2
    from src.perception.detection import Detection, FrameDetections
    from src.perception.video import looks_like_ref
    W, H = video.width, video.height
    cam = BroadcastCamera(img_w=W, img_h=H)
    start = max(0, int((pause_sec - window_sec) * video.fps))
    end = int(pause_sec * video.fps)

    names = list(calibration.points.keys())
    court_pts = np.array([_LANDMARK_COURT[n] for n in names], dtype=float)

    # Read the frames (sequential disk decode), then detect (optionally parallel).
    frames = []                                    # (idx, rgb, gray)
    idx = 0
    for f in range(start, end + 1, stride):
        rgb = video.frame_at_index(f)
        if rgb is None:
            continue
        frames.append((idx, rgb, video.gray(rgb)))
        idx += 1

    if detect_workers > 1 and len(frames) > 1:
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=detect_workers) as ex:
            dets = list(ex.map(lambda t: detector.detect(t[1], t[0]), frames))
    else:
        dets = [detector.detect(rgb, i) for i, rgb, _ in frames]

    # Drop referee-looking players, and seed the ball on the first frame.
    for (i, rgb, _), fd in zip(frames, dets):
        kept = []
        for d in fd.detections:
            if d.cls == "player" and exclude_refs and looks_like_ref(rgb, d.bbox):
                continue                                   # referee -> drop
            if d.cls == "ball" and i == 0 and ball_seed is not None:
                continue                                   # replaced by the seed below
            kept.append(d)
        if i == 0 and ball_seed is not None:
            bx, by = float(ball_seed[0]), float(ball_seed[1])
            kept.append(Detection("ball", (bx - 8, by - 8, bx + 8, by + 8), 1.0))
        fd.detections = kept

    clip = BroadcastClip()
    tracker: HomographyTracker | None = None
    cutdet = CutDetector(threshold=0.5)
    for (i, rgb, gray), det in zip(frames, dets):
        if tracker is None:
            tracker = HomographyTracker(gray, calibration)
            kp = np.array([calibration.points[n] for n in names], dtype=float)
        else:
            tracker.update(gray)           # propagate H via optical flow
            kp = tracker.pts.reshape(-1, 2)

        cut = cutdet.update(cv2.resize(rgb, (64, 36)))
        if cut and tracker is not None:    # a cut breaks the calibration
            tracker = HomographyTracker(gray, calibration)

        clip.frames.append(BroadcastFrame(
            frame_idx=i, camera=cam, detections=det,
            kp_pixels=kp.copy(), kp_court=court_pts.copy(), kp_conf=np.ones(len(names)),
            clock_read=None, cut=cut, image=rgb,
        ))
    return clip
