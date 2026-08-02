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
                         pause_sec: float, window_sec: float, stride: int) -> BroadcastClip:
    """Build a BroadcastClip from the ``window_sec`` of video before ``pause_sec``."""
    import cv2
    W, H = video.width, video.height
    cam = BroadcastCamera(img_w=W, img_h=H)
    start = max(0, int((pause_sec - window_sec) * video.fps))
    end = int(pause_sec * video.fps)

    names = list(calibration.points.keys())
    court_pts = np.array([_LANDMARK_COURT[n] for n in names], dtype=float)

    clip = BroadcastClip()
    tracker: HomographyTracker | None = None
    cutdet = CutDetector(threshold=0.5)
    idx = 0
    for f in range(start, end + 1, stride):
        rgb = video.frame_at_index(f)
        if rgb is None:
            continue
        gray = video.gray(rgb)
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
            frame_idx=idx, camera=cam, detections=detector.detect(rgb, idx),
            kp_pixels=kp.copy(), kp_court=court_pts.copy(), kp_conf=np.ones(len(names)),
            clock_read=None, cut=cut,
        ))
        idx += 1
    return clip
