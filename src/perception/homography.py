"""Court homography — the load-bearing perception component (roadmap 4.6).

Given detected court-keypoint pixels and their known canonical court coordinates,
solve a homography that maps pixels -> court feet, then project each player's
FOOT position (bottom-center of the box, not the box center — the homography maps
the court plane) into court space.

Not optional, per 4.6:
  1. Reprojection-error check in feet; reject frames above a threshold so a bad
     homography fails loudly instead of silently poisoning the state.
  2. Temporal smoothing: smooth the projected image positions of the four court
     corners over time, then re-solve — smoothing the raw 3x3 elements produces
     geometric artifacts.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import cv2
import numpy as np

# Four court corners in court feet — the stable reference for temporal smoothing.
_CORNERS_COURT = np.array([[0, 0], [0, 50], [94, 0], [94, 50]], dtype=float)


@dataclass
class HomographyResult:
    H: np.ndarray | None            # pixel -> court (feet), or None if unsolved
    median_reproj_ft: float         # median reprojection error, feet
    n_points: int                   # inliers used
    ok: bool                        # passed the reprojection gate


def solve_homography(pixel_pts, court_pts, conf=None, conf_thresh=0.5,
                     ransac_reproj_px=8.0, max_reproj_ft=2.0) -> HomographyResult:
    """Solve pixel->court from keypoint correspondences (roadmap 4.6).

    pixel_pts, court_pts: (N,2). conf: optional (N,) confidences; points below
    conf_thresh are dropped. Returns a :class:`HomographyResult`; ``ok`` is False
    (and H may still be present) when the median reprojection error exceeds
    max_reproj_ft or fewer than 4 points survive.
    """
    pixel_pts = np.asarray(pixel_pts, dtype=np.float64)
    court_pts = np.asarray(court_pts, dtype=np.float64)
    if conf is not None:
        keep = np.asarray(conf) >= conf_thresh
        pixel_pts, court_pts = pixel_pts[keep], court_pts[keep]

    if len(pixel_pts) < 4:
        return HomographyResult(None, float("inf"), len(pixel_pts), False)

    H, mask = cv2.findHomography(pixel_pts, court_pts, cv2.RANSAC, ransac_reproj_px)
    if H is None:
        return HomographyResult(None, float("inf"), 0, False)

    inliers = mask.ravel().astype(bool)
    proj = project_points(H, pixel_pts[inliers])
    err = np.linalg.norm(proj - court_pts[inliers], axis=1)  # feet
    median = float(np.median(err)) if len(err) else float("inf")
    ok = median <= max_reproj_ft and inliers.sum() >= 4
    return HomographyResult(H, median, int(inliers.sum()), ok)


def project_points(H: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Apply a 3x3 homography to (N,2) points using cv2.perspectiveTransform."""
    pts = np.asarray(pts, dtype=np.float64).reshape(-1, 1, 2)
    out = cv2.perspectiveTransform(pts, H)
    return out.reshape(-1, 2)


def project_feet_to_court(H: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    """Map each detection box's foot point (bottom-center) into court feet.

    boxes: (N,4) as [x1,y1,x2,y2]. Roadmap 4.6: use the foot, not the box center
    — a torso is five feet above the court plane and projects badly.
    """
    boxes = np.asarray(boxes, dtype=float)
    feet = np.column_stack([(boxes[:, 0] + boxes[:, 2]) / 2, boxes[:, 3]])
    return project_points(H, feet)


class TemporalHomography:
    """Smooths H over time by averaging the projected court corners (roadmap 4.6).

    Broadcast cameras pan and zoom smoothly, so the corners' image positions move
    smoothly. We keep a short window of each frame's corner-in-pixel estimate,
    average it, and re-solve pixel->court from the smoothed corners.
    """

    def __init__(self, window: int = 5):
        self.window = window
        self._corner_px = deque(maxlen=window)  # each: (4,2) pixel corners
        self.last_H: np.ndarray | None = None

    def update(self, result: HomographyResult) -> HomographyResult:
        if result.H is None or not result.ok:
            # Gap: reuse the last good smoothed homography if we have one.
            if self.last_H is not None:
                return HomographyResult(self.last_H, result.median_reproj_ft,
                                        result.n_points, False)
            return result

        # Where do the four canonical corners land in pixels under this H?
        H_court_to_px = np.linalg.inv(result.H)
        corners_px = project_points(H_court_to_px, _CORNERS_COURT)
        self._corner_px.append(corners_px)

        smoothed_px = np.mean(np.stack(self._corner_px), axis=0)
        H_smoothed, _ = cv2.findHomography(smoothed_px, _CORNERS_COURT, 0)
        if H_smoothed is None:
            return result
        self.last_H = H_smoothed
        return HomographyResult(H_smoothed, result.median_reproj_ft, result.n_points, True)
