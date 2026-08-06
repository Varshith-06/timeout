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
    # RANSAC can return an H whose consensus set is empty (degenerate or wildly
    # inconsistent correspondences). That is a failed solve, not a usable H —
    # and projecting the empty point set would otherwise crash below.
    if not inliers.any():
        return HomographyResult(None, float("inf"), 0, False)
    proj = project_points(H, pixel_pts[inliers])
    err = np.linalg.norm(proj - court_pts[inliers], axis=1)  # feet
    median = float(np.median(err)) if len(err) else float("inf")
    ok = median <= max_reproj_ft and inliers.sum() >= 4
    return HomographyResult(H, median, int(inliers.sum()), ok)


def project_points(H: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Apply a 3x3 homography to (N,2) points using cv2.perspectiveTransform.

    An empty input is passed straight back: cv2 returns None rather than an empty
    array for a zero-length point set, which would otherwise surface as an
    AttributeError several frames later.
    """
    pts = np.asarray(pts, dtype=np.float64).reshape(-1, 1, 2)
    if len(pts) == 0:
        return np.empty((0, 2), dtype=np.float64)
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


def plausible_court_homography(H: np.ndarray, img_size, min_px_per_ft: float = 8.0,
                               max_px_per_ft: float = 150.0,
                               max_scale_ratio: float = 12.0,
                               min_visible: int = 6) -> bool:
    """Is this H geometrically believable as a broadcast view of a court?

    Reprojection error cannot answer that. It only asks whether a homography is
    consistent with the correspondences it was fitted to — so a keypoint model
    that hallucinates a self-consistent but wrong set of landmarks produces a
    *low* error and a confidently wrong court. This is the independent check: it
    ignores the correspondences entirely and asks whether the mapping looks like
    a camera pointed at a basketball court.

    The test is scale. On the measured click calibrations a foot of court spans
    17-56 px at broadcast zoom, tightly grouped; a hallucinated homography puts
    it at 1.5-5 px — it has effectively placed the camera in the next building.
    That is a 5-10x separation, which is why an absolute band works here.

    Scale is sampled **only where the court is actually visible**. Probing fixed
    court locations instead is what makes this fragile: on a zoomed shot the far
    baseline sits beyond the vanishing point, where scale legitimately runs to
    tens of thousands of px/ft, and a real calibration gets rejected. Convexity
    of the projected court fails for the same reason and is deliberately not
    tested.
    """
    if H is None or not np.isfinite(np.asarray(H)).all():
        return False
    try:
        H_court_to_px = np.linalg.inv(np.asarray(H, dtype=np.float64))
    except np.linalg.LinAlgError:
        return False

    w, h = float(img_size[0]), float(img_size[1])
    gx, gy = np.meshgrid(np.linspace(2.0, 92.0, 10), np.linspace(2.0, 48.0, 6))
    grid = np.column_stack([gx.ravel(), gy.ravel()])

    a = project_points(H_court_to_px, grid)
    b = project_points(H_court_to_px, grid + np.array([5.0, 0.0]))
    ok = np.isfinite(a).all(axis=1) & np.isfinite(b).all(axis=1)
    inside = ok & (a[:, 0] >= 0) & (a[:, 0] < w) & (a[:, 1] >= 0) & (a[:, 1] < h)
    if inside.sum() < min_visible:
        return False        # almost no court in frame: nothing to trust

    px_per_ft = np.linalg.norm(b[inside] - a[inside], axis=1) / 5.0
    px_per_ft = px_per_ft[np.isfinite(px_per_ft) & (px_per_ft > 0)]
    if len(px_per_ft) < min_visible:
        return False

    med = float(np.median(px_per_ft))
    if not (min_px_per_ft <= med <= max_px_per_ft):
        return False
    # Within the visible court, perspective foreshortening is bounded; a wild
    # spread means a near-singular fit rather than a real camera.
    lo, hi = np.percentile(px_per_ft, [5, 95])
    return bool(lo > 0 and hi / lo <= max_scale_ratio)


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
