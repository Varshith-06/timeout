"""Manual court calibration — click known landmarks to solve the homography.

Broadcast footage has no trained court-keypoint model, so the pixel->court
homography is bootstrapped by clicking a handful of known court landmarks on one
frame. We already know every landmark's court coordinate (in feet), so the click
supplies only the pixel location. 6-8 well-spread line intersections give a
robust homography; :class:`HomographyTracker` then propagates it across a camera
shot with optical flow, so you calibrate once per continuous shot, not per frame.

The interactive picker needs a display; the solve/save/load/track logic is pure
and unit-tested.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from src.perception.homography import HomographyResult, solve_homography
from src.state import court

# Curated, easy-to-click landmarks (line intersections), left and right ends.
# (name, (court_x, court_y) feet, human description). Click the ones visible in
# your frame; skip the rest. The picker highlights each on a court diagram so
# left/right/near/far is never ambiguous.
CLICK_LANDMARKS = [
    ("lpaint_bl", (0.0, 17.0), "left baseline x left lane line"),
    ("lpaint_tl", (0.0, 33.0), "left baseline x right lane line"),
    ("lpaint_br", (19.0, 17.0), "free-throw line x left lane line (left end)"),
    ("lpaint_tr", (19.0, 33.0), "free-throw line x right lane line (left end)"),
    ("l3_break_bottom", (14.0, 3.0), "left corner-three break (bottom)"),
    ("l3_break_top", (14.0, 47.0), "left corner-three break (top)"),
    ("corner_bl", (0.0, 0.0), "left baseline x bottom sideline"),
    ("corner_tl", (0.0, 50.0), "left baseline x top sideline"),
    ("mid_bottom", (47.0, 0.0), "half-court line x bottom sideline"),
    ("mid_top", (47.0, 50.0), "half-court line x top sideline"),
    ("rpaint_br", (94.0, 17.0), "right baseline x right lane line"),
    ("rpaint_tr", (94.0, 33.0), "right baseline x left lane line"),
    ("rpaint_bl", (75.0, 17.0), "free-throw line x right lane line (right end)"),
    ("rpaint_tl", (75.0, 33.0), "free-throw line x left lane line (right end)"),
    ("r3_break_bottom", (80.0, 3.0), "right corner-three break (bottom)"),
    ("r3_break_top", (80.0, 47.0), "right corner-three break (top)"),
]
_LANDMARK_COURT = {n: xy for n, xy, _ in CLICK_LANDMARKS}


@dataclass
class Calibration:
    """A solved pixel->court homography plus the clicks that produced it."""
    H: np.ndarray                      # pixel -> court (feet)
    points: dict = field(default_factory=dict)   # landmark name -> [px, py]
    img_size: tuple = (0, 0)           # (width, height)
    reproj_error_ft: float = 0.0
    time: float | None = None          # video time (s) it was clicked; anchors its camera shot

    def save(self, path):
        Path(path).write_text(json.dumps({
            "H": self.H.tolist(), "points": {k: list(v) for k, v in self.points.items()},
            "img_size": list(self.img_size), "reproj_error_ft": self.reproj_error_ft,
            "time": self.time,
        }, indent=2), encoding="utf-8")

    @staticmethod
    def load(path):
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        return Calibration(np.array(d["H"]), {k: tuple(v) for k, v in d["points"].items()},
                           tuple(d["img_size"]), d.get("reproj_error_ft", 0.0), d.get("time"))


def solve_calibration(clicks: dict, img_size=(0, 0)) -> Calibration | None:
    """Solve pixel->court from {landmark_name: (px, py)} clicks. None if <4 or bad."""
    names = [n for n in clicks if n in _LANDMARK_COURT]
    if len(names) < 4:
        return None
    pixel = np.array([clicks[n] for n in names], dtype=float)
    court_pts = np.array([_LANDMARK_COURT[n] for n in names], dtype=float)
    res: HomographyResult = solve_homography(pixel, court_pts, max_reproj_ft=3.0)
    if res.H is None:
        return None
    return Calibration(res.H, {n: tuple(clicks[n]) for n in names}, tuple(img_size),
                       round(res.median_reproj_ft, 3))


# --- Optical-flow propagation across a camera shot ---------------------------
class HomographyTracker:
    """Propagates a calibration across frames by tracking the clicked points.

    Broadcast cameras pan and zoom smoothly, so the clicked landmarks move
    smoothly too. We track them with Lucas-Kanade optical flow each frame and
    re-solve the homography. When too few points survive (occlusion, a camera
    cut), :meth:`update` returns None — the signal to recalibrate.
    """

    def __init__(self, gray_frame: np.ndarray, calibration: Calibration):
        import cv2
        self.cv2 = cv2
        self.prev = gray_frame
        names = list(calibration.points.keys())
        self.pts = np.array([calibration.points[n] for n in names], dtype=np.float32).reshape(-1, 1, 2)
        self.court = np.array([_LANDMARK_COURT[n] for n in names], dtype=np.float64)
        self.H = calibration.H

    def update(self, gray_frame: np.ndarray):
        cv2 = self.cv2
        new_pts, status, _ = cv2.calcOpticalFlowPyrLK(self.prev, gray_frame, self.pts, None)
        if new_pts is None:
            return None
        good = status.ravel() == 1
        if good.sum() < 4:
            return None
        H, _ = cv2.findHomography(new_pts[good], self.court[good], cv2.RANSAC, 5.0)
        if H is not None:
            self.H = H
        self.prev = gray_frame
        self.pts = new_pts
        return self.H


# --- Interactive picker (needs a display) ------------------------------------
def _draw_reference(ax, highlight_xy):
    """Draw a top-down court on ax and highlight the current target landmark."""
    from src.perception.video_overlay import court_polylines
    for poly in court_polylines():
        ax.plot(poly[:, 0], poly[:, 1], color="#888", lw=1)
    for _, (cx, cy), _ in CLICK_LANDMARKS:
        ax.plot(cx, cy, "o", color="#ccc", ms=4)
    ax.plot(highlight_xy[0], highlight_xy[1], "o", color="red", ms=12)
    ax.set_aspect("equal"); ax.set_xlim(-4, 98); ax.set_ylim(-4, 54)
    ax.set_title("Click THIS point in the frame  (right-click to skip)")
    ax.axis("off")


def interactive_calibrate(frame_rgb: np.ndarray, out_path=None, time=None) -> Calibration | None:
    """Guided click-to-calibrate on one frame. Returns the solved Calibration.

    frame_rgb: (H, W, 3) image. For each landmark it highlights the target on a
    court diagram; left-click the matching spot in the frame, or right-click to
    skip a landmark that is off-screen. After 4+ clicks it solves, overlays the
    projected court lines for you to verify, and (if out_path) saves the result.
    """
    import matplotlib
    for backend in ("TkAgg", "QtAgg", "MacOSX"):     # ensure an INTERACTIVE backend
        try:
            matplotlib.use(backend, force=True); break
        except Exception:
            continue
    import matplotlib.pyplot as plt

    h, w = frame_rgb.shape[:2]
    clicks: dict = {}
    fig, (axf, axc) = plt.subplots(1, 2, figsize=(15, 6))
    axf.imshow(frame_rgb); axf.set_xlim(0, w); axf.set_ylim(h, 0); axf.axis("off")

    for name, court_xy, desc in CLICK_LANDMARKS:
        axc.clear(); _draw_reference(axc, court_xy)
        axf.set_title(f"Click: {desc}   [{len(clicks)} captured]  (right-click = skip)")
        fig.canvas.draw()
        pts = plt.ginput(1, timeout=-1, mouse_add=1, mouse_stop=3, mouse_pop=2)
        if pts:
            clicks[name] = (float(pts[0][0]), float(pts[0][1]))
            axf.plot(pts[0][0], pts[0][1], "g+", ms=12, mew=2)
            axf.annotate(name, pts[0], color="lime", fontsize=8)

    plt.close(fig)
    calib = solve_calibration(clicks, img_size=(w, h))
    if calib is None:
        print("Need at least 4 good clicks that solve — try again with more spread-out points.")
        return None

    calib.time = time                  # anchors this calibration to its camera shot
    _verify_plot(frame_rgb, calib)
    print(f"Calibration solved: {len(calib.points)} points, "
          f"median reprojection error {calib.reproj_error_ft:.2f} ft")
    if out_path:
        calib.save(out_path)
        print(f"Saved -> {out_path}")
    return calib


def _verify_plot(frame_rgb, calib: Calibration):
    """Overlay the projected court lines on the frame so the user can eyeball fit."""
    import cv2
    from src.perception.video_overlay import court_polylines  # this pins Agg...
    import matplotlib
    matplotlib.use("TkAgg", force=True)                        # ...switch back to interactive
    import matplotlib.pyplot as plt

    h, w = frame_rgb.shape[:2]
    H_c2p = np.linalg.inv(calib.H)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.imshow(frame_rgb); ax.set_xlim(0, w); ax.set_ylim(h, 0); ax.axis("off")
    for poly in court_polylines():
        px = cv2.perspectiveTransform(poly.reshape(-1, 1, 2).astype(np.float64), H_c2p).reshape(-1, 2)
        ax.plot(px[:, 0], px[:, 1], color="yellow", lw=1.5)
    for name, (x, y) in calib.points.items():
        ax.plot(x, y, "r+", ms=10)
    ax.set_title(f"Verify: yellow lines should sit on the real court "
                 f"(reproj {calib.reproj_error_ft:.2f} ft). Close to accept.")
    plt.show()
