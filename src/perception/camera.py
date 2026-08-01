"""Pinhole broadcast-camera model (roadmap 4.1 / 4.6 support).

A broadcast camera viewing the court plane (z=0) induces a 3x3 homography from
court coordinates (feet) to image pixels. This module builds that homography
from physical camera parameters (position, look-at, focal length) so the
synthetic broadcast is geometrically consistent — and so the homography solver
in :mod:`src.perception.homography` has a real, non-trivial transform to recover.

Pan and zoom are exposed as smooth per-frame parameters, matching how a real
broadcast camera moves (roadmap 4.6: H should change smoothly).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


def _normalize(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else v


@dataclass
class BroadcastCamera:
    img_w: int = 1280
    img_h: int = 720
    position: np.ndarray = field(default_factory=lambda: np.array([47.0, -28.0, 24.0]))
    target: np.ndarray = field(default_factory=lambda: np.array([47.0, 25.0, 0.0]))
    focal: float = 850.0   # wide enough that a half-court set usually fits in frame

    def _K(self) -> np.ndarray:
        return np.array([[self.focal, 0, self.img_w / 2],
                         [0, self.focal, self.img_h / 2],
                         [0, 0, 1.0]])

    def _R_world2cam(self) -> np.ndarray:
        up = np.array([0.0, 0.0, 1.0])
        zc = _normalize(self.target - self.position)   # forward
        xc = _normalize(np.cross(up, zc))               # right
        yc = np.cross(zc, xc)                           # down
        # Rows are camera axes expressed in world frame -> world->cam rotation.
        return np.stack([xc, yc, zc], axis=0)

    def homography_court_to_img(self) -> np.ndarray:
        """3x3 H mapping court (X, Y, 1) -> image (u, v, w), normalized by H[2,2]."""
        R = self._R_world2cam()
        t = -R @ self.position
        # For plane z=0 the third rotation column drops out.
        H = self._K() @ np.column_stack([R[:, 0], R[:, 1], t])
        return H / H[2, 2]

    def project(self, court_xy: np.ndarray) -> np.ndarray:
        """Project (N,2) court points to (N,2) pixel points."""
        pts = np.atleast_2d(court_xy).astype(float)
        H = self.homography_court_to_img()
        hom = np.column_stack([pts, np.ones(len(pts))]) @ H.T
        return hom[:, :2] / hom[:, 2:3]

    def project3d(self, points3d: np.ndarray) -> np.ndarray:
        """Full pinhole projection of (N,3) world points (feet) to (N,2) pixels.

        Used for off-plane objects — the ball in flight (its z is height) and the
        rim — which the court-plane homography cannot place.
        """
        pts = np.atleast_2d(points3d).astype(float)
        R = self._R_world2cam()
        cam = (pts - self.position) @ R.T          # world -> camera
        img = cam @ self._K().T
        return img[:, :2] / img[:, 2:3]

    def in_view(self, pixels: np.ndarray, margin: int = 0) -> np.ndarray:
        """Boolean mask of pixel points inside the image frame."""
        u, v = pixels[:, 0], pixels[:, 1]
        return (u >= -margin) & (u < self.img_w + margin) & (v >= -margin) & (v < self.img_h + margin)

    def panned(self, pan_deg: float = 0.0, zoom: float = 1.0) -> "BroadcastCamera":
        """Return a copy with the look-at yawed by pan_deg and focal scaled by zoom."""
        a = np.radians(pan_deg)
        # Rotate the target around the camera's vertical axis through court center.
        cx, cy = 47.0, 25.0
        tx, ty = self.target[0] - cx, self.target[1] - cy
        rx = cx + tx * np.cos(a) - ty * np.sin(a)
        ry = cy + tx * np.sin(a) + ty * np.cos(a)
        return BroadcastCamera(
            img_w=self.img_w, img_h=self.img_h,
            position=self.position.copy(),
            target=np.array([rx, ry, 0.0]),
            focal=self.focal * zoom,
        )
