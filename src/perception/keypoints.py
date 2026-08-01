"""Canonical court keypoints for homography (roadmap 4.6).

A symmetric set spanning the full 94x50 court so a single model covers both
halves. Coordinates are in feet, in the same frame as :mod:`src.state.court`.
Not all are visible in any given broadcast frame — that is expected and handled
by the homography solver, which uses whatever high-confidence points it has.
"""
from __future__ import annotations

import numpy as np

from src.state import court

# name -> (x, y) in feet. ~24 points across the groups in the roadmap 4.6 table.
CANONICAL_KEYPOINTS: dict[str, tuple[float, float]] = {
    # Court corners (4)
    "corner_bl": (0.0, 0.0),
    "corner_tl": (0.0, 50.0),
    "corner_br": (94.0, 0.0),
    "corner_tr": (94.0, 50.0),
    # Half-court sideline intersections + center circle top/bottom (4)
    "mid_bottom": (47.0, 0.0),
    "mid_top": (47.0, 50.0),
    "center_top": (47.0, 31.0),
    "center_bottom": (47.0, 19.0),
    # Left paint corners (4)
    "lpaint_bl": (0.0, 17.0),
    "lpaint_tl": (0.0, 33.0),
    "lpaint_br": (19.0, 17.0),
    "lpaint_tr": (19.0, 33.0),
    # Right paint corners (4)
    "rpaint_br": (94.0, 17.0),
    "rpaint_tr": (94.0, 33.0),
    "rpaint_bl": (75.0, 17.0),
    "rpaint_tl": (75.0, 33.0),
    # Free-throw circle tops/bottoms (2 each end -> 4)
    "lft_top": (19.0, 31.0),
    "lft_bottom": (19.0, 19.0),
    "rft_top": (75.0, 31.0),
    "rft_bottom": (75.0, 19.0),
    # Three-point corner breaks (2 each end -> 4)
    "l3_break_bottom": (14.0, 3.0),
    "l3_break_top": (14.0, 47.0),
    "r3_break_bottom": (80.0, 3.0),
    "r3_break_top": (80.0, 47.0),
    # Three-point arc apex (1 each end -> 2)
    "l3_apex": (court.BASKET_LEFT[0] + court.THREE_ARC_R, 25.0),
    "r3_apex": (court.BASKET_RIGHT[0] - court.THREE_ARC_R, 25.0),
}

KEYPOINT_NAMES = list(CANONICAL_KEYPOINTS.keys())
KEYPOINT_XY = np.array([CANONICAL_KEYPOINTS[n] for n in KEYPOINT_NAMES], dtype=float)


def canonical_array() -> np.ndarray:
    """(N, 2) array of canonical keypoint court coordinates, feet."""
    return KEYPOINT_XY.copy()
