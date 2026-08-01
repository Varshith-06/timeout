"""NBA court geometry — the single source of truth for coordinates.

Roadmap 2.3: "Write this once, in src/state/court.py, and never hand-roll a
coordinate again."

Coordinate system (raw SportVU):
    x in [0, 94]  (length, baseline-to-baseline)
    y in [0, 50]  (width, sideline-to-sideline)
    origin at a baseline corner, units are feet.

After the half-court flip (:func:`flip_to_left`) the attacking basket is always
BASKET_LEFT, so every possession is expressed in one canonical frame.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

# --- Court constants, all in feet (roadmap 2.3) -----------------------------
COURT_LENGTH = 94.0
COURT_WIDTH = 50.0

BASKET_LEFT = (5.25, 25.0)      # rim center, 63" from the baseline
BASKET_RIGHT = (88.75, 25.0)
BACKBOARD_INSET = 4.0           # from baseline

PAINT_WIDTH = 16.0              # the lane spans y in [17, 33]
PAINT_Y = (17.0, 33.0)
FT_LINE_DIST = 19.0            # free-throw line, from baseline
FT_CIRCLE_R = 6.0

THREE_ARC_R = 23.75            # arc radius, measured from rim center
THREE_CORNER_Y = (3.0, 47.0)   # the straight corner segments
CORNER_BREAK_X = 14.0          # from baseline, where arc meets the corner line
RESTRICTED_R = 4.0             # restricted-area arc, from rim center

CENTER_CIRCLE = (47.0, 25.0)
CENTER_CIRCLE_R = 6.0

HALFCOURT_X = 47.0

# The attacking rim after a canonical flip. Everything zone/polar keys off this.
ATTACK_RIM = np.array(BASKET_LEFT, dtype=float)


# --- Half-court flip ---------------------------------------------------------
def basket_side(x: float) -> str:
    """Which half a point sits in. 'left' if nearer BASKET_LEFT."""
    return "left" if x < HALFCOURT_X else "right"


def flip_to_left(x, y):
    """Mirror coordinates so the attacking basket is always BASKET_LEFT.

    Points already on the left are returned unchanged. Points on the right are
    reflected through the court center: (x, y) -> (94 - x, 50 - y). Accepts
    scalars or numpy arrays. Roadmap 2.3(1): doubles effective sample size.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    xf = COURT_LENGTH - x
    yf = COURT_WIDTH - y
    return xf, yf


def flip_velocity(vx, vy):
    """A 180deg point reflection negates both velocity components."""
    return -np.asarray(vx, dtype=float), -np.asarray(vy, dtype=float)


# --- Basket-relative polar coordinates (roadmap 2.3(2)) ----------------------
def to_polar(x, y, rim=ATTACK_RIM):
    """Distance to rim (ft) and angle from the baseline (degrees, 0-180).

    Angle is measured at the rim: 0deg points along the baseline toward the
    nearer sideline of increasing y is arbitrary, so we define it as the angle
    of the (player - rim) vector measured from the positive-x axis (pointing
    away from the baseline into the court), in [-180, 180].
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    dx = x - rim[0]
    dy = y - rim[1]
    dist = np.hypot(dx, dy)
    angle = np.degrees(np.arctan2(dy, dx))
    return dist, angle


# --- Zone assignment (roadmap 2.3(3)) ---------------------------------------
# Standard shot zones, keyed off the attacking rim at BASKET_LEFT.
ZONES = (
    "restricted_area",
    "paint_non_ra",
    "mid_range_left",
    "mid_range_center",
    "mid_range_right",
    "corner_three_left",
    "corner_three_right",
    "above_break_three_left",
    "above_break_three_center",
    "above_break_three_right",
    "backcourt",
)


def _is_three(x: float, y: float, rim=ATTACK_RIM) -> bool:
    """True if a shot from (x, y) is worth three, using real NBA geometry.

    Corner threes: beyond the straight lines at y=3 / y=47, but only in the
    corner region (x < CORNER_BREAK_X). Everywhere else it is the 23.75 ft arc.
    """
    dist = math.hypot(x - rim[0], y - rim[1])
    in_corner_band = y <= THREE_CORNER_Y[0] or y >= THREE_CORNER_Y[1]
    if in_corner_band and x <= CORNER_BREAK_X:
        # Corner three: the line is the y=3 / y=47 straight segment.
        return True
    return dist >= THREE_ARC_R


def assign_zone(x: float, y: float, rim=ATTACK_RIM) -> str:
    """Bucket a court location into a standard shot zone.

    Left/right are from the offense's attacking perspective (looking at the
    rim from center court): larger y is the left side.
    """
    if x > HALFCOURT_X:
        return "backcourt"

    dist = math.hypot(x - rim[0], y - rim[1])
    in_paint = PAINT_Y[0] <= y <= PAINT_Y[1] and x <= FT_LINE_DIST

    # Left/center/right split by angle off the rim-to-baseline axis.
    # dy > 0 -> left side, dy < 0 -> right side; a central wedge is "center".
    dy = y - rim[1]
    dx = x - rim[0]
    angle = math.degrees(math.atan2(dy, dx))  # 0 = straight out from baseline
    if angle > 30:
        side = "left"
    elif angle < -30:
        side = "right"
    else:
        side = "center"

    if _is_three(x, y, rim):
        # Corner threes are the two flat bands; everything else is above-break.
        if (y <= THREE_CORNER_Y[0] or y >= THREE_CORNER_Y[1]) and x <= CORNER_BREAK_X:
            return "corner_three_left" if dy > 0 else "corner_three_right"
        return f"above_break_three_{side}"

    if dist <= RESTRICTED_R:
        return "restricted_area"
    if in_paint:
        return "paint_non_ra"
    return f"mid_range_{side}"


# --- Point-in-region helpers -------------------------------------------------
def in_paint(x: float, y: float) -> bool:
    return PAINT_Y[0] <= y <= PAINT_Y[1] and x <= FT_LINE_DIST


def crossed_halfcourt(x: float) -> bool:
    return x <= HALFCOURT_X


@dataclass(frozen=True)
class CourtDims:
    """Convenience bundle for the renderer."""

    length: float = COURT_LENGTH
    width: float = COURT_WIDTH
    rim_left: tuple = BASKET_LEFT
    rim_right: tuple = BASKET_RIGHT
