"""Court geometry — the load-bearing module, so it gets the most tests."""
import math

import numpy as np

from src.state import court


def test_flip_left_is_identity_for_left_points():
    x, y = court.flip_to_left(10.0, 20.0)
    assert math.isclose(float(x), 84.0)  # 94 - 10
    assert math.isclose(float(y), 30.0)  # 50 - 20


def test_flip_is_involution():
    xs = np.array([10.0, 88.0, 47.0])
    ys = np.array([20.0, 40.0, 25.0])
    x1, y1 = court.flip_to_left(xs, ys)
    x2, y2 = court.flip_to_left(x1, y1)
    assert np.allclose(x2, xs)
    assert np.allclose(y2, ys)


def test_polar_at_rim_is_zero():
    d, _ = court.to_polar(*court.BASKET_LEFT)
    assert math.isclose(float(d), 0.0, abs_tol=1e-9)


def test_restricted_area_zone():
    # A point right at the rim is restricted area.
    assert court.assign_zone(6.0, 25.0) == "restricted_area"


def test_corner_three_zone():
    # Deep corner, past the y=3 line, near the baseline.
    z = court.assign_zone(6.0, 1.5)
    assert z == "corner_three_right"


def test_above_break_three_zone():
    # Top of the arc, well beyond 23.75 ft from the rim.
    z = court.assign_zone(32.0, 25.0)
    assert z == "above_break_three_center"


def test_mid_range_zone():
    # Outside the lane (y=10 < 17), inside the arc -> a mid-range shot.
    z = court.assign_zone(18.0, 10.0)
    assert z.startswith("mid_range")


def test_paint_non_ra_zone():
    # Inside the lane but outside the restricted area.
    assert court.assign_zone(18.0, 25.0) == "paint_non_ra"


def test_backcourt():
    assert court.assign_zone(60.0, 25.0) == "backcourt"


def test_three_point_geometry_corner_vs_arc():
    # Corner distance (22 ft laterally) is a three by the straight line...
    assert court._is_three(5.25, 3.0)
    # ...and a 15-ft mid-range shot is not.
    assert not court._is_three(19.0, 25.0)
