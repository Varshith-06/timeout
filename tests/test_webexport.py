"""Tests for the web-app overlay-spec geometry helpers."""
from src.app.webexport import _action_geometry, _label
from src.value.actions import Action
from src.value.scoring import ScoredAction


def _sc(action, kind, q=1.0, target=None, direction=None):
    a = Action(action, actor=1, id="a1", target=target, direction=direction)
    return ScoredAction(a, q=q, success_prob=0.5, v_next=q, kind=kind)


def test_label_variants():
    assert _label(_sc("PASS_TO", "pass", 1.23, target=7)).startswith("PASS (EPV 1.23")
    assert _label(_sc("DRIVE", "drive", 0.9, direction="left")) == "DRIVE left (EPV 0.90)"
    assert _label(_sc("SHOOT", "shoot", 1.1)) == "SHOOT (EPV 1.10)"


def test_pass_geometry_points_handler_to_target():
    feet = {7: [400.0, 300.0]}
    g = _action_geometry(_sc("PASS_TO", "pass", target=7), handler_px=[100.0, 200.0],
                         feet=feet, rim_px=[10.0, 20.0])
    assert g["circle"] == [100.0, 200.0]
    assert g["arrow"] == [[100.0, 200.0], [400.0, 300.0]]
    assert g["target"] == [400.0, 300.0]


def test_drive_geometry_points_handler_to_rim():
    g = _action_geometry(_sc("DRIVE", "drive", direction="middle"), handler_px=[100.0, 200.0],
                         feet={}, rim_px=[10.0, 20.0])
    assert g["arrow"] == [[100.0, 200.0], [10.0, 20.0]]


def test_reset_geometry_has_no_arrow():
    g = _action_geometry(_sc("RESET", "reset"), handler_px=[100.0, 200.0], feet={}, rim_px=[10.0, 20.0])
    assert g["arrow"] is None and g["circle"] == [100.0, 200.0]


def test_pass_geometry_missing_target_no_arrow():
    # Target not tracked this frame -> no arrow drawn (never invent a coordinate).
    g = _action_geometry(_sc("PASS_TO", "pass", target=99), handler_px=[1.0, 2.0],
                         feet={}, rim_px=[10.0, 20.0])
    assert "arrow" not in g or g["arrow"] is None