"""Tests for manual court calibration (solve, save/load, optical-flow tracker)."""
import numpy as np
import pytest

from src.perception.calibrate import (CLICK_LANDMARKS, Calibration, HomographyTracker,
                                      solve_calibration)
from src.perception.camera import BroadcastCamera
from src.perception.homography import project_feet_to_court


def _synthetic_clicks(cam):
    clicks = {}
    for name, (cx, cy), _ in CLICK_LANDMARKS:
        px = cam.project(np.array([[cx, cy]]))[0]
        if cam.in_view(px[None])[0]:
            clicks[name] = (float(px[0]), float(px[1]))
    return clicks


def test_solve_recovers_homography():
    cam = BroadcastCamera(focal=850)
    calib = solve_calibration(_synthetic_clicks(cam), img_size=(1280, 720))
    assert calib is not None
    assert calib.reproj_error_ft < 1e-3
    # A player's foot pixel maps back to the true court spot.
    truth = np.array([[24.0, 25.0], [14.0, 8.0], [8.0, 40.0]])
    foot = cam.project(truth)
    boxes = np.column_stack([foot[:, 0] - 15, foot[:, 1] - 60, foot[:, 0] + 15, foot[:, 1]])
    rec = project_feet_to_court(calib.H, boxes)
    assert np.allclose(rec, truth, atol=1e-2)


def test_solve_needs_four_points():
    assert solve_calibration({"corner_bl": (10, 10), "corner_tl": (10, 20)}) is None


def test_save_load_roundtrip(tmp_path):
    cam = BroadcastCamera(focal=850)
    calib = solve_calibration(_synthetic_clicks(cam), img_size=(1280, 720))
    p = tmp_path / "cal.json"
    calib.save(p)
    loaded = Calibration.load(p)
    assert np.allclose(calib.H, loaded.H)
    assert loaded.points.keys() == calib.points.keys()
    assert loaded.img_size == (1280, 720)


def test_time_persists_for_multishot(tmp_path):
    # The click-time anchors a calibration to its camera shot in a multi-shot build.
    cam = BroadcastCamera(focal=850)
    calib = solve_calibration(_synthetic_clicks(cam), img_size=(1280, 720))
    assert calib.time is None
    calib.time = 95.0
    p = tmp_path / "shot2.json"; calib.save(p)
    assert Calibration.load(p).time == 95.0


def test_tracker_stable_on_static_frame():
    cv2 = pytest.importorskip("cv2")
    cam = BroadcastCamera(focal=850)  # default 1280x720 pixel space
    calib = solve_calibration(_synthetic_clicks(cam), img_size=(1280, 720))
    # A textured frame (matching the camera's pixel space) so optical flow locks on.
    rng = np.random.default_rng(0)
    gray = rng.integers(0, 255, (720, 1280), dtype=np.uint8)
    tracker = HomographyTracker(gray, calib)
    H2 = tracker.update(gray)                 # identical frame -> no motion
    assert H2 is not None
    # Homography should be essentially unchanged frame-to-frame.
    assert np.allclose(H2 / H2[2, 2], calib.H / calib.H[2, 2], atol=1e-4)
