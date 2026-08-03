"""Tests for the real-video robustness helpers: class mapping, roster gate,
ball interpolation, and handler smoothing."""
from src.perception.video import _class_map, _match_class, _roboflow_predictions
from src.perception.state_from_cv import (_interpolate_track, _select_roster,
                                          _smooth_possessor)
from src.state.court import COURT_WIDTH


def test_class_map_coco():
    # person -> player, sports ball -> ball; baseball bat/glove and refrigerator dropped.
    m = _class_map({0: "person", 32: "sports ball", 34: "baseball bat",
                    35: "baseball glove", 72: "refrigerator"})
    assert m == {0: "player", 32: "ball"}


def test_match_class_roboflow_names():
    # The Roboflow workflow's class names map onto our schema.
    assert _match_class("person") == "player"
    assert _match_class("ball") == "ball"
    assert _match_class("basket") == "rim"
    assert _match_class("scoreboard") is None


def test_roboflow_predictions_extraction():
    resp = {"outputs": [{"predictions": {"image": {"width": 1280, "height": 720},
            "predictions": [{"x": 665.0, "y": 343.0, "width": 24.0, "height": 29.0,
                             "confidence": 0.87, "class": "ball"}]}}]}
    preds = _roboflow_predictions(resp)
    assert len(preds) == 1 and preds[0]["class"] == "ball"
    assert _roboflow_predictions({"outputs": []}) == []


def test_class_map_basketball_model():
    m = _class_map({0: "Player", 1: "Ball", 2: "Hoop", 3: "Referee"})
    assert m == {0: "player", 1: "ball", 2: "rim", 3: "referee"}


def test_interpolate_track_fills_interior_and_holds_ends():
    seen = [None, (0.0, 0.0), None, None, (30.0, 60.0), None]
    out = _interpolate_track(seen)
    assert out[0] == (0.0, 0.0)                 # leading gap holds first
    assert out[2] == (10.0, 20.0)               # 1/3 of the way
    assert out[3] == (20.0, 40.0)               # 2/3 of the way
    assert out[5] == (30.0, 60.0)               # trailing gap holds last


def test_interpolate_track_all_none():
    assert _interpolate_track([None, None]) == [None, None]


def test_select_roster_keeps_interior_drops_sideline():
    mid = COURT_WIDTH / 2
    frames = []
    for _ in range(8):
        frames.append({
            1: (25.0, mid),              # interior player -> kept
            2: (20.0, COURT_WIDTH - 0.3),  # hugging the far sideline -> crowd, dropped
            3: (15.0, 0.2),              # hugging the near sideline -> crowd, dropped
        })
    roster = _select_roster(frames)
    assert roster == {1}


def test_select_roster_requires_presence():
    # A one-off detection (present in 1 of 10 frames) is not a roster player.
    frames = [{1: (25.0, 25.0)} if i == 0 else {} for i in range(10)]
    assert _select_roster(frames) == set()


def test_smooth_possessor_outvotes_single_flicker():
    poss = [7, 7, 3, 7, 7]        # a lone 3 among 7s
    assert _smooth_possessor(poss, window=2) == [7, 7, 7, 7, 7]


def test_smooth_possessor_handles_none():
    assert _smooth_possessor([None, None]) == [None, None]
