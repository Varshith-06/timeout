"""End-to-end Phase 1 pipeline on synthetic data: parse -> segment -> state -> actions."""
import pytest

from src.ingest.possessions import frames_to_dataframe, iter_possessions
from src.ingest.sportvu import parse_game
from src.ingest.synthetic import generate_game
from src.state.schema import build_states, convex_hull_area, defender_pressure, roster_jersey_map
from src.value.actions import enumerate_actions


@pytest.fixture(scope="module")
def game():
    return parse_game(generate_game(n_possessions=2, attack="mixed", seed=7))


def test_parse_shapes(game):
    assert game.roster.height == 10
    # 2 possessions x 125 frames x 11 entities.
    assert game.moments.height == 2 * 125 * 11
    # Exactly 11 entities per moment.
    counts = game.moments.group_by(["quarter", "game_clock"]).len()["len"].unique().to_list()
    assert counts == [11]


def test_segmentation_and_handler(game):
    poss = list(iter_possessions(game))
    assert len(poss) == 2
    for p in poss:
        assert len(p) == 125
        # Canonical: attacking basket is always the left rim.
        assert p.attacking_basket == (5.25, 25.0)
        handlers = [h for h in {f.handler_player_id for f in p.frames} if h is not None]
        # PG starts, wing finishes -> at least two distinct handlers.
        assert len(handlers) >= 2
        # A shot happens (ball rises above 10 ft).
        assert p.terminal_event == "shot_attempt"


def test_handler_smoothing_no_short_runs(game):
    """No handler run shorter than the minimum hold survives smoothing."""
    from src.ingest.possessions import HANDLER_MIN_HOLD

    for p in iter_possessions(game):
        seq = [f.handler_player_id for f in p.frames]
        i = 0
        while i < len(seq):
            j = i
            while j < len(seq) and seq[j] == seq[i]:
                j += 1
            if seq[i] is not None:
                assert (j - i) >= HANDLER_MIN_HOLD
            i = j


def test_state_schema_fields(game):
    poss = list(iter_possessions(game))[0]
    states = build_states(poss, roster_jersey_map(game))
    s = states[20]
    assert s.context.confidence == 1.0
    assert s.context.n_players_observed == 10
    assert s.context.spacing_area_sqft > 0
    assert s.context.defense_scheme in {"man", "zone", "unknown"}
    h = s.handler
    assert h is not None and h.has_ball
    assert h.seconds_since_touch == 0.0
    assert h.jersey is not None
    # Orientation in Phase 1 is the velocity heading, explicitly marked.
    assert h.orientation_source.startswith("velocity_heading")


def test_action_enumeration_bounds(game):
    poss = list(iter_possessions(game))[0]
    states = build_states(poss, roster_jersey_map(game))
    # A frame with a settled handler yields a legal, bounded action set.
    acts = enumerate_actions(states[20])
    assert 1 <= len(acts) <= 13
    kinds = {a.action for a in acts}
    assert "SHOOT" in kinds and "RESET" in kinds
    assert sum(a.action == "PASS_TO" for a in acts) == 4
    # Ball-in-flight frames offer no ball-handler decision.
    flight = next(f for f in poss.frames if f.ball_in_flight)
    fstate = build_states(poss, roster_jersey_map(game))[flight.frame_idx]
    assert enumerate_actions(fstate) == []


def test_shoot_pruned_when_far_and_clock_high():
    from src.value.actions import _shoot_legal

    class H:
        dist_to_rim = 40.0

    assert not _shoot_legal(H(), shot_clock=15.0)
    assert _shoot_legal(H(), shot_clock=1.0)  # desperation heave is legal


def test_convex_hull_area_square():
    import numpy as np

    pts = np.array([[0, 0], [0, 10], [10, 10], [10, 0], [5, 5]])
    assert convex_hull_area(pts) == pytest.approx(100.0)


def test_defender_pressure_monotonic():
    # A closer defender exerts more pressure.
    near = defender_pressure((20, 25), (22, 25))
    far = defender_pressure((20, 25), (30, 25))
    assert near > far


def test_frame_dataframe_deliverable(game):
    poss = list(iter_possessions(game))
    df = frames_to_dataframe(poss)
    for col in ["possession_id", "frame_idx", "handler_player_id", "ball_in_flight",
                "ball_x", "ball_y", "ball_z", "players", "terminal_event"]:
        assert col in df.columns
    assert df.height == 2 * 125
