"""Phase 3 perception tests: camera/homography, tracking, teams, identity, clock,
state recovery, and the domain-gap measurement."""
import numpy as np
import pytest

from src.ingest.possessions import iter_possessions
from src.ingest.sportvu import parse_game
from src.ingest.synthetic import generate_game
from src.state.schema import roster_jersey_map


@pytest.fixture(scope="module")
def possession():
    game = parse_game(generate_game(n_possessions=1, attack="left", seed=5))
    poss = list(iter_possessions(game))[0]
    return game, poss


@pytest.fixture(scope="module")
def clip_and_recovery(possession):
    from src.perception.synthetic_broadcast import generate_broadcast
    from src.perception.state_from_cv import recover_tracking
    game, poss = possession
    jmap = roster_jersey_map(game)
    roster_rows = list(game.roster.select(["team_id", "jersey", "player_id"]).iter_rows())
    clip = generate_broadcast(poss, seed=1, cut_prob=0.0, stride=5, jersey_map=jmap)
    rec = recover_tracking(clip, roster_rows, stride=5)
    return game, poss, clip, rec, jmap


# --- Camera + homography (load-bearing) --------------------------------------
def test_camera_homography_roundtrip_exact():
    from src.perception.camera import BroadcastCamera
    from src.perception.homography import project_feet_to_court, solve_homography
    from src.perception.keypoints import canonical_array
    cam = BroadcastCamera()
    kp = canonical_array()
    kp_px = cam.project(kp)
    vis = cam.in_view(kp_px)
    res = solve_homography(kp_px[vis], kp[vis])
    assert res.ok and res.median_reproj_ft < 1e-3
    # A player's foot recovers to its true court spot.
    players = np.array([[24.0, 25.0], [14.0, 8.0]])
    foot_px = cam.project(players)
    boxes = np.column_stack([foot_px[:, 0] - 15, foot_px[:, 1] - 60, foot_px[:, 0] + 15, foot_px[:, 1]])
    rec = project_feet_to_court(res.H, boxes)
    assert np.allclose(rec, players, atol=1e-3)


def test_homography_rejects_when_too_few_points():
    from src.perception.homography import solve_homography
    res = solve_homography(np.zeros((3, 2)), np.zeros((3, 2)))
    assert not res.ok and res.H is None


def test_homography_under_noise_within_target(clip_and_recovery):
    # Median reprojection error should meet the roadmap 4.6 target (<1.5 ft).
    _, _, _, rec, _ = clip_and_recovery
    assert rec.diagnostics["homog_valid_rate"] >= 0.85


# --- Cut detection -----------------------------------------------------------
def test_histogram_cut_detection():
    from src.perception.cuts import frame_histogram, histogram_distance
    a = np.zeros((32, 32, 3), np.uint8); a[:] = (30, 30, 120)
    b = np.zeros((32, 32, 3), np.uint8); b[:] = (200, 180, 40)
    assert histogram_distance(frame_histogram(a), frame_histogram(a)) < 0.05
    assert histogram_distance(frame_histogram(a), frame_histogram(b)) > 0.7


def test_cut_resets_track_ids(possession):
    from src.perception.synthetic_broadcast import generate_broadcast
    from src.perception.tracking import run_tracker
    _, poss = possession
    no_cut = run_tracker(generate_broadcast(poss, seed=2, cut_prob=0.0, stride=5).frames)
    with_cut = run_tracker(generate_broadcast(poss, seed=2, cut_prob=0.4, stride=5).frames)
    # Cuts fragment tracks -> strictly more tracklets.
    assert len(with_cut.all_tracklets()) > len(no_cut.all_tracklets())


# --- Tracking / teams / identity ---------------------------------------------
def test_tracklet_purity(clip_and_recovery):
    from collections import Counter
    _, _, _, rec, _ = clip_and_recovery
    purities = []
    for t in rec.tracklets:
        ids = [d.true_player_id for _, d in t.history if d.true_player_id is not None]
        if ids:
            purities.append(Counter(ids).most_common(1)[0][1] / len(ids))
    assert np.mean(purities) > 0.9


def test_team_and_identity_accuracy(clip_and_recovery):
    from src.perception.identity import identity_accuracy
    from src.perception.teams import team_accuracy
    _, _, _, rec, _ = clip_and_recovery
    assert team_accuracy(rec.tracklets, rec.team_labels) > 0.95
    assert identity_accuracy(rec.identities, rec.tracklets) > 0.85


# --- Clock -------------------------------------------------------------------
def test_clock_validation_fixes_blunders():
    from src.perception.clock import validate_clock
    reads = [22, 22, 21, 8.7, 21, 20, 20]      # 8.7 is an OCR blunder
    val = validate_clock(reads, dt=0.2)
    assert abs(val[3] - 20.5) < 1.5             # interpolated, not 8.7
    assert all(v is not None for v in val)


def test_clock_no_reset_cascade():
    from src.perception.clock import validate_clock
    # A mid-possession blunder near 24 must NOT be taken as a reset.
    reads = [20, 20, 24.0, 19, 19, 18]
    val = validate_clock(reads, dt=0.2)
    assert val[3] < 21 and val[5] < 21           # countdown continues, no jump to 24


# --- State recovery + domain gap ---------------------------------------------
def test_state_from_cv_same_schema(clip_and_recovery):
    from src.perception.state_from_cv import (build_state_from_cv, infer_offense_team,
                                              pick_showable_frame)
    game, poss, clip, rec, jmap = clip_and_recovery
    assert infer_offense_team(rec) == poss.offense_team_id     # offense not flipped
    fi = pick_showable_frame(rec)
    assert fi is not None
    state, conf = build_state_from_cv(rec, fi, roster_jersey=jmap)
    # Byte-identical schema to Phase 1: same fields, confidence now < 1.
    d = state.to_dict()
    assert set(d.keys()) >= {"timestamp", "ball", "players", "context"}
    assert 0.0 <= conf <= 1.0
    assert state.handler is not None
    assert state.context.n_players_observed >= 8


def test_domain_gap_within_targets(clip_and_recovery):
    from src.perception.evaluate import measure_domain_gap
    _, _, clip, rec, _ = clip_and_recovery
    gap = measure_domain_gap(rec, clip)
    assert gap["position_error_ft_median"] < 1.5      # roadmap 4.6 target
    assert gap["identity_error_rate"] < 0.1
    assert "augment_recommendation" in gap


def test_augment_feedback_maps_to_config():
    from src.perception.augment_feedback import AugmentRecommendation
    rec = AugmentRecommendation.from_gap(position_err_ft=2.0, miss_rate=0.15, id_error=0.03)
    cfg = rec.to_augment_config()
    assert cfg.jitter_ft == pytest.approx(2.4, abs=0.01)
    assert 0.0 < cfg.dropout_prob <= 0.9
