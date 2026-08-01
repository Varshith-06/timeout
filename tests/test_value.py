"""Phase 2 value-stack tests: features, sub-models, V(s), transition, scoring."""
import numpy as np
import pytest

from src.value import features as F
from src.value import simulation as SIM
from src.value.actions import enumerate_actions
from src.value.features import PlayerVocab, entity_tensor
from src.value.scoring import score_action, score_actions
from src.value.state_value import train_value_model
from src.value.submodels import train_submodels
from src.value.transition import apply_action


@pytest.fixture(scope="module")
def small_dataset():
    return SIM.build_dataset(n_possessions=400, seed=3)


@pytest.fixture(scope="module")
def trained(small_dataset):
    ds = small_dataset
    submodels = train_submodels(ds)
    vocab = PlayerVocab(ds.player_ids)
    value_model, _ = train_value_model(ds.possessions, vocab, epochs=8, seed=1)
    return submodels, value_model


# --- Features ----------------------------------------------------------------
def test_entity_tensor_shapes(small_dataset):
    s = small_dataset.possessions[0].states[0]
    ents, pidx, mask, glob = entity_tensor(s)
    assert ents.shape == (F.MAX_ENTITIES, F.ENTITY_DIM)
    assert pidx.shape == (F.MAX_ENTITIES,)
    assert glob.shape == (F.GLOBAL_DIM,)
    # Ball entity present and flagged.
    assert mask.sum() >= 6
    assert ents[:, 6].max() == 1.0  # is_ball flag set somewhere


def test_feature_vector_lengths(small_dataset):
    s = small_dataset.possessions[0].states[0]
    h = s.handler
    assert len(F.shot_features(s, h)) == len(F.SHOT_FEATURES)
    r = next(p for p in s.offense() if p.player_id != h.player_id)
    assert len(F.pass_features(s, h, r)) == len(F.PASS_FEATURES)
    assert len(F.drive_features(s, h, "left")) == len(F.DRIVE_FEATURES)


def test_ground_truth_probs_monotone():
    # Farther, more-pressured shots are less likely to go in.
    close = SIM.true_make_prob(2, 0.2, 0, 0, 1)
    far = SIM.true_make_prob(25, 0.2, 1, 0, 0)
    assert close > far
    # More defenders in the lane lowers completion.
    assert SIM.true_complete_prob(15, 0, 5, 1) > SIM.true_complete_prob(15, 3, 5, 1)


# --- Sub-models --------------------------------------------------------------
def test_shot_model_beats_baseline_and_calibrates(small_dataset):
    ds = small_dataset
    sm = train_submodels(ds)
    te = SIM.build_dataset(n_possessions=300, seed=77)
    p = sm.shot.predict_proba(te.shot_X, te.shot_player)
    assert p.shape == (len(te.shot_y),)
    assert p.min() >= 0 and p.max() <= 1
    # Predictions correlate with the (held-out) ground-truth make probability.
    truth = np.array([SIM.true_make_prob(x[0], x[4], x[6], ds.player_skill.get(int(pid), 0), x[7])
                      for x, pid in zip(te.shot_X, te.shot_player)])
    assert np.corrcoef(p, truth)[0, 1] > 0.4


def test_eb_prior_shrinks_toward_league(small_dataset):
    sm = train_submodels(small_dataset)
    # An unseen player falls back to the league mean.
    assert sm.shot.prior.get(999999) == pytest.approx(sm.shot.prior.league)


# --- Transition --------------------------------------------------------------
def test_pass_transition_moves_ball_to_receiver(small_dataset):
    s = small_dataset.possessions[0].states[0]
    acts = enumerate_actions(s)
    pass_a = next(a for a in acts if a.action == "PASS_TO")
    s2 = apply_action(s, pass_a)
    assert s2.handler.player_id == pass_a.target
    # Ball is now at the receiver.
    recv = next(p for p in s.offense() if p.player_id == pass_a.target)
    assert abs(s2.ball["x"] - recv.x) < 1.0


def test_shoot_transition_is_terminal(small_dataset):
    s = small_dataset.possessions[0].states[0]
    shoot = next(a for a in enumerate_actions(s) if a.action == "SHOOT")
    assert apply_action(s, shoot) is None


def test_drive_moves_handler_toward_rim(small_dataset):
    s = small_dataset.possessions[0].states[0]
    drive = next(a for a in enumerate_actions(s) if a.action == "DRIVE")
    s2 = apply_action(s, drive)
    assert s2.handler.dist_to_rim < s.handler.dist_to_rim


# --- Scoring -----------------------------------------------------------------
def test_scores_are_expected_points(trained, small_dataset):
    submodels, value_model = trained
    s = small_dataset.possessions[0].states[0]
    acts = enumerate_actions(s)
    scored = score_actions(s, acts, submodels, value_model)
    assert len(scored) == len(acts)
    # Ranked descending, probabilities valid, values finite.
    qs = [sc.q for sc in scored]
    assert qs == sorted(qs, reverse=True)
    for sc in scored:
        assert 0.0 <= sc.success_prob <= 1.0
        assert np.isfinite(sc.q)


def test_shoot_q_is_prob_times_points(trained, small_dataset):
    submodels, value_model = trained
    s = small_dataset.possessions[0].states[0]
    shoot = next(a for a in enumerate_actions(s) if a.action == "SHOOT")
    sc = score_action(s, shoot, submodels, value_model)
    points = 3 if "three" in s.handler.zone else 2
    assert sc.q == pytest.approx(sc.success_prob * points, abs=1e-6)


def test_model_beats_random_baseline(trained, small_dataset):
    from src.value.evaluate import ordering_eval
    submodels, value_model = trained
    res = ordering_eval(small_dataset.possessions[:150], submodels, value_model,
                        small_dataset.player_skill)
    assert res["model"]["mean_regret"] < res["random"]["mean_regret"]
