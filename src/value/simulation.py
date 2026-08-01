"""Ground-truth possession simulator for Phase 2 training/validation.

Real Phase 2 trains on 60k-100k SportVU shots and every possession frame. With
no downloads we instead simulate possessions from a *known* generative model and
learn against it. This is not a toy: it lets us prove the whole value stack is
wired correctly and, crucially, that the sub-models come out **calibrated**
(roadmap 3.4) — because we can compare learned probabilities against the true
ones. Swap in real logs and the same feature/label contract trains for real.

The ground-truth functions below are the ONLY place the true probabilities live.
The learned models in :mod:`src.value.submodels` and :mod:`src.value.state_value`
never see them — they see only features and sampled 0/1 outcomes, exactly as they
would from real tracking.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from src.ingest.possessions import Frame, PlayerFrame
from src.ingest.synthetic import _shade_toward
from src.state import court
from src.state.schema import State, build_state
from src.value import features as F
from src.value.actions import enumerate_actions

OFF_TEAM = 1
DEF_TEAM = 2
OFF_IDS = [100, 101, 102, 103, 104]
DEF_IDS = [200, 201, 202, 203, 204]

# Canonical per-player shooting skill (log-odds bump), FIXED across every dataset.
# A given player has one true skill in the world, so train and eval must agree on
# it — otherwise the shot model's empirical-Bayes per-player prior is learned on
# one skill and scored against another, which silently wrecks recommendations.
PLAYER_SKILLS = {
    pid: float(s)
    for pid, s in zip(OFF_IDS, np.random.default_rng(12345).normal(0, 0.35, len(OFF_IDS)))
}

MAX_STEPS = 6
SHOTCLOCK_START = 20.0
SHOTCLOCK_PER_STEP = 3.0


def _sigmoid(z: float) -> float:
    return 1.0 / (1.0 + math.exp(-z))


def _phase2_set(rim, rng):
    """A realistically spaced half-court set (canonical, attacking left rim).

    Offsets are [into-court, sideward] in feet; larger y is the offense's left.
    Spans real shot zones — above-break three, wings, a corner three, a big
    inside — so the sub-models see rim/mid/three variety (roadmap 3.4).
    """
    rx, ry = rim
    spots = np.array(
        [
            [rx + 24.0, ry],          # 0 PG: top of the key (above-break three)
            [rx + 17.0, ry - 14.0],   # 1 right wing (three)
            [rx + 17.0, ry + 14.0],   # 2 left wing (three)
            [rx + 8.0, ry - 21.5],    # 3 right corner (corner three)
            [rx + 7.0, ry + 5.0],     # 4 big: short of the paint
        ],
        dtype=float,
    )
    return spots + rng.normal(0, 0.6, spots.shape)


# --- Ground-truth probability model (never seen by learners) -----------------
def true_make_prob(dist, pressure, is_three, skill, catch_and_shoot) -> float:
    z = 1.15 - 0.05 * dist - 1.1 * pressure + skill + 0.2 * catch_and_shoot
    return _sigmoid(z)


def true_complete_prob(pass_dist, defs_in_lane, receiver_sep, facing) -> float:
    z = 3.6 - 0.05 * pass_dist - 1.3 * defs_in_lane + 0.15 * receiver_sep + 0.4 * facing
    return _sigmoid(z)


def true_drive_prob(help_defenders, primary_lateral, speed, dist_to_rim) -> float:
    z = 1.7 - 0.7 * help_defenders + 0.15 * primary_lateral + 0.1 * speed - 0.02 * dist_to_rim
    return _sigmoid(z)


def _cushion(ball_owner, rng):
    """Per-defender rim-side cushion (ft): tight on the ball, looser off-ball.

    Off-ball sag is what creates open shooters — so a pass that finds an open
    man raises expected value, which is the EPV structure V(s) must learn."""
    feet = rng.uniform(3.5, 7.5, 5)
    if ball_owner is not None:
        feet[ball_owner] = rng.uniform(2.0, 3.0)
    return feet


# --- Samples -----------------------------------------------------------------
@dataclass
class PossessionSample:
    states: list[State]
    step_index: list[int]
    realized_points: float


@dataclass
class LabeledDataset:
    shot_X: np.ndarray
    shot_y: np.ndarray            # made 0/1
    shot_points: np.ndarray       # 2 or 3
    shot_player: np.ndarray       # player_id per shot
    pass_X: np.ndarray
    pass_y: np.ndarray
    drive_X: np.ndarray
    drive_y: np.ndarray
    possessions: list[PossessionSample] = field(default_factory=list)
    player_skill: dict = field(default_factory=dict)
    player_ids: list = field(default_factory=list)


# --- State construction ------------------------------------------------------
def _make_state(off_pos, def_pos, off_vel, def_vel, ball_owner, ball_xy, ball_z,
                shot_clock, poss_id, frame_idx) -> State:
    players = []
    for i, pid in enumerate(OFF_IDS):
        players.append(PlayerFrame(pid, OFF_TEAM, "offense",
                                   float(off_pos[i, 0]), float(off_pos[i, 1]),
                                   float(off_vel[i, 0]), float(off_vel[i, 1])))
    for i, pid in enumerate(DEF_IDS):
        players.append(PlayerFrame(pid, DEF_TEAM, "defense",
                                   float(def_pos[i, 0]), float(def_pos[i, 1]),
                                   float(def_vel[i, 0]), float(def_vel[i, 1])))
    handler_id = OFF_IDS[ball_owner] if ball_owner is not None else None
    frame = Frame(
        possession_id=poss_id, frame_idx=frame_idx, quarter=1,
        game_clock=600.0 - frame_idx, shot_clock=shot_clock,
        ball_x=float(ball_xy[0]), ball_y=float(ball_xy[1]), ball_z=float(ball_z),
        ball_vx=0.0, ball_vy=0.0,
        handler_player_id=handler_id, ball_in_flight=handler_id is None,
        players=players, offense_team_id=OFF_TEAM,
    )
    return build_state(frame, roster_jersey=None, last_touch_time=None)


def simulate_possession(rng: np.random.Generator, skills: dict, poss_id: str):
    """Simulate one possession; return (PossessionSample, shot/pass/drive events)."""
    rim = np.array(court.BASKET_LEFT)
    off_pos = _phase2_set(rim, rng)
    def_pos = _shade_toward(off_pos, rim, feet=_cushion(0, rng)) + rng.normal(0, 0.4, (5, 2))

    ball_owner = 0  # PG
    shot_clock = SHOTCLOCK_START
    states, step_idx = [], []
    shots, passes, drives = [], [], []
    realized = 0.0

    for step in range(MAX_STEPS):
        off_vel = rng.normal(0, 0.8, (5, 2))
        def_vel = rng.normal(0, 0.8, (5, 2))
        ball_xy = off_pos[ball_owner]
        state = _make_state(off_pos, def_pos, off_vel, def_vel, ball_owner,
                            ball_xy, 3.5, shot_clock, poss_id, step)
        states.append(state)
        step_idx.append(step)

        handler = state.handler
        actions = enumerate_actions(state)
        force_shot = shot_clock <= SHOTCLOCK_PER_STEP or step == MAX_STEPS - 1

        # Score each action by its TRUE expected points to build a semi-rational policy.
        scored = []
        for a in actions:
            scored.append((a, _true_action_value(state, a, skills)))
        if force_shot:
            scored = [(a, v) for a, v in scored if a.action == "SHOOT"] or scored

        actions_only = [a for a, _ in scored]
        qs = np.array([v for _, v in scored])
        probs = _softmax(qs, temp=0.45)
        choice = actions_only[rng.choice(len(actions_only), p=probs)]

        # Resolve the chosen action.
        if choice.action == "SHOOT":
            # Neutral league prior here; the ShotModel replaces this column with an
            # empirical-Bayes estimate learned from observed outcomes (no truth leak).
            feats = F.shot_features(state, handler, player_prior=0.45)
            p = true_make_prob(handler.dist_to_rim, handler.defender_pressure,
                               F._is_three(handler), skills[handler.player_id],
                               F._catch_and_shoot(handler))
            made = int(rng.random() < p)
            pts = 3 if F._is_three(handler) else 2
            shots.append((feats, made, pts, handler.player_id))
            realized = float(pts if made else 0)
            break

        if choice.action == "PASS_TO":
            receiver = next(pp for pp in state.offense() if pp.player_id == choice.target)
            feats = F.pass_features(state, handler, receiver)
            p = true_complete_prob(feats[0], feats[1], feats[2], feats[3])
            complete = int(rng.random() < p)
            passes.append((feats, complete))
            if not complete:
                realized = 0.0  # turnover
                break
            ball_owner = OFF_IDS.index(receiver.player_id)

        elif choice.action == "DRIVE":
            feats = F.drive_features(state, handler, choice.direction)
            p = true_drive_prob(feats[0], feats[1], feats[3], feats[4])
            success = int(rng.random() < p)
            drives.append((feats, success))
            if not success:
                realized = 0.0  # stripped / charge
                break
            # Success: handler collapses to a rim-zone position.
            off_pos = off_pos.copy()
            off_pos[ball_owner] = rim + rng.normal(0, 1.2, 2)

        elif choice.action == "SCREEN_WITH":
            # Slight relief: nudge the handler a step toward space.
            off_pos = off_pos.copy()
            off_pos[ball_owner] = off_pos[ball_owner] + rng.normal(0, 1.0, 2)

        # RESET / continue: defense re-anchors (tight on the new handler), clock ticks.
        def_pos = _shade_toward(off_pos, rim, feet=_cushion(ball_owner, rng)) + rng.normal(0, 0.5, (5, 2))
        shot_clock -= SHOTCLOCK_PER_STEP

    sample = PossessionSample(states=states, step_index=step_idx, realized_points=realized)
    return sample, shots, passes, drives


def _true_action_value(state: State, action, skills) -> float:
    """Ground-truth expected points of an action, for the behavior policy only."""
    h = state.handler
    if action.action == "SHOOT":
        p = true_make_prob(h.dist_to_rim, h.defender_pressure, F._is_three(h),
                           skills[h.player_id], F._catch_and_shoot(h))
        return p * (3 if F._is_three(h) else 2)
    if action.action == "PASS_TO":
        r = next(pp for pp in state.offense() if pp.player_id == action.target)
        pf = F.pass_features(state, h, r)
        pc = true_complete_prob(pf[0], pf[1], pf[2], pf[3])
        # Approx: receiver takes his shot next.
        pm = true_make_prob(r.dist_to_rim, r.defender_pressure, F._is_three(r),
                            skills[r.player_id], 1)
        return pc * pm * (3 if F._is_three(r) else 2)
    if action.action == "DRIVE":
        df = F.drive_features(state, h, action.direction)
        pd = true_drive_prob(df[0], df[1], df[3], df[4])
        return pd * 2 * true_make_prob(2.0, 0.3, 0, skills[h.player_id], 0)
    return 0.85  # SCREEN_WITH / RESET baseline continuation value


def _softmax(x, temp=0.5):
    z = x / max(temp, 1e-6)
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


def build_dataset(n_possessions: int = 2000, seed: int = 0, skills: dict | None = None) -> LabeledDataset:
    rng = np.random.default_rng(seed)
    # Skills are a property of the players, not the dataset -> fixed across seeds.
    skills = dict(PLAYER_SKILLS) if skills is None else skills

    shots, passes, drives, possessions = [], [], [], []
    for i in range(n_possessions):
        sample, s, p, d = simulate_possession(rng, skills, f"sim_p{i:05d}")
        possessions.append(sample)
        shots += s
        passes += p
        drives += d

    shot_X = np.array([r[0] for r in shots]) if shots else np.zeros((0, len(F.SHOT_FEATURES)))
    shot_y = np.array([r[1] for r in shots])
    shot_points = np.array([r[2] for r in shots])
    shot_player = np.array([r[3] for r in shots])
    pass_X = np.array([r[0] for r in passes]) if passes else np.zeros((0, len(F.PASS_FEATURES)))
    pass_y = np.array([r[1] for r in passes])
    drive_X = np.array([r[0] for r in drives]) if drives else np.zeros((0, len(F.DRIVE_FEATURES)))
    drive_y = np.array([r[1] for r in drives])

    return LabeledDataset(
        shot_X=shot_X, shot_y=shot_y, shot_points=shot_points, shot_player=shot_player,
        pass_X=pass_X, pass_y=pass_y, drive_X=drive_X, drive_y=drive_y,
        possessions=possessions, player_skill=skills, player_ids=OFF_IDS + DEF_IDS,
    )
