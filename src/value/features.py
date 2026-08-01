"""Feature extraction for the value sub-models and V(s) (roadmap 3.2).

Two families:
  * Tabular per-action features for the LightGBM sub-models (shot / pass / drive).
  * A permutation-invariant entity tensor for the state-value network V(s).

Feature order is frozen by the ``*_FEATURES`` name lists so a trained model and
inference always agree. Everything is derived from the deterministic state
schema (roadmap 2.5); nothing here is learned.
"""
from __future__ import annotations

import math

import numpy as np

from src.state import court
from src.state.schema import PlayerState, State

# --- Shot features (3.2a) ----------------------------------------------------
SHOT_FEATURES = [
    "dist_to_rim",
    "angle_to_rim_deg",
    "closest_defender_dist",
    "defender_angle_deg",
    "defender_pressure",
    "shot_clock",
    "is_three",
    "catch_and_shoot",
    "player_shot_prior",  # empirical-Bayes shrunk make rate for this player
]

# --- Pass features (3.2b) ----------------------------------------------------
PASS_FEATURES = [
    "pass_distance",
    "defenders_in_lane",
    "receiver_separation",
    "passer_facing_receiver",
    "receiver_pressure",
    "receiver_dist_to_rim",
]

# --- Drive features (3.2c) ---------------------------------------------------
DRIVE_FEATURES = [
    "help_defenders_on_side",
    "primary_defender_lateral",
    "primary_defender_dist",
    "driver_speed",
    "driver_dist_to_rim",
    "direction_sign",  # left=+1, middle=0, right=-1
]


def _is_three(ps: PlayerState) -> int:
    return int("three" in ps.zone)


def _catch_and_shoot(ps: PlayerState) -> int:
    """Roadmap 3.2a: catch-and-shoot vs off-the-dribble. Phase 1 proxy uses
    seconds_since_touch — a quick shot after a catch is catch-and-shoot."""
    sst = ps.seconds_since_touch
    if sst is None:
        return 0
    return int(sst <= 0.4)


def shot_features(state: State, shooter: PlayerState, player_prior: float = 0.45) -> np.ndarray:
    nd = shooter.nearest_defender
    shot_clock = state.timestamp.get("shot_clock")
    return np.array(
        [
            shooter.dist_to_rim,
            shooter.angle_to_rim_deg,
            nd.dist if nd else 30.0,
            nd.angle_deg if nd else 90.0,
            shooter.defender_pressure,
            shot_clock if shot_clock is not None else 12.0,
            _is_three(shooter),
            _catch_and_shoot(shooter),
            player_prior,
        ],
        dtype=float,
    )


# --- Pass-lane geometry ------------------------------------------------------
def _point_to_segment_dist(p, a, b) -> float:
    """Perpendicular distance from point p to segment ab, clamped to the segment."""
    p, a, b = np.asarray(p), np.asarray(a), np.asarray(b)
    ab = b - a
    denom = float(ab @ ab)
    if denom < 1e-9:
        return float(np.linalg.norm(p - a))
    t = float(np.clip((p - a) @ ab / denom, 0.0, 1.0))
    proj = a + t * ab
    return float(np.linalg.norm(p - proj))


def defenders_in_lane(state: State, passer: PlayerState, receiver: PlayerState, thresh: float = 3.0) -> int:
    a, b = (passer.x, passer.y), (receiver.x, receiver.y)
    return sum(
        1 for d in state.defense() if _point_to_segment_dist((d.x, d.y), a, b) < thresh
    )


def _facing(passer: PlayerState, receiver: PlayerState) -> float:
    """cos(angle) between passer heading and passer->receiver direction, in [-1,1].

    Phase 1 orientation is velocity heading (roadmap 2.5); when unavailable
    (near-stationary) we return 0 = neutral."""
    if passer.orientation_deg is None:
        return 0.0
    heading = math.radians(passer.orientation_deg)
    hv = np.array([math.cos(heading), math.sin(heading)])
    to_r = np.array([receiver.x - passer.x, receiver.y - passer.y])
    n = np.linalg.norm(to_r)
    if n < 1e-6:
        return 0.0
    return float(hv @ (to_r / n))


def pass_features(state: State, passer: PlayerState, receiver: PlayerState) -> np.ndarray:
    dist = math.hypot(receiver.x - passer.x, receiver.y - passer.y)
    sep = receiver.nearest_defender.dist if receiver.nearest_defender else 30.0
    return np.array(
        [
            dist,
            defenders_in_lane(state, passer, receiver),
            sep,
            _facing(passer, receiver),
            receiver.defender_pressure,
            receiver.dist_to_rim,
        ],
        dtype=float,
    )


# --- Drive features ----------------------------------------------------------
_DIR_SIGN = {"left": 1.0, "middle": 0.0, "right": -1.0}


def help_defenders_on_side(state: State, driver: PlayerState, direction: str) -> int:
    """Defenders between the driver and the rim on the drive side."""
    rim = court.ATTACK_RIM
    # Region: closer to the rim than the driver, and on the drive's y-side.
    count = 0
    for d in state.defense():
        if d.dist_to_rim >= driver.dist_to_rim:
            continue
        dy = d.y - rim[1]
        if direction == "left" and dy > 0:
            count += 1
        elif direction == "right" and dy < 0:
            count += 1
        elif direction == "middle" and abs(dy) < 8:
            count += 1
    return count


def drive_features(state: State, driver: PlayerState, direction: str) -> np.ndarray:
    nd = driver.nearest_defender
    lateral = 0.0
    if nd:
        defender = next((d for d in state.defense() if d.player_id == nd.player_id), None)
        if defender is not None:
            lateral = abs(defender.y - driver.y)
    return np.array(
        [
            help_defenders_on_side(state, driver, direction),
            lateral,
            nd.dist if nd else 30.0,
            driver.speed,
            driver.dist_to_rim,
            _DIR_SIGN[direction],
        ],
        dtype=float,
    )


# --- Entity tensor for V(s) (3.2d) -------------------------------------------
MAX_ENTITIES = 11  # 10 players + ball
ENTITY_DIM = 8     # x_rel, y_rel, vx, vy, is_off, is_def, is_ball, has_ball
GLOBAL_DIM = 3     # shot_clock_norm, spacing_norm, n_players_norm


def entity_tensor(state: State):
    """Return (entities[MAX,ENTITY_DIM], player_idx[MAX], mask[MAX], globals[GLOBAL_DIM]).

    Permutation-invariant by construction: the consumer pools over the entity
    axis. Padded to MAX_ENTITIES with a mask so a variable entity count (dropped
    players, roadmap 3.3) is handled natively. player_idx indexes a learned
    embedding; -1 (ball / unknown) maps to a dedicated row in the network.
    """
    rim = court.ATTACK_RIM
    ents = np.zeros((MAX_ENTITIES, ENTITY_DIM), dtype=np.float32)
    pidx = np.full(MAX_ENTITIES, -1, dtype=np.int64)
    mask = np.zeros(MAX_ENTITIES, dtype=np.float32)

    i = 0
    for p in state.players:
        if i >= MAX_ENTITIES - 1:  # leave a slot for the ball
            break
        ents[i] = [
            (p.x - rim[0]) / court.HALFCOURT_X,
            (p.y - rim[1]) / (court.COURT_WIDTH / 2),
            p.vx / 10.0,
            p.vy / 10.0,
            1.0 if p.side == "offense" else 0.0,
            1.0 if p.side == "defense" else 0.0,
            0.0,
            1.0 if p.has_ball else 0.0,
        ]
        pidx[i] = p.player_id
        mask[i] = 1.0
        i += 1

    # Ball as its own entity.
    b = state.ball
    ents[i] = [
        (b["x"] - rim[0]) / court.HALFCOURT_X,
        (b["y"] - rim[1]) / (court.COURT_WIDTH / 2),
        b["vx"] / 10.0,
        b["vy"] / 10.0,
        0.0, 0.0, 1.0, 0.0,
    ]
    mask[i] = 1.0

    shot_clock = state.timestamp.get("shot_clock")
    globals_ = np.array(
        [
            (shot_clock if shot_clock is not None else 12.0) / 24.0,
            state.context.spacing_area_sqft / 500.0,
            state.context.n_players_observed / 10.0,
        ],
        dtype=np.float32,
    )
    return ents, pidx, mask, globals_


class PlayerVocab:
    """Maps player_id -> a small contiguous embedding index (0 reserved for unk/ball)."""

    def __init__(self, player_ids=None):
        self._map = {-1: 0}
        if player_ids:
            for pid in sorted(set(player_ids)):
                self._map.setdefault(pid, len(self._map))

    def __len__(self):
        return len(self._map)

    def index(self, player_id: int) -> int:
        return self._map.get(player_id, 0)

    def encode(self, pidx_array: np.ndarray) -> np.ndarray:
        return np.array([self.index(int(p)) for p in pidx_array], dtype=np.int64)
