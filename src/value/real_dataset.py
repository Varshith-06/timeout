"""Build training data from REAL tracking + labels (roadmap 3.2).

Produces the same :class:`src.value.simulation.LabeledDataset` shape the Phase 2
trainers already consume — but every row comes from real 2015-16 SportVU
tracking joined to real shot / play-by-play labels, not the physics simulator.

Currently implemented: the shot make-probability model (3.2a) — the flagship
sub-model and the one whose labels (make/miss, 2PT/3PT) are directly available.
Features are extracted from the tracking at each shot's moment, in the same
canonical (attack-left) convention as the synthetic features, so the trained
model and its calibration transfer.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import polars as pl

from src.ingest.real_data import load_shots, moment_positions, parsed_game_paths
from src.ingest.sportvu import parse_game
from src.state import court
from src.state.schema import defender_pressure
from src.value import features as F
from src.value.simulation import LabeledDataset


def _shot_feature_row(players, ball, shooter_id, shooter_team, is_three, shot_clock,
                      shot_distance):
    """Extract a shot feature vector (SHOT_FEATURES order) at one moment.

    ``shot_distance`` (feet, from the shots CSV) is the authoritative distance;
    the tracking gives the defender context. Coordinates are flipped so the
    attacked rim is always BASKET_LEFT — the convention the synthetic features use.
    """
    if shooter_id not in players:
        return None
    sx, sy, _ = players[shooter_id]
    # Attack the nearer rim; flip to canonical (attack-left) if on the right half.
    if sx > court.HALFCOURT_X:
        pos = {pid: (court.COURT_LENGTH - x, court.COURT_WIDTH - y, t)
               for pid, (x, y, t) in players.items()}
    else:
        pos = players
    sx, sy, _ = pos[shooter_id]
    rim = court.ATTACK_RIM
    dist_to_rim = float(shot_distance)  # exact, from the shot label
    angle_to_rim = abs(math.degrees(math.atan2(sy - rim[1], sx - rim[0])))

    # Nearest defender (opposite team).
    best_d, best = math.inf, None
    for pid, (x, y, t) in pos.items():
        if t == shooter_team or pid == shooter_id:
            continue
        d = math.hypot(x - sx, y - sy)
        if d < best_d:
            best_d, best = d, (x, y)
    if best is None:
        best_d, best = 30.0, (sx, sy - 30.0)
    to_rim = np.array([rim[0] - sx, rim[1] - sy])
    to_def = np.array([best[0] - sx, best[1] - sy])
    n1, n2 = np.linalg.norm(to_rim), np.linalg.norm(to_def)
    def_angle = math.degrees(math.acos(np.clip(np.dot(to_rim, to_def) / (n1 * n2 + 1e-9), -1, 1)))
    pressure = defender_pressure((sx, sy), best, rim)

    return np.array([
        dist_to_rim,
        angle_to_rim,
        best_d,
        def_angle,
        pressure,
        shot_clock if shot_clock is not None else 12.0,
        float(is_three),
        0.0,          # catch_and_shoot: not derivable from a single frame; left 0
        0.45,         # player_shot_prior placeholder (ShotModel injects the EB prior)
    ], dtype=float)


def build_real_shot_dataset(json_dir, shots_csv, max_games: int | None = None,
                            paths=None) -> LabeledDataset:
    """Join real shots to real tracking and return a shot-only LabeledDataset.

    Pass ``paths`` (a list of game JSON paths) to control the exact games — used
    for a game-level train/test split that avoids frame leakage.
    """
    if paths is None:
        paths = parsed_game_paths(json_dir)
        if max_games:
            paths = paths[:max_games]

    X, y, points, players_col = [], [], [], []
    game_ids = []
    all_players = set()
    for p in paths:
        game = parse_game(p)
        game_ids.append(game.game_id)
        shots = load_shots(shots_csv, [game.game_id])
        for s in shots.iter_rows(named=True):
            pos, ball, shot_clock = moment_positions(
                game, s["quarter"], float(s["game_clock"]), shooter_id=s["player_id"])
            if pos is None:
                continue
            row = _shot_feature_row(pos, ball, s["player_id"], s["team_id"],
                                    s["is_three"], shot_clock, s["shot_distance"])
            if row is None:
                continue
            X.append(row); y.append(int(s["made"])); points.append(int(s["points"]))
            players_col.append(int(s["player_id"])); all_players.add(int(s["player_id"]))

    shot_X = np.array(X) if X else np.zeros((0, len(F.SHOT_FEATURES)))
    empty = np.zeros((0, len(F.PASS_FEATURES)))
    return LabeledDataset(
        shot_X=shot_X, shot_y=np.array(y), shot_points=np.array(points),
        shot_player=np.array(players_col),
        pass_X=empty, pass_y=np.array([]),
        drive_X=np.zeros((0, len(F.DRIVE_FEATURES))), drive_y=np.array([]),
        possessions=[], player_skill={}, player_ids=sorted(all_players),
    )
