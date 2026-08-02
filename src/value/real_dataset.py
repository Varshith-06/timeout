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

from src.ingest.possessions import iter_possessions
from src.ingest.real_data import (load_pbp, load_shots, moment_positions,
                                  parsed_game_paths, pbp_possessions,
                                  possession_lookup)
from src.ingest.sportvu import parse_game
from src.state import court
from src.state.schema import build_state, defender_pressure
from src.value import features as F
from src.value.simulation import LabeledDataset, PossessionSample

TURNOVER = 5  # NBA play-by-play EVENTMSGTYPE
DRIVE_MIN_DROP = 6.0     # ft the handler must close on the rim to count as a drive
DRIVE_SUCCESS_RIM = 8.0  # ft from rim that counts as reaching a rim look


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


def _extract_passes(possession):
    """Completed passes as (pre_pass_frame_idx, passer_id, receiver_id).

    A completed pass is a handler A -> (ball in flight) -> handler B transition
    between two offensive players (roadmap 3.2b, completion side)."""
    passes = []
    frames = possession.frames
    last_handler, last_idx = None, None
    for i, fr in enumerate(frames):
        h = fr.handler_player_id
        if h is not None:
            if last_handler is not None and h != last_handler:
                passes.append((last_idx, last_handler, h))
            last_handler, last_idx = h, i
    return passes


def _driver_xy(frame, pid):
    for p in frame.players:
        if p.player_id == pid:
            return p.x, p.y
    return None


def _extract_drives(possession):
    """Drives as (start_frame_idx, driver_id, direction, success).

    Tracks the *specific* ball handler's rim approach over ~2s, following his
    position even through brief handler-tag gaps (fast dribbles), and stops only
    when the ball is clearly passed to a different player. Direction is the side
    of the floor the drive starts from (canonical: larger y is the left side)."""
    frames = possession.frames
    rim = court.ATTACK_RIM
    drives, i = [], 0
    while i < len(frames):
        h = frames[i].handler_player_id
        start = _driver_xy(frames[i], h) if h is not None else None
        if start is None:
            i += 1
            continue
        d0 = math.hypot(start[0] - rim[0], start[1] - rim[1])
        min_d, j = d0, i
        while j < len(frames) and j < i + 50:
            hj = frames[j].handler_player_id
            if hj is not None and hj != h:
                break  # ball passed to someone else
            pj = _driver_xy(frames[j], h)
            if pj is not None:
                min_d = min(min_d, math.hypot(pj[0] - rim[0], pj[1] - rim[1]))
            j += 1
        if d0 - min_d >= DRIVE_MIN_DROP:
            direction = "left" if start[1] > 29 else ("right" if start[1] < 21 else "middle")
            drives.append((i, h, direction, int(min_d <= DRIVE_SUCCESS_RIM)))
            i = max(j, i + 1)
        else:
            i += 1
    return drives


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
    all_players = set()
    for p in paths:
        game = parse_game(p)
        sx, sy, sp, spl = _extract_shots(game, load_shots(shots_csv, [game.game_id]))
        X += sx; y += sy; points += sp; players_col += spl
        all_players.update(spl)

    shot_X = np.array(X) if X else np.zeros((0, len(F.SHOT_FEATURES)))
    empty = np.zeros((0, len(F.PASS_FEATURES)))
    return LabeledDataset(
        shot_X=shot_X, shot_y=np.array(y), shot_points=np.array(points),
        shot_player=np.array(players_col),
        pass_X=empty, pass_y=np.array([]),
        drive_X=np.zeros((0, len(F.DRIVE_FEATURES))), drive_y=np.array([]),
        possessions=[], player_skill={}, player_ids=sorted(all_players),
    )


def _extract_shots(game, shots):
    X, y, pts, players = [], [], [], []
    for s in shots.iter_rows(named=True):
        pos, ball, shot_clock = moment_positions(
            game, s["quarter"], float(s["game_clock"]), shooter_id=s["player_id"])
        if pos is None:
            continue
        row = _shot_feature_row(pos, ball, s["player_id"], s["team_id"],
                                s["is_three"], shot_clock, s["shot_distance"])
        if row is None:
            continue
        X.append(row); y.append(int(s["made"])); pts.append(int(s["points"]))
        players.append(int(s["player_id"]))
    return X, y, pts, players


def build_real_dataset(json_dir, shots_csv, pbp_dir, max_games=None, paths=None,
                       subsample=12) -> LabeledDataset:
    """Full real-data LabeledDataset: shots, passes, drives, and possession values.

    Possession realized points come from the play-by-play join (roadmap 2.4);
    passes are the completed handler-to-handler transitions with turnover-ending
    possessions supplying negatives (3.2b); drives are handler rim-attacks labeled
    by whether they reached a rim look (3.2c). V(s) trains on the subsampled
    possession states -> realized points, with the opening seconds down-weighted.
    """
    from pathlib import Path
    if paths is None:
        paths = parsed_game_paths(json_dir)
        if max_games:
            paths = paths[:max_games]

    shot_X, shot_y, shot_pts, shot_pl = [], [], [], []
    pass_X, pass_y, drive_X, drive_y = [], [], [], []
    possessions, all_players = [], set()

    for p in paths:
        game = parse_game(p)
        pbp_path = Path(pbp_dir) / f"events_{game.game_id}.csv"
        if not pbp_path.exists():
            continue
        pbp = load_pbp(pbp_path)
        lookup = possession_lookup(pbp_possessions(pbp))

        sx, sy, sp, spl = _extract_shots(game, load_shots(shots_csv, [game.game_id]))
        shot_X += sx; shot_y += sy; shot_pts += sp; shot_pl += spl; all_players.update(spl)

        for poss in iter_possessions(game):
            if len(poss) < subsample:
                continue
            clocks = [fr.game_clock for fr in poss.frames]
            mid = sorted(clocks)[len(clocks) // 2]
            real = lookup(poss.frames[0].quarter, mid)
            # Skip fragments we can't map, or where the tracking-inferred offense
            # disagrees with the play-by-play — keeps the target consistent with
            # the entity tensor's offense/defense flags.
            if real is None or real["offense_team_id"] != poss.offense_team_id:
                continue
            pts = real["points"]
            is_turnover = pts == 0 and _has_turnover(
                pbp, poss.frames[0].quarter, real["end_clock"], real["start_clock"],
                poss.offense_team_id)

            # V(s): subsampled states -> realized points, opening seconds down-weighted.
            states, steps = [], []
            for i in range(0, len(poss.frames), subsample):
                fr = poss.frames[i]
                st = build_state(fr, roster_jersey=None, last_touch_time=None)
                states.append(st)
                steps.append(int(max(0.0, real["start_clock"] - fr.game_clock) // 2))
            if states:
                possessions.append(PossessionSample(states, steps, float(pts)))
                all_players.update(p.player_id for p in states[0].players)

            # Passes: completed transitions (label 1); last one is the turnover (label 0).
            passes = _extract_passes(poss)
            for k, (idx, passer_id, recv_id) in enumerate(passes):
                st = build_state(poss.frames[idx], roster_jersey=None, last_touch_time=None)
                passer = next((pp for pp in st.offense() if pp.player_id == passer_id), None)
                recv = next((pp for pp in st.offense() if pp.player_id == recv_id), None)
                if passer is None or recv is None:
                    continue
                label = 0 if (is_turnover and k == len(passes) - 1) else 1
                pass_X.append(F.pass_features(st, passer, recv)); pass_y.append(label)

            # Drives.
            for (idx, driver_id, direction, success) in _extract_drives(poss):
                st = build_state(poss.frames[idx], roster_jersey=None, last_touch_time=None)
                driver = next((pp for pp in st.offense() if pp.player_id == driver_id), None)
                if driver is None:
                    continue
                drive_X.append(F.drive_features(st, driver, direction)); drive_y.append(success)

    def arr(rows, ncol):
        return np.array(rows) if rows else np.zeros((0, ncol))

    return LabeledDataset(
        shot_X=arr(shot_X, len(F.SHOT_FEATURES)), shot_y=np.array(shot_y),
        shot_points=np.array(shot_pts), shot_player=np.array(shot_pl),
        pass_X=arr(pass_X, len(F.PASS_FEATURES)), pass_y=np.array(pass_y),
        drive_X=arr(drive_X, len(F.DRIVE_FEATURES)), drive_y=np.array(drive_y),
        possessions=possessions, player_skill={}, player_ids=sorted(all_players),
    )


def _has_turnover(pbp, quarter, lo, hi, offense_team_id) -> bool:
    sub = pbp.filter(
        (pl.col("quarter") == quarter)
        & (pl.col("game_clock") >= lo) & (pl.col("game_clock") <= hi)
        & (pl.col("event_type") == TURNOVER)
        & (pl.col("team_id").cast(pl.Int64, strict=False) == offense_team_id)
    )
    return sub.height > 0
