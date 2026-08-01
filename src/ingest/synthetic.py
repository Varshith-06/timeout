"""Generate synthetic tracking in the *real* SportVU JSON shape.

Purpose: make the entire Phase 1 pipeline runnable end-to-end with zero
downloads. The output dict is byte-compatible with what
:func:`src.ingest.sportvu.parse_game` expects, so nothing downstream knows or
cares that the data is synthetic. Replace with real logs (setup_data.sh) and the
same code path runs unchanged.

The motion is deliberately simple but basketball-plausible: a half-court set,
man defenders shading toward the rim, a ball handler who dribbles, one pass, and
a terminal shot. It is enough to exercise segmentation, the state schema, action
enumeration, and the renderer.
"""
from __future__ import annotations

import numpy as np

from src.state import court

HZ = 25
DT = 1.0 / HZ

_HOME_PLAYERS = [
    (2547, "Alpha", "One", 13, "G"),
    (201609, "Bravo", "Two", 7, "G"),
    (202710, "Charlie", "Three", 21, "F"),
    (203110, "Delta", "Four", 32, "F"),
    (203500, "Echo", "Five", 44, "C"),
]
_VISITOR_PLAYERS = [
    (201142, "Foxtrot", "Six", 5, "G"),
    (203081, "Golf", "Seven", 11, "G"),
    (2544, "Hotel", "Eight", 23, "F"),
    (201939, "India", "Nine", 30, "F"),
    (201935, "Juliet", "Ten", 42, "C"),
]

HOME_TEAM_ID = 1610612748
VISITOR_TEAM_ID = 1610612739


def _roster_block(team_id, abbr, players):
    return {
        "teamid": team_id,
        "name": abbr,
        "abbreviation": abbr,
        "players": [
            {
                "playerid": pid,
                "firstname": fn,
                "lastname": ln,
                "jersey": str(jersey),
                "position": pos,
            }
            for (pid, fn, ln, jersey, pos) in players
        ],
    }


def _shade_toward(spots, rim, feet=3.0):
    """Move each spot ``feet`` toward the rim, staying near the man."""
    to_rim = rim - spots
    dist = np.linalg.norm(to_rim, axis=1, keepdims=True)
    unit = np.divide(to_rim, dist, out=np.zeros_like(to_rim), where=dist > 1e-6)
    feet = np.asarray(feet, dtype=float)
    if feet.ndim == 1:  # per-defender cushion -> column vector for broadcasting
        feet = feet[:, None]
    return spots + unit * feet


def _halfcourt_set(rim, rng):
    """Five offensive spots around the attacking rim, plus small jitter."""
    rx, ry = rim
    sign = 1.0 if rx < court.HALFCOURT_X else -1.0  # +x points into the court
    spots = np.array(
        [
            [rx + sign * 19.0, ry],          # 0: PG at the top
            [rx + sign * 9.0, ry - 17.0],    # 1: right wing
            [rx + sign * 9.0, ry + 17.0],    # 2: left wing
            [rx + sign * 3.0, ry - 8.0],     # 3: right corner-ish
            [rx + sign * 6.0, ry + 10.0],    # 4: left elbow / big
        ],
        dtype=float,
    )
    spots += rng.normal(0, 0.6, spots.shape)
    return spots


def generate_game(
    n_possessions: int = 2,
    frames_per_possession: int = 125,
    attack: str = "left",
    seed: int = 0,
) -> dict:
    """Build a full game dict with ``n_possessions`` events.

    attack: "left", "right", or "mixed" (alternate) — controls which basket the
    offense attacks in raw coordinates, so the half-court flip gets exercised.
    """
    rng = np.random.default_rng(seed)
    events = []
    game_clock = 700.0
    unix_ms = 1_450_000_000_000

    for p in range(n_possessions):
        if attack == "mixed":
            side = "left" if p % 2 == 0 else "right"
        else:
            side = attack
        rim = np.array(court.BASKET_LEFT if side == "left" else court.BASKET_RIGHT)

        off_spots = _halfcourt_set(rim, rng)
        # Man defenders sit ~3 ft rim-side of their assignment.
        def_spots = _shade_toward(off_spots, rim, feet=3.0)
        def_spots += rng.normal(0, 0.4, def_spots.shape)

        handler = 0  # PG starts with the ball
        pass_frame = int(frames_per_possession * 0.55)
        shot_frame = int(frames_per_possession * 0.85)
        receiver = 2  # PG swings to the left wing, who shoots

        # Per-frame random-walk velocities for a little life.
        off_vel = rng.normal(0, 0.4, off_spots.shape)
        def_vel = rng.normal(0, 0.4, def_spots.shape)

        moments = []
        shot_clock = 22.0
        ball_z = 3.5
        current_handler = handler

        for f in range(frames_per_possession):
            # Drift offense/defense with mild mean-reversion to keep the set.
            off_vel = 0.9 * off_vel + rng.normal(0, 0.15, off_spots.shape)
            def_vel = 0.9 * def_vel + rng.normal(0, 0.15, def_spots.shape)
            off_spots = off_spots + off_vel * DT
            # Defense tracks its man, staying ~3 ft rim-side.
            def_target = _shade_toward(off_spots, rim, feet=3.0)
            def_spots = def_spots + 0.15 * (def_target - def_spots) + def_vel * DT

            # Ball logic. Keep positional jitter tiny: the real SportVU ball
            # moves smoothly, and the handler test compares ball vs player
            # velocity with a 6 ft/s tolerance (see possessions.HANDLER_MAX_VEL_DIFF).
            if f < pass_frame:
                current_handler = handler
                ball_xy = off_spots[handler] + rng.normal(0, 0.04, 2)
                ball_z = 3.5 + rng.normal(0, 0.1)
            elif f < shot_frame:
                # Pass in flight then held by the receiver.
                if f < pass_frame + 8:
                    t = (f - pass_frame) / 8.0
                    ball_xy = (1 - t) * off_spots[handler] + t * off_spots[receiver]
                    ball_z = 5.0
                    current_handler = -1  # in flight
                else:
                    current_handler = receiver
                    ball_xy = off_spots[receiver] + rng.normal(0, 0.04, 2)
                    ball_z = 3.5
            else:
                # Shot in flight: arc up toward the rim.
                t = (f - shot_frame) / max(1, frames_per_possession - shot_frame)
                ball_xy = (1 - t) * off_spots[receiver] + t * rim
                ball_z = 3.5 + 9.0 * np.sin(np.pi * t)  # up and back down
                current_handler = -1

            # Ball entry first: team_id=-1, player_id=-1, z is height in feet.
            coords = [[-1, -1, float(ball_xy[0]), float(ball_xy[1]), float(ball_z)]]
            for i, (pid, *_rest) in enumerate(_HOME_PLAYERS):
                coords.append([HOME_TEAM_ID, pid, float(off_spots[i, 0]), float(off_spots[i, 1]), 0.0])
            for i, (pid, *_rest) in enumerate(_VISITOR_PLAYERS):
                coords.append([VISITOR_TEAM_ID, pid, float(def_spots[i, 0]), float(def_spots[i, 1]), 0.0])

            moments.append([1, unix_ms, round(game_clock, 2), round(shot_clock, 2), None, coords])

            game_clock -= DT
            shot_clock = max(0.0, shot_clock - DT)
            unix_ms += int(DT * 1000)

        events.append(
            {
                "eventId": str(300 + p),
                "home": _roster_block(HOME_TEAM_ID, "HOM", _HOME_PLAYERS),
                "visitor": _roster_block(VISITOR_TEAM_ID, "VIS", _VISITOR_PLAYERS),
                "moments": moments,
                # Non-standard hint fields the parser ignores but tests can read.
                "_meta": {
                    "offense_team_id": HOME_TEAM_ID,
                    "attack_side": side,
                    "shooter_id": _HOME_PLAYERS[receiver][0],
                    "shot_frame": shot_frame,
                },
            }
        )
        game_clock -= 3.0  # gap between possessions

    return {
        "gameid": f"00215{seed:05d}",
        "gamedate": "2015-12-23",
        "events": events,
    }
