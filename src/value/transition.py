"""Approximate one-step transition s -> s'_a (roadmap 3.1).

To score a non-terminal action we predict the state it produces and evaluate V
there. This is a deliberately simple *geometric* transition — the ball and the
acting player move; defenders are left where they are (they have not yet
reacted). It is a one-step lookahead on purpose: deeper trees compound
prediction error until the leaves are noise (3.1).

The transition never invents information beyond moving the ball/handler; success
or failure of the action is handled probabilistically in :mod:`src.value.scoring`.
"""
from __future__ import annotations

import numpy as np

from src.ingest.possessions import Frame, PlayerFrame
from src.state import court
from src.state.schema import State, build_state

DRIVE_STEP_FT = 8.0
DRIVE_LATERAL_FT = 4.0


def frame_from_state(state: State) -> Frame:
    """Reconstruct a mutable Frame from a built State."""
    players = [
        PlayerFrame(p.player_id, p.team_id, p.side, p.x, p.y, p.vx, p.vy)
        for p in state.players
    ]
    b = state.ball
    handler = state.handler
    return Frame(
        possession_id=state.possession_id,
        frame_idx=0,
        quarter=state.timestamp["quarter"],
        game_clock=state.timestamp["game_clock"],
        shot_clock=state.timestamp["shot_clock"],
        ball_x=b["x"], ball_y=b["y"], ball_z=b["z"], ball_vx=b["vx"], ball_vy=b["vy"],
        handler_player_id=handler.player_id if handler else None,
        ball_in_flight=b["in_flight"],
        players=players,
        offense_team_id=state.offense_team_id,
    )


def _rebuild(frame: Frame) -> State:
    return build_state(frame, roster_jersey=None, last_touch_time=None)


def _guarding_defender(frame: Frame, handler: PlayerFrame) -> PlayerFrame | None:
    """The defender currently closest to the handler."""
    defs = [p for p in frame.players if p.side == "defense"]
    if not defs:
        return None
    return min(defs, key=lambda d: (d.x - handler.x) ** 2 + (d.y - handler.y) ** 2)


def _move_handler(frame, handler, new_xy):
    """Move the handler to new_xy and drag their guarding defender along.

    Without this the relocated handler would look artificially open (his
    defender left behind), which inflates the leaf shot value of any
    handler-moving action (DRIVE / RESET / SCREEN)."""
    defender = _guarding_defender(frame, handler)
    dx, dy = new_xy[0] - handler.x, new_xy[1] - handler.y
    handler.x, handler.y = float(new_xy[0]), float(new_xy[1])
    if defender is not None:
        defender.x += dx
        defender.y += dy
    frame.ball_x, frame.ball_y = handler.x, handler.y


def apply_action(state: State, action) -> State | None:
    """Return the approximate resulting state, or None for a terminal SHOOT."""
    if action.action == "SHOOT":
        return None  # terminal; scored as P(make) * points, no s'

    frame = frame_from_state(state)
    pmap = {p.player_id: p for p in frame.players}
    handler = pmap.get(action.actor)
    if handler is None:
        return state
    rim = np.array(court.BASKET_LEFT)

    if action.action == "PASS_TO":
        recv = pmap.get(action.target)
        if recv is not None:
            frame.ball_x, frame.ball_y = recv.x, recv.y
            frame.handler_player_id = recv.player_id

    elif action.action == "DRIVE":
        pos = np.array([handler.x, handler.y])
        to_rim = rim - pos
        d = np.linalg.norm(to_rim)
        step = min(DRIVE_STEP_FT, max(0.0, d - 2.0))
        unit = to_rim / d if d > 1e-6 else np.zeros(2)
        perp = np.array([-unit[1], unit[0]])
        lat = {"left": DRIVE_LATERAL_FT, "right": -DRIVE_LATERAL_FT, "middle": 0.0}[action.direction]
        new = pos + unit * step + perp * lat
        # On a *successful* drive the handler beats his man, so the guarding
        # defender only partially recovers — leave a step of separation.
        defender = _guarding_defender(frame, handler)
        handler.x, handler.y = float(new[0]), float(new[1])
        handler.vx, handler.vy = float(unit[0] * 6), float(unit[1] * 6)
        if defender is not None:
            defender.x += (new[0] - pos[0]) * 0.7
            defender.y += (new[1] - pos[1]) * 0.7
        frame.ball_x, frame.ball_y = handler.x, handler.y

    elif action.action == "SCREEN_WITH":
        # Handler uses the screen: nudge toward the middle; his man trails.
        ny = handler.y + (2.0 if handler.y < 25 else -2.0)
        _move_handler(frame, handler, (handler.x, ny))

    elif action.action == "RESET":
        top = rim + np.array([24.0, 0.0])
        _move_handler(frame, handler, (float(top[0]), float(top[1])))

    return _rebuild(frame)
