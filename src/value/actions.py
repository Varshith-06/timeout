"""Candidate action enumeration (roadmap 2.6).

For a given state, generate the legal action set for the ball handler. Small and
unambiguous — thirteen candidates maximum:

    PASS_TO(teammate)   x4    DRIVE(dir)  x3   SHOOT x1
    SCREEN_WITH(mate)   x4    RESET       x1

Each candidate is a *structured object*, never a coordinate. The renderer
resolves it to geometry; the model never emits pixels or feet.

Illegal candidates are pruned before scoring (2.6):
  * no SHOOT beyond ~32 ft unless shot clock < 2
  * no DRIVE(left) if already on the left sideline (and symmetrically)
  * no SCREEN_WITH a teammate more than ~25 ft away
"""
from __future__ import annotations

from dataclasses import dataclass, field

from src.state import court
from src.state.schema import State

# Pruning thresholds (2.6).
MAX_SHOT_DIST = 32.0        # ft
SHOTCLOCK_HEAVE = 2.0       # s; below this a desperation heave is legal
SIDELINE_MARGIN = 3.0       # ft from a sideline counts as "on" it
SCREEN_MAX_DIST = 25.0      # ft, max handler-to-screener distance
RIM_ZONE = 4.0             # ft; inside this a DRIVE(middle) is pointless

DRIVE_DIRECTIONS = ("left", "middle", "right")


@dataclass
class Action:
    action: str                 # PASS_TO | DRIVE | SHOOT | SCREEN_WITH | RESET
    actor: int                  # ball-handler player_id
    id: str
    target: int | None = None   # teammate player_id (PASS_TO / SCREEN_WITH)
    direction: str | None = None  # DRIVE only: left | middle | right
    meta: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = {"action": self.action, "actor": self.actor, "id": self.id}
        if self.target is not None:
            d["target"] = self.target
        if self.direction is not None:
            d["direction"] = self.direction
        if self.meta:
            d["meta"] = self.meta
        return d


def _drive_legal(handler, direction: str) -> bool:
    # Canonical frame: attacking BASKET_LEFT, larger y = offense's left.
    if direction == "left":
        return handler.y < court.COURT_WIDTH - SIDELINE_MARGIN
    if direction == "right":
        return handler.y > SIDELINE_MARGIN
    # middle
    return handler.dist_to_rim > RIM_ZONE


def _shoot_legal(handler, shot_clock) -> bool:
    if handler.dist_to_rim <= MAX_SHOT_DIST:
        return True
    return shot_clock is not None and shot_clock < SHOTCLOCK_HEAVE


def enumerate_actions(state: State) -> list[Action]:
    """Return the pruned, legal candidate set for the current ball handler.

    Empty if the ball is in flight (no ball-handler decision to make) — those
    frames are a pass or a shot already underway (roadmap 2.4/2.6).
    """
    handler = state.handler
    if handler is None:
        return []

    actor = handler.player_id
    teammates = [p for p in state.offense() if p.player_id != actor]
    shot_clock = state.timestamp.get("shot_clock")

    actions: list[Action] = []
    n = 0

    def _next_id() -> str:
        nonlocal n
        n += 1
        return f"a{n}"

    # SHOOT
    if _shoot_legal(handler, shot_clock):
        actions.append(
            Action("SHOOT", actor, _next_id(),
                   meta={"zone": handler.zone, "dist_to_rim": handler.dist_to_rim})
        )

    # PASS_TO each teammate
    for t in teammates:
        actions.append(
            Action("PASS_TO", actor, _next_id(), target=t.player_id,
                   meta={"target_zone": t.zone, "target_open": round(1.0 - t.defender_pressure, 3)})
        )

    # DRIVE(direction)
    for d in DRIVE_DIRECTIONS:
        if _drive_legal(handler, d):
            actions.append(Action("DRIVE", actor, _next_id(), direction=d))

    # SCREEN_WITH each nearby teammate
    for t in teammates:
        dist = ((t.x - handler.x) ** 2 + (t.y - handler.y) ** 2) ** 0.5
        if dist <= SCREEN_MAX_DIST:
            actions.append(
                Action("SCREEN_WITH", actor, _next_id(), target=t.player_id,
                       meta={"screener_dist": round(dist, 1)})
            )

    # RESET
    actions.append(Action("RESET", actor, _next_id()))

    return actions
