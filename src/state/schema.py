"""The deterministic state schema (roadmap 2.5).

This is the JSON object one might be tempted to ask an LLM to produce. It is
produced by code, deterministically, from tracking data. Every field is measured
or computed — none is inferred by a language model.

Build states at the possession level (:func:`build_states`) so per-player history
(seconds_since_touch) is available; a single :func:`build_state` is exposed for
one-off frames.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field

import numpy as np

from src.ingest.possessions import Frame, PlayerFrame, Possession
from src.state import court

# --- Tunables kept in one place (roadmap 2.5 field notes) -------------------
PRESSURE_LENGTH = 4.0     # ft, decay constant in defender_pressure
SPEED_EPS = 0.5           # ft/s; below this, velocity heading is meaningless
SCREEN_DIST = 3.5         # ft, teammate-to-handler proximity that flags a screen
MAN_SCHEME_MEDIAN = 7.0   # ft, median nearest-defender dist below which -> man


@dataclass
class NearestDefender:
    player_id: int
    dist: float
    angle_deg: float  # 0 = defender sits directly between player and rim


@dataclass
class PlayerState:
    player_id: int
    jersey: int | None
    team_id: int
    side: str
    x: float
    y: float
    vx: float
    vy: float
    speed: float
    orientation_deg: float | None
    orientation_source: str
    has_ball: bool
    dist_to_rim: float
    angle_to_rim_deg: float
    zone: str
    nearest_defender: NearestDefender | None
    defender_pressure: float
    seconds_since_touch: float | None


@dataclass
class Context:
    n_players_observed: int
    spacing_area_sqft: float
    defense_scheme: str
    active_screen: dict | None
    confidence: float


@dataclass
class State:
    timestamp: dict
    possession_id: str
    offense_team_id: int
    attacking_basket: list
    ball: dict
    players: list[PlayerState]
    context: Context

    def to_dict(self) -> dict:
        d = asdict(self)
        return d

    @property
    def handler(self) -> PlayerState | None:
        for p in self.players:
            if p.has_ball:
                return p
        return None

    def offense(self) -> list[PlayerState]:
        return [p for p in self.players if p.side == "offense"]

    def defense(self) -> list[PlayerState]:
        return [p for p in self.players if p.side == "defense"]


# --- Geometry helpers --------------------------------------------------------
def defender_pressure(player_xy, defender_xy, rim=court.ATTACK_RIM) -> float:
    """exp(-dist/L) up-weighted when the defender is between player and rim.

    Roadmap 2.5: kept in one place so it can be tuned. Range ~[0, 1.5]; a
    defender draped on the rim side reads higher than one trailing.
    """
    d = math.hypot(defender_xy[0] - player_xy[0], defender_xy[1] - player_xy[1])
    base = math.exp(-d / PRESSURE_LENGTH)
    # Alignment: 1 when defender is exactly on the player->rim ray, 0 when opposite.
    to_rim = np.array([rim[0] - player_xy[0], rim[1] - player_xy[1]])
    to_def = np.array([defender_xy[0] - player_xy[0], defender_xy[1] - player_xy[1]])
    nr, nd = np.linalg.norm(to_rim), np.linalg.norm(to_def)
    if nr < 1e-6 or nd < 1e-6:
        align = 0.0
    else:
        align = max(0.0, float(np.dot(to_rim, to_def) / (nr * nd)))
    return float(base * (1.0 + 0.5 * align))


def _angle_between(v1, v2) -> float:
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 < 1e-6 or n2 < 1e-6:
        return 0.0
    c = float(np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0))
    return math.degrees(math.acos(c))


def _nearest_defender(off: PlayerFrame, defenders, rim=court.ATTACK_RIM):
    best, best_d = None, math.inf
    for d in defenders:
        dist = math.hypot(d.x - off.x, d.y - off.y)
        if dist < best_d:
            best_d, best = dist, d
    if best is None:
        return None
    to_rim = np.array([rim[0] - off.x, rim[1] - off.y])
    to_def = np.array([best.x - off.x, best.y - off.y])
    return NearestDefender(
        player_id=best.player_id,
        dist=round(best_d, 2),
        angle_deg=round(_angle_between(to_rim, to_def), 1),
    )


def convex_hull_area(points: np.ndarray) -> float:
    """Area (sq ft) of the convex hull of >=3 points via monotone chain."""
    pts = sorted(map(tuple, points))
    if len(pts) < 3:
        return 0.0

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    hull = lower[:-1] + upper[:-1]
    # Shoelace.
    area = 0.0
    for i in range(len(hull)):
        x1, y1 = hull[i]
        x2, y2 = hull[(i + 1) % len(hull)]
        area += x1 * y2 - x2 * y1
    return abs(area) / 2.0


def _defense_scheme(offense, defense) -> str:
    """Rule-based (2.5): tight, consistent nearest-defender distances -> man."""
    if not offense or not defense:
        return "unknown"
    nearest = []
    for d in defense:
        dd = min(math.hypot(d.x - o.x, d.y - o.y) for o in offense)
        nearest.append(dd)
    return "man" if float(np.median(nearest)) < MAN_SCHEME_MEDIAN else "zone"


def _active_screen(handler, offense) -> dict | None:
    """Heuristic screen flag: a teammate parked next to the ball handler."""
    if handler is None:
        return None
    for o in offense:
        if o.player_id == handler.player_id:
            continue
        if math.hypot(o.x - handler.x, o.y - handler.y) <= SCREEN_DIST:
            return {"screener_id": o.player_id, "phase": "approach", "source": "heuristic"}
    return None


# --- Builders ----------------------------------------------------------------
def _orientation(pf: PlayerFrame) -> tuple[float | None, str]:
    """Phase 1 orientation is the velocity heading, explicitly marked (2.5)."""
    if pf.speed < SPEED_EPS:
        return None, "velocity_heading_unavailable"
    return round(math.degrees(math.atan2(pf.vy, pf.vx)), 1), "velocity_heading"


def build_state(
    frame: Frame,
    roster_jersey: dict[int, int] | None = None,
    last_touch_time: dict[int, float] | None = None,
) -> State:
    """Build one :class:`State` from a :class:`Frame`.

    roster_jersey: player_id -> jersey number (for the OCR bridge later).
    last_touch_time: player_id -> game_clock when they last held the ball, used
    for seconds_since_touch. game_clock counts down, so elapsed = last - current.
    """
    roster_jersey = roster_jersey or {}
    last_touch_time = last_touch_time or {}
    rim = court.ATTACK_RIM

    offense_pf = [p for p in frame.players if p.side == "offense"]
    defense_pf = [p for p in frame.players if p.side == "defense"]

    player_states: list[PlayerState] = []
    for pf in frame.players:
        dist_to_rim, angle_to_rim = court.to_polar(pf.x, pf.y, rim)
        nearest = _nearest_defender(pf, defense_pf, rim) if pf.side == "offense" else None
        pressure = 0.0
        if nearest is not None:
            defender = next(d for d in defense_pf if d.player_id == nearest.player_id)
            pressure = round(defender_pressure((pf.x, pf.y), (defender.x, defender.y), rim), 3)
        orient, orient_src = _orientation(pf)
        has_ball = frame.handler_player_id == pf.player_id
        sst = None
        if pf.player_id in last_touch_time and frame.game_clock is not None:
            sst = round(max(0.0, last_touch_time[pf.player_id] - frame.game_clock), 2)
        if has_ball:
            sst = 0.0

        player_states.append(
            PlayerState(
                player_id=pf.player_id,
                jersey=roster_jersey.get(pf.player_id),
                team_id=pf.team_id,
                side=pf.side,
                x=round(pf.x, 2), y=round(pf.y, 2),
                vx=round(pf.vx, 2), vy=round(pf.vy, 2), speed=round(pf.speed, 2),
                orientation_deg=orient, orientation_source=orient_src,
                has_ball=has_ball,
                dist_to_rim=round(float(dist_to_rim), 2),
                angle_to_rim_deg=round(float(angle_to_rim), 1),
                zone=court.assign_zone(pf.x, pf.y, rim),
                nearest_defender=nearest,
                defender_pressure=pressure,
                seconds_since_touch=sst,
            )
        )

    handler_state = next((p for p in player_states if p.has_ball), None)
    spacing = convex_hull_area(np.array([[p.x, p.y] for p in offense_pf])) if len(offense_pf) >= 3 else 0.0

    context = Context(
        n_players_observed=len(frame.players),
        spacing_area_sqft=round(spacing, 1),
        defense_scheme=_defense_scheme(offense_pf, defense_pf),
        active_screen=_active_screen(handler_state, [p for p in player_states if p.side == "offense"]),
        confidence=1.0,  # always 1.0 in Phase 1 (perfect tracking)
    )

    return State(
        timestamp={
            "quarter": frame.quarter,
            "game_clock": frame.game_clock,
            "shot_clock": frame.shot_clock,
        },
        possession_id=frame.possession_id,
        offense_team_id=frame.offense_team_id,
        attacking_basket=list(frame.attacking_basket),
        ball={
            "x": round(frame.ball_x, 2), "y": round(frame.ball_y, 2), "z": round(frame.ball_z, 2),
            "vx": round(frame.ball_vx, 2), "vy": round(frame.ball_vy, 2),
            "in_flight": frame.ball_in_flight,
        },
        players=player_states,
        context=context,
    )


def build_states(possession: Possession, roster_jersey: dict[int, int] | None = None) -> list[State]:
    """Build states for every frame, threading seconds_since_touch through time."""
    states = []
    last_touch: dict[int, float] = {}
    for fr in possession.frames:
        if fr.handler_player_id is not None:
            last_touch[fr.handler_player_id] = fr.game_clock
        states.append(build_state(fr, roster_jersey, dict(last_touch)))
    return states


def roster_jersey_map(game) -> dict[int, int]:
    """player_id -> jersey from a parsed Game's roster frame."""
    out = {}
    for pid, jersey in game.roster.select(["player_id", "jersey"]).iter_rows():
        if jersey is not None:
            out[pid] = jersey
    return out
