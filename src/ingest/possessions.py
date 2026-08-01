"""Possession segmentation and per-frame ball-handler assignment (roadmap 2.4).

Production segmentation joins to play-by-play on (quarter, game_clock); that hook
is documented below. For a self-contained Phase 1 we segment on the two signals
present in the tracking itself — event boundaries and shot-clock resets — which
is enough to drive the state schema, action enumeration, and renderer.

Everything a possession emits is in *canonical* coordinates: the half-court flip
(court.flip_to_left) has been applied so the attacking basket is always
BASKET_LEFT. Velocities are finite differences in that canonical frame, in ft/s.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

import numpy as np
import polars as pl

from src.ingest.sportvu import BALL_PLAYER_ID, Game
from src.state import court

# Ball-handler thresholds (roadmap 2.4).
HANDLER_MAX_DIST = 4.0        # ft, player-to-ball
HANDLER_MAX_BALL_Z = 9.0      # ft, above this the ball is a shot/lob in flight
HANDLER_MAX_VEL_DIFF = 6.0    # ft/s, ball must move with the player
HANDLER_MIN_HOLD = 5          # consecutive frames (0.2s) to count as a handler

# Segmentation.
SHOTCLOCK_RESET_JUMP = 3.0    # ft... seconds: an upward jump implies a new possession
SHOT_Z_THRESHOLD = 10.0       # ball height that marks a probable shot attempt


@dataclass
class PlayerFrame:
    player_id: int
    team_id: int
    side: str          # "offense" | "defense"
    x: float
    y: float
    vx: float
    vy: float

    @property
    def speed(self) -> float:
        return float(np.hypot(self.vx, self.vy))


@dataclass
class Frame:
    possession_id: str
    frame_idx: int
    quarter: int
    game_clock: float
    shot_clock: float | None
    ball_x: float
    ball_y: float
    ball_z: float
    ball_vx: float
    ball_vy: float
    handler_player_id: int | None
    ball_in_flight: bool
    players: list[PlayerFrame]
    offense_team_id: int
    attacking_basket: tuple = court.BASKET_LEFT


@dataclass
class Possession:
    possession_id: str
    offense_team_id: int
    attacking_basket: tuple
    frames: list[Frame] = field(default_factory=list)
    terminal_event: str | None = None
    points_scored: float | None = None

    def __len__(self) -> int:
        return len(self.frames)


# --- Segmentation ------------------------------------------------------------
def _split_indices_on_shotclock(shot_clock: np.ndarray) -> list[int]:
    """Boundary frame indices where the shot clock jumps upward (a reset)."""
    boundaries = [0]
    for i in range(1, len(shot_clock)):
        prev, cur = shot_clock[i - 1], shot_clock[i]
        if np.isnan(prev) or np.isnan(cur):
            continue
        if cur - prev > SHOTCLOCK_RESET_JUMP:
            boundaries.append(i)
    return boundaries


def _moment_arrays(sub: pl.DataFrame):
    """Turn a long per-event frame into ordered per-moment arrays.

    Returns (times, ball[N,3], players list-of-dicts per moment). Moments are
    ordered by descending game_clock (time increasing).
    """
    sub = sub.sort("game_clock", descending=True)
    # Unique moments in time order.
    moment_keys = sub.select(["game_clock", "shot_clock"]).unique(maintain_order=True)
    return sub, moment_keys


# --- Velocity ----------------------------------------------------------------
def _finite_diff(pos: np.ndarray, dt: np.ndarray) -> np.ndarray:
    """Backward finite difference along axis 0; first row is zero."""
    vel = np.zeros_like(pos)
    if len(pos) < 2:
        return vel
    dpos = np.diff(pos, axis=0)
    safe_dt = np.where(dt[1:, None] > 1e-4, dt[1:, None], DT_FALLBACK)
    vel[1:] = dpos / safe_dt
    return vel


DT_FALLBACK = 1.0 / 25


def iter_possessions(game: Game):
    """Yield :class:`Possession` objects in canonical coordinates.

    Strategy: group by SportVU event, then sub-split on shot-clock resets. Real
    deployments should replace this with a play-by-play join (see module docstring).
    """
    moments = game.moments
    if moments.height == 0:
        return

    for event_id, ev_df in moments.group_by("event_id", maintain_order=True):
        ev_id = event_id[0] if isinstance(event_id, tuple) else event_id
        ev_df = ev_df.sort("game_clock", descending=True)

        # Build ordered list of unique moments with their coordinate rows.
        moment_index = (
            ev_df.select(["quarter", "game_clock", "shot_clock"])
            .unique(maintain_order=True)
        )
        n = moment_index.height
        if n < HANDLER_MIN_HOLD:
            continue

        game_clock = moment_index["game_clock"].to_numpy()
        shot_clock = moment_index["shot_clock"].to_numpy().astype(float)
        # Forward-fill shot_clock within the event (roadmap 2.2).
        shot_clock = _forward_fill(shot_clock)

        # Map (game_clock) -> row order.
        gc_to_idx = {round(float(g), 3): i for i, g in enumerate(game_clock)}

        # Assemble position tensors: ball[n,3], and per-player[n,2] keyed by pid.
        player_ids = sorted(
            ev_df.filter(pl.col("player_id") != BALL_PLAYER_ID)["player_id"].unique().to_list()
        )
        pid_team = dict(
            ev_df.filter(pl.col("player_id") != BALL_PLAYER_ID)
            .select(["player_id", "team_id"]).unique().iter_rows()
        )
        pos = {pid: np.full((n, 2), np.nan) for pid in player_ids}
        ball = np.full((n, 3), np.nan)

        for row in ev_df.iter_rows(named=True):
            i = gc_to_idx.get(round(float(row["game_clock"]), 3))
            if i is None:
                continue
            if row["player_id"] == BALL_PLAYER_ID:
                ball[i] = (row["x"], row["y"], row["z"])
            else:
                pos[row["player_id"]][i] = (row["x"], row["y"])

        # dt between consecutive moments (positive seconds).
        dt = np.zeros(n)
        dt[1:] = np.clip(game_clock[:-1] - game_clock[1:], 0, None)

        # Determine offense and attacking basket, then flip to canonical.
        offense_team_id = _infer_offense(ball, pos, pid_team)
        off_ids = [p for p in player_ids if pid_team[p] == offense_team_id]
        mean_off_x = np.nanmean([np.nanmean(pos[p][:, 0]) for p in off_ids]) if off_ids else 0.0
        flip = mean_off_x > court.HALFCOURT_X

        if flip:
            ball[:, 0], ball[:, 1] = court.flip_to_left(ball[:, 0], ball[:, 1])
            for pid in player_ids:
                pos[pid][:, 0], pos[pid][:, 1] = court.flip_to_left(pos[pid][:, 0], pos[pid][:, 1])

        # Velocities in canonical frame.
        ball_vel = _finite_diff(ball[:, :2], dt)
        vel = {pid: _finite_diff(pos[pid], dt) for pid in player_ids}

        # Split this event into possessions on shot-clock resets.
        boundaries = _split_indices_on_shotclock(shot_clock)
        boundaries.append(n)
        for b in range(len(boundaries) - 1):
            lo, hi = boundaries[b], boundaries[b + 1]
            if hi - lo < HANDLER_MIN_HOLD:
                continue
            poss_id = f"{game.game_id}_e{ev_id}_p{b}"
            poss = _build_possession(
                poss_id, lo, hi, game_clock, shot_clock, ball, ball_vel,
                pos, vel, player_ids, pid_team, offense_team_id,
                moment_index["quarter"].to_numpy(),
            )
            if poss is not None:
                yield poss


def _forward_fill(a: np.ndarray) -> np.ndarray:
    out = a.copy()
    last = np.nan
    for i in range(len(out)):
        if np.isnan(out[i]):
            out[i] = last
        else:
            last = out[i]
    return out


def _infer_offense(ball, pos, pid_team) -> int:
    """Offense = the team whose player is nearest the ball on the most frames."""
    n = ball.shape[0]
    votes = Counter()
    for i in range(n):
        if np.isnan(ball[i, 0]):
            continue
        best_pid, best_d = None, np.inf
        for pid, p in pos.items():
            if np.isnan(p[i, 0]):
                continue
            d = np.hypot(p[i, 0] - ball[i, 0], p[i, 1] - ball[i, 1])
            if d < best_d:
                best_d, best_pid = d, pid
        if best_pid is not None and best_d < HANDLER_MAX_DIST + 2:
            votes[pid_team[best_pid]] += 1
    return votes.most_common(1)[0][0] if votes else next(iter(set(pid_team.values())))


def _assign_handler(i, ball, ball_vel, pos, vel, player_ids, offense_team_id, pid_team):
    """Raw (unsmoothed) handler for frame i, or None if ball is in flight."""
    if np.isnan(ball[i, 0]) or ball[i, 2] >= HANDLER_MAX_BALL_Z:
        return None
    bvx, bvy = ball_vel[i]
    best_pid, best_d = None, np.inf
    for pid in player_ids:
        if pid_team[pid] != offense_team_id:
            continue  # only the offense handles the ball on offense
        p = pos[pid][i]
        if np.isnan(p[0]):
            continue
        d = np.hypot(p[0] - ball[i, 0], p[1] - ball[i, 1])
        if d >= HANDLER_MAX_DIST:
            continue
        v = vel[pid][i]
        if np.hypot(v[0] - bvx, v[1] - bvy) >= HANDLER_MAX_VEL_DIFF:
            continue
        if d < best_d:
            best_d, best_pid = d, pid
    return best_pid


def _smooth_handler(raw: list[int | None]) -> list[int | None]:
    """A handler must hold >=HANDLER_MIN_HOLD consecutive frames to count (2.4)."""
    out: list[int | None] = list(raw)
    n = len(raw)
    i = 0
    while i < n:
        j = i
        while j < n and raw[j] == raw[i]:
            j += 1
        if raw[i] is not None and (j - i) < HANDLER_MIN_HOLD:
            for k in range(i, j):
                out[k] = None  # too brief -> treat as flicker / in flight
        i = j
    return out


def _build_possession(
    poss_id, lo, hi, game_clock, shot_clock, ball, ball_vel,
    pos, vel, player_ids, pid_team, offense_team_id, quarters,
) -> Possession | None:
    raw_handlers = [
        _assign_handler(i, ball, ball_vel, pos, vel, player_ids, offense_team_id, pid_team)
        for i in range(lo, hi)
    ]
    handlers = _smooth_handler(raw_handlers)

    frames = []
    for local_idx, i in enumerate(range(lo, hi)):
        players = []
        for pid in player_ids:
            p = pos[pid][i]
            if np.isnan(p[0]):
                continue
            v = vel[pid][i]
            players.append(
                PlayerFrame(
                    player_id=pid,
                    team_id=pid_team[pid],
                    side="offense" if pid_team[pid] == offense_team_id else "defense",
                    x=float(p[0]), y=float(p[1]), vx=float(v[0]), vy=float(v[1]),
                )
            )
        handler = handlers[local_idx]
        frames.append(
            Frame(
                possession_id=poss_id,
                frame_idx=local_idx,
                quarter=int(quarters[i]),
                game_clock=float(game_clock[i]),
                shot_clock=None if np.isnan(shot_clock[i]) else float(shot_clock[i]),
                ball_x=float(ball[i, 0]), ball_y=float(ball[i, 1]), ball_z=float(ball[i, 2]),
                ball_vx=float(ball_vel[i, 0]), ball_vy=float(ball_vel[i, 1]),
                handler_player_id=handler,
                ball_in_flight=handler is None,
                players=players,
                offense_team_id=offense_team_id,
            )
        )

    if not frames:
        return None

    terminal = _classify_terminal(ball[lo:hi])
    return Possession(
        possession_id=poss_id,
        offense_team_id=offense_team_id,
        attacking_basket=court.BASKET_LEFT,
        frames=frames,
        terminal_event=terminal,
        points_scored=None,  # requires play-by-play; left null in Phase 1
    )


def _classify_terminal(ball_slice: np.ndarray) -> str:
    """Coarse terminal-event guess from ball height (no PBP available)."""
    z = ball_slice[:, 2]
    if np.nanmax(z) >= SHOT_Z_THRESHOLD:
        return "shot_attempt"
    return "unknown"


# --- Deliverable: parquet frame table (roadmap 2.4) --------------------------
def frames_to_dataframe(possessions: list[Possession]) -> pl.DataFrame:
    """The 2.4 deliverable: one row per frame, players as a struct list."""
    rows = []
    for poss in possessions:
        for fr in poss.frames:
            rows.append(
                {
                    "possession_id": fr.possession_id,
                    "frame_idx": fr.frame_idx,
                    "quarter": fr.quarter,
                    "game_clock": fr.game_clock,
                    "shot_clock": fr.shot_clock,
                    "handler_player_id": fr.handler_player_id,
                    "ball_in_flight": fr.ball_in_flight,
                    "offense_team_id": fr.offense_team_id,
                    "ball_x": fr.ball_x, "ball_y": fr.ball_y, "ball_z": fr.ball_z,
                    "terminal_event": poss.terminal_event,
                    "points_scored": poss.points_scored,
                    "players": [
                        {
                            "player_id": p.player_id, "team_id": p.team_id, "side": p.side,
                            "x": p.x, "y": p.y, "vx": p.vx, "vy": p.vy,
                        }
                        for p in fr.players
                    ],
                }
            )
    return pl.DataFrame(rows)
