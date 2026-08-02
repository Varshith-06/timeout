"""Load real 2015-16 labels and join them to SportVU tracking (roadmap 2.2/2.4).

Two public label sources accompany the tracking:
  * ``shots/shots_fixed.csv`` — every shot with make/miss, type (2PT/3PT),
    distance, zone, and the game clock — the target for the shot model (3.2a).
  * ``events/{game_id}.csv`` — full NBA play-by-play (event type, description,
    running score, game clock) — the join the roadmap uses for possession
    outcomes (2.4).

Nothing here is synthetic; this is the swap from the simulator to real data.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl

# NBA play-by-play EVENTMSGTYPE codes.
MADE_SHOT, MISSED_SHOT, FREE_THROW, REBOUND, TURNOVER = 1, 2, 3, 4, 5


def _clock_to_seconds(s: str) -> float | None:
    """'MM:SS' game clock -> seconds remaining in the period."""
    if not isinstance(s, str) or ":" not in s:
        return None
    m, sec = s.split(":")
    try:
        return int(m) * 60 + float(sec)
    except ValueError:
        return None


def load_shots(csv_path: str | Path, game_ids: list[str] | None = None) -> pl.DataFrame:
    """Load shots_fixed.csv, filtered to game_ids, with a tracking-join key.

    Columns kept: game_id, player_id, team_id, quarter, game_clock (seconds
    remaining), made (0/1), points (2/3), is_three, shot_distance.
    """
    # infer_schema_length=None scans the whole file so a late float in an
    # otherwise-int column (or an early all-null run) can't mistype the schema.
    df = pl.read_csv(csv_path, infer_schema_length=None)
    df = df.with_columns(pl.col("GAME_ID").cast(pl.Utf8).str.zfill(10))
    if game_ids:
        df = df.filter(pl.col("GAME_ID").is_in([str(g) for g in game_ids]))
    df = df.with_columns(
        (pl.col("MINUTES_REMAINING") * 60 + pl.col("SECONDS_REMAINING")).alias("game_clock"),
        pl.col("SHOT_TYPE").str.contains("3PT").cast(pl.Int64).alias("is_three"),
    )
    return df.select(
        pl.col("GAME_ID").alias("game_id"),
        pl.col("PLAYER_ID").alias("player_id"),
        pl.col("TEAM_ID").alias("team_id"),
        pl.col("PERIOD").alias("quarter"),
        "game_clock",
        pl.col("SHOT_MADE_FLAG").alias("made"),
        (pl.col("is_three") * 3 + (1 - pl.col("is_three")) * 2).alias("points"),
        "is_three",
        pl.col("SHOT_DISTANCE").alias("shot_distance"),
    )


def load_pbp(csv_path: str | Path) -> pl.DataFrame:
    """Load a game's play-by-play with a parsed game_clock and points-scored."""
    df = pl.read_csv(csv_path, infer_schema_length=None)
    clock = [ _clock_to_seconds(s) for s in df["PCTIMESTRING"].to_list() ]
    df = df.with_columns(pl.Series("game_clock", clock))
    # Points for a scoring event: made FG (3 if the description says 3PT else 2),
    # made free throw = 1. Missed shots / non-scoring events = 0.
    desc = (df["HOMEDESCRIPTION"].fill_null("") + " " + df["VISITORDESCRIPTION"].fill_null("")).to_list()
    etype = df["EVENTMSGTYPE"].to_list()
    pts = []
    for t, d in zip(etype, desc):
        if t == MADE_SHOT:
            pts.append(3 if "3PT" in d else 2)
        elif t == FREE_THROW and "MISS" not in d:
            pts.append(1)
        else:
            pts.append(0)
    return df.with_columns(pl.Series("points", pts)).select(
        pl.col("PERIOD").alias("quarter"), "game_clock",
        pl.col("EVENTMSGTYPE").alias("event_type"),
        pl.col("PLAYER1_ID").alias("player1_id"),
        pl.col("PLAYER1_TEAM_ID").alias("team_id"),
        "points",
    )


def _snap(game, quarter, gc):
    snap = game.moments.filter((pl.col("quarter") == quarter) & (pl.col("game_clock") == gc))
    players, ball, shot_clock = {}, None, None
    for row in snap.iter_rows(named=True):
        shot_clock = row["shot_clock"]
        if row["is_ball"]:
            ball = (row["x"], row["y"], row["z"])
        else:
            players[row["player_id"]] = (row["x"], row["y"], row["team_id"])
    return players, ball, shot_clock


def moment_positions(game, quarter: int, game_clock: float, tol: float = 0.15,
                     shooter_id: int | None = None, window: float = 0.8):
    """Positions of all entities at the tracking moment for a (quarter, clock).

    Returns (players: {player_id: (x, y, team_id)}, ball, shot_clock) or
    (None, None, None). The public shot clock is only 1-second granular, so when
    ``shooter_id`` is given we search a ``window``-second span and pick the frame
    where the ball is nearest the shooter — the catch/release instant — which
    makes the defender positions accurate.
    """
    span = max(tol, window) if shooter_id is not None else tol
    m = game.moments.filter(
        (pl.col("quarter") == quarter) & ((pl.col("game_clock") - game_clock).abs() <= span)
    )
    if m.height == 0:
        return None, None, None

    if shooter_id is None:
        gc = m.with_columns((pl.col("game_clock") - game_clock).abs().alias("d")).sort("d")["game_clock"][0]
        return _snap(game, quarter, gc)

    # Pick the candidate frame minimizing ball-to-shooter distance.
    best_gc, best_d = None, float("inf")
    for gc in m["game_clock"].unique().to_list():
        frame = m.filter(pl.col("game_clock") == gc)
        sh = frame.filter(pl.col("player_id") == shooter_id)
        bl = frame.filter(pl.col("is_ball"))
        if sh.height == 0 or bl.height == 0:
            continue
        d = (sh["x"][0] - bl["x"][0]) ** 2 + (sh["y"][0] - bl["y"][0]) ** 2
        if d < best_d:
            best_d, best_gc = d, gc
    if best_gc is None:
        return None, None, None
    return _snap(game, quarter, best_gc)


def pbp_possessions(pbp: pl.DataFrame) -> list[dict]:
    """Parse play-by-play into possessions (roadmap 2.4).

    Returns dicts with quarter, start_clock, end_clock (start > end, clock counts
    down), offense_team_id, and points. Possession boundaries: a made field goal
    or turnover flips the ball; a missed shot only flips on a *defensive* rebound
    (an offensive rebound continues the same possession). Free throws add to the
    current possession. This is what gives V(s) a real target and a trustworthy
    offense assignment, independent of the tracking segmentation.
    """
    out: list[dict] = []
    for q in sorted(pbp["quarter"].unique().to_list()):
        ev = pbp.filter(pl.col("quarter") == q).sort("game_clock", descending=True)
        cur_team, start, pts = None, 720.0, 0.0

        def close(clock):
            nonlocal cur_team, start, pts
            if cur_team is not None and start > clock:
                out.append({"quarter": q, "start_clock": start, "end_clock": clock,
                            "offense_team_id": cur_team, "points": pts})
            start, pts, cur_team = clock, 0.0, None

        for r in ev.iter_rows(named=True):
            t, clk, p = r["event_type"], r["game_clock"], r["points"]
            team = None if r["team_id"] is None else int(r["team_id"])
            if clk is None:
                continue
            if t in (MADE_SHOT, MISSED_SHOT, TURNOVER):
                if cur_team is None:
                    cur_team = team
                elif team is not None and team != cur_team:
                    close(clk); cur_team = team  # ball already changed hands
                pts += p
                if t in (MADE_SHOT, TURNOVER):
                    close(clk)
            elif t == FREE_THROW:
                if cur_team is None:
                    cur_team = team
                pts += p
            elif t == REBOUND:
                if team is not None and cur_team is not None and team != cur_team:
                    close(clk); cur_team = team  # defensive rebound flips possession
        close(0.0)
    return out


def possession_lookup(possessions: list[dict]):
    """Return f(quarter, game_clock) -> possession dict (or None)."""
    def f(quarter, clock):
        for p in possessions:
            if p["quarter"] == quarter and p["end_clock"] <= clock <= p["start_clock"]:
                return p
        return None
    return f


def parsed_game_paths(json_dir: str | Path) -> list[Path]:
    return sorted(Path(json_dir).glob("*.json"))
