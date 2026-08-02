"""Parse the SportVU 2015-16 tracking JSON into tidy polars frames.

Roadmap 2.2. One game is one JSON file:

    {
      "gameid": "0021500431",
      "gamedate": "2015-12-23",
      "events": [
        {"eventId": "303",
         "home":    {"teamid", "name", "abbreviation",
                     "players": [{"playerid","firstname","lastname","jersey","position"}]},
         "visitor": {...},
         "moments": [[quarter, unix_ms, game_clock, shot_clock, null,
                      [[team_id, player_id, x, y, z], ...]]]}
      ]
    }

Sampled at 25 Hz. Each moment holds 11 entries: 10 players + the ball. The ball
is team_id == -1, player_id == -1 and its z is height in feet.

Landmines handled here (all from 2.2):
  * Overlapping events -> de-duplicate moments on (quarter, game_clock).
  * Moments with no coordinate array -> dropped, never interpolated.
  * shot_clock null -> left null here; forward-filled per possession downstream.
"""
from __future__ import annotations

import gzip
import json
from dataclasses import dataclass
from pathlib import Path

import polars as pl

BALL_TEAM_ID = -1
BALL_PLAYER_ID = -1

# Explicit moment schema: some games start with the shot clock off (all-null),
# so letting polars infer the column type from the first rows can mistype
# shot_clock as Null and then fail on the first real value.
_MOMENT_SCHEMA = {
    "game_id": pl.Utf8, "event_id": pl.Utf8, "quarter": pl.Int64,
    "game_clock": pl.Float64, "shot_clock": pl.Float64, "unix_ms": pl.Int64,
    "team_id": pl.Int64, "player_id": pl.Int64, "x": pl.Float64,
    "y": pl.Float64, "z": pl.Float64, "is_ball": pl.Boolean,
}


@dataclass
class Game:
    """A parsed game: identity, rosters, and deduplicated long-format moments."""

    game_id: str
    game_date: str
    roster: pl.DataFrame   # one row per player
    moments: pl.DataFrame  # long format, one row per (moment, entity)


def load_game_json(path: str | Path) -> dict:
    """Load a game JSON, transparently handling .json, .json.gz, and dicts."""
    path = Path(path)
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8") as fh:
            return json.load(fh)
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _roster_from_team(team: dict, side: str) -> list[dict]:
    rows = []
    for p in team.get("players", []):
        rows.append(
            {
                "team_id": team.get("teamid"),
                "team_abbr": team.get("abbreviation"),
                "side": side,  # "home" or "visitor"
                "player_id": p.get("playerid"),
                "firstname": p.get("firstname"),
                "lastname": p.get("lastname"),
                "jersey": _safe_int(p.get("jersey")),
                "position": p.get("position"),
            }
        )
    return rows


def _safe_int(v):
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def extract_roster(game: dict) -> pl.DataFrame:
    """Roster for the whole game, deduped across events (players are stable)."""
    rows: list[dict] = []
    for ev in game.get("events", []):
        if "home" in ev:
            rows += _roster_from_team(ev["home"], "home")
        if "visitor" in ev:
            rows += _roster_from_team(ev["visitor"], "visitor")
    if not rows:
        return pl.DataFrame(
            schema={
                "team_id": pl.Int64, "team_abbr": pl.Utf8, "side": pl.Utf8,
                "player_id": pl.Int64, "firstname": pl.Utf8, "lastname": pl.Utf8,
                "jersey": pl.Int64, "position": pl.Utf8,
            }
        )
    return pl.DataFrame(rows).unique(subset=["player_id"], keep="first")


def extract_moments(game: dict) -> pl.DataFrame:
    """Flatten every event's moments into one deduplicated long-format frame.

    Columns: game_id, event_id, quarter, game_clock, shot_clock, unix_ms,
             team_id, player_id, x, y, z, is_ball.
    """
    game_id = game.get("gameid", "unknown")
    records: list[dict] = []
    for ev in game.get("events", []):
        event_id = ev.get("eventId")
        for moment in ev.get("moments", []):
            # moment = [quarter, unix_ms, game_clock, shot_clock, null, coords]
            if not moment or len(moment) < 6:
                continue
            quarter, unix_ms, game_clock, shot_clock = moment[0], moment[1], moment[2], moment[3]
            coords = moment[5]
            if not coords:  # moment with no coordinates -> drop (2.2)
                continue
            for ent in coords:
                if len(ent) < 5:
                    continue
                team_id, player_id, x, y, z = ent[0], ent[1], ent[2], ent[3], ent[4]
                records.append(
                    {
                        "game_id": game_id,
                        "event_id": event_id,
                        "quarter": int(quarter),
                        "game_clock": float(game_clock),
                        "shot_clock": None if shot_clock is None else float(shot_clock),
                        "unix_ms": int(unix_ms),
                        "team_id": int(team_id),
                        "player_id": int(player_id),
                        "x": float(x),
                        "y": float(y),
                        "z": float(z),
                        "is_ball": team_id == BALL_TEAM_ID,
                    }
                )

    if not records:
        return pl.DataFrame(schema=_MOMENT_SCHEMA)

    df = pl.DataFrame(records, schema=_MOMENT_SCHEMA)
    # De-duplicate overlapping events: a moment is uniquely a (quarter,
    # game_clock, player_id). Keep the first occurrence (2.2).
    df = df.unique(subset=["quarter", "game_clock", "player_id"], keep="first")
    # Stable ordering: quarter down, game_clock down (clock counts down).
    df = df.sort(["quarter", "game_clock", "player_id"], descending=[False, True, False])
    return df


def parse_game(source: str | Path | dict) -> Game:
    """Parse a game from a path or an already-loaded dict into a :class:`Game`."""
    game = source if isinstance(source, dict) else load_game_json(source)
    return Game(
        game_id=game.get("gameid", "unknown"),
        game_date=game.get("gamedate", ""),
        roster=extract_roster(game),
        moments=extract_moments(game),
    )
