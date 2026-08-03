"""Roster + player-stats linking for real broadcast video (accuracy layer).

Jersey OCR gives numbers; a roster maps those numbers to a team, a name, and
optional shooting stats. Supplying one fixes three things at once:

  * team assignment by *number* (not jersey colour) — kills offense/defense
    mislabels when two teams wear similar colours;
  * real player names in the overlay and rationale ("Curry", not "#30");
  * player-aware EPV — a player's real 3P%/FG% can prime the shot model instead
    of the league-average prior.

Format (``--roster roster.json``)::

    {"teams": [
       {"id": "GSW", "players": [
          {"jersey": 30, "name": "Curry",  "fg3_pct": 0.42, "fg_pct": 0.45},
          {"jersey": 22, "name": "Wiggins", "fg3_pct": 0.38}]},
       {"id": "LAL", "players": [
          {"jersey": 15, "name": "Reaves", "fg3_pct": 0.36}]}
    ]}

Only ``jersey`` is required per player; the rest are optional. No roster ->
everything falls back to the current colour clustering and ``#N`` labels.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Roster:
    team_of: dict = field(default_factory=dict)     # jersey (int) -> team id (str)
    name_of: dict = field(default_factory=dict)     # jersey -> player name
    fg3_of: dict = field(default_factory=dict)      # jersey -> 3P% (0..1)
    fg_of: dict = field(default_factory=dict)       # jersey -> FG% (0..1)

    @staticmethod
    def load(path) -> "Roster":
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        r = Roster()
        for team in d.get("teams", []):
            tid = str(team.get("id", ""))
            for p in team.get("players", []):
                j = int(p["jersey"])
                r.team_of[j] = tid
                if p.get("name"):
                    r.name_of[j] = p["name"]
                if p.get("fg3_pct") is not None:
                    r.fg3_of[j] = float(p["fg3_pct"])
                if p.get("fg_pct") is not None:
                    r.fg_of[j] = float(p["fg_pct"])
        return r

    def label(self, jersey) -> str | None:
        """Display label for a jersey: name if known, else '#N'."""
        if jersey is None:
            return None
        return self.name_of.get(int(jersey), f"#{int(jersey)}")
