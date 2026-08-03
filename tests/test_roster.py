"""Tests for roster linking (jersey -> team/name/stats)."""
import json

from src.perception.roster import Roster


def test_roster_load_and_label(tmp_path):
    p = tmp_path / "roster.json"
    p.write_text(json.dumps({"teams": [
        {"id": "GSW", "players": [{"jersey": 30, "name": "Curry", "fg3_pct": 0.42},
                                  {"jersey": 22, "name": "Wiggins"}]},
        {"id": "LAL", "players": [{"jersey": 15, "name": "Reaves", "fg3_pct": 0.36}]},
    ]}))
    r = Roster.load(p)
    assert r.team_of == {30: "GSW", 22: "GSW", 15: "LAL"}
    assert r.name_of[30] == "Curry"
    assert r.fg3_of[30] == 0.42 and 22 not in r.fg3_of
    assert r.label(30) == "Curry"       # name when known
    assert r.label(99) == "#99"         # unknown -> number
    assert r.label(None) is None
