"""Tests for real-data label parsing (no downloads needed — inline CSVs)."""
import numpy as np

from src.ingest.real_data import _clock_to_seconds, load_pbp, load_shots


def test_clock_to_seconds():
    assert _clock_to_seconds("3:35") == 215
    assert _clock_to_seconds("11:41") == 701
    assert _clock_to_seconds("0:04.5") == 4.5
    assert _clock_to_seconds(None) is None
    assert _clock_to_seconds("bad") is None


SHOTS_CSV = """ACTION_TYPE,GAME_ID,MINUTES_REMAINING,SECONDS_REMAINING,PERIOD,PLAYER_ID,TEAM_ID,SHOT_MADE_FLAG,SHOT_DISTANCE,SHOT_TYPE
Jump Shot,0021500490,3,35,1,2547,1610612737,0,15,2PT Field Goal
Pullup,0021500490,8,38,2,201609,1610612737,1,25,3PT Field Goal
Dunk,0021500491,0,10,4,203500,1610612765,1,1,2PT Field Goal
"""


def test_load_shots(tmp_path):
    p = tmp_path / "shots.csv"
    p.write_text(SHOTS_CSV, encoding="utf-8")
    df = load_shots(p, game_ids=["0021500490"])
    assert df.height == 2                       # filtered to one game
    row = df.filter(df["is_three"] == 1).row(0, named=True)
    assert row["points"] == 3 and row["game_clock"] == 8 * 60 + 38
    two = df.filter(df["is_three"] == 0).row(0, named=True)
    assert two["points"] == 2 and two["game_clock"] == 3 * 60 + 35


PBP_CSV = """GAME_ID,EVENTMSGTYPE,PERIOD,PCTIMESTRING,HOMEDESCRIPTION,VISITORDESCRIPTION,PLAYER1_ID,PLAYER1_TEAM_ID
0021500490,1,1,11:41,Horford 3PT Jump Shot,,201143,1610612737
0021500490,1,1,11:20,Drummond Layup,,203083,1610612765
0021500490,2,1,11:00,MISS Horford Jumper,,201143,1610612737
0021500490,3,1,10:50,Horford Free Throw 1 of 2,,201143,1610612737
0021500490,3,1,10:49,MISS Horford Free Throw 2 of 2,,201143,1610612737
0021500490,5,1,10:30,Horford Turnover,,201143,1610612737
"""


def test_load_pbp_points(tmp_path):
    p = tmp_path / "pbp.csv"
    p.write_text(PBP_CSV, encoding="utf-8")
    df = load_pbp(p)
    pts = df["points"].to_list()
    assert pts == [3, 2, 0, 1, 0, 0]            # 3PT made, layup, miss, made FT, missed FT, turnover
    assert df["game_clock"].to_list()[0] == 11 * 60 + 41
