"""
Pull per-event NBA video clips from the official stats API.

Each clip arrives already paired with its play-by-play event, so you get
free labels: shot/pass/rebound type, players involved, period, clock.
That pairing is the whole reason this beats scraping YouTube.

    pip install nba_api requests

Notes before you run this:
  - The videoUrls field names have shifted across nba_api versions, so this
    probes the response dict rather than hard-coding keys.
  - NBA blacklists many cloud provider IP ranges. Run it from a home
    connection, or proxy through one. On AWS/GCP you will get empty
    responses that look like "no video available" but aren't.
  - Be polite: sleep between calls. There is no published rate limit,
    which means the unpublished one is enforced by silent blocking.
"""

import json
import pathlib
import time

import requests
from nba_api.stats.endpoints import leaguegamefinder, playbyplayv3, videoeventsasset

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36",
    "Referer": "https://www.nba.com/",
}
OUT = pathlib.Path("data/video/clips")
SLEEP = 0.8


def largest_url(video_urls: dict) -> str | None:
    """Pick the highest-resolution URL without assuming field names."""
    candidates = [
        v for k, v in video_urls.items()
        if isinstance(v, str) and v.startswith("http") and v.endswith(".mp4")
    ]
    if not candidates:
        return None
    # 'lurl' (large) > 'murl' (medium) > 'surl' (small); fall back to longest key match
    for prefix in ("l", "m", "s"):
        for k, v in video_urls.items():
            if k.lower().startswith(prefix) and isinstance(v, str) and v.endswith(".mp4"):
                return v
    return candidates[0]


def clips_for_game(game_id: str, limit: int | None = None) -> int:
    pbp = playbyplayv3.PlayByPlayV3(game_id=game_id, start_period=1, end_period=4)
    events = pbp.get_data_frames()[0]

    # Only events the NBA actually hosts video for.
    events = events[events["videoAvailable"] == 1]

    # Half-court offense only, per roadmap section 0. Field goals and turnovers;
    # skip free throws, substitutions, timeouts.
    keep = events["actionType"].isin(["Made Shot", "Missed Shot", "Turnover"])
    events = events[keep]
    if limit:
        events = events.head(limit)

    game_dir = OUT / game_id
    game_dir.mkdir(parents=True, exist_ok=True)
    saved = 0

    for _, ev in events.iterrows():
        event_id = int(ev["actionNumber"])
        dest = game_dir / f"{event_id:04d}.mp4"
        if dest.exists():
            continue

        try:
            # VideoEventsAsset is the current endpoint; the old VideoEvents
            # returns an empty resultSets. Default headers only — an Origin
            # header trips NBA's bot block.
            resp = videoeventsasset.VideoEventsAsset(
                game_id=game_id, game_event_id=event_id, timeout=30
            ).get_dict()
            urls = resp["resultSets"]["Meta"]["videoUrls"]
            if not urls:
                continue
            url = largest_url(urls[0])
            if not url:
                continue

            r = requests.get(url, headers=HEADERS, timeout=30)
            r.raise_for_status()
            dest.write_bytes(r.content)

            # Save the label alongside the clip. This pairing is the point.
            (game_dir / f"{event_id:04d}.json").write_text(json.dumps({
                "game_id": game_id,
                "event_id": event_id,
                "period": int(ev["period"]),
                "clock": ev["clock"],
                "action_type": ev["actionType"],
                "sub_type": ev["subType"],
                "player_id": int(ev["personId"]) if ev["personId"] else None,
                "team_tricode": ev["teamTricode"],
                "shot_result": ev["shotResult"],
                "shot_distance": ev["shotDistance"],
                "x_legacy": ev["xLegacy"],
                "y_legacy": ev["yLegacy"],
                "description": ev["description"],
            }, indent=2))
            saved += 1

        except Exception as e:
            print(f"   event {event_id}: {e}")
        time.sleep(SLEEP)

    return saved


if __name__ == "__main__":
    # Miami Heat games, current season. Swap season for the 2015-16 paired-data
    # trick in roadmap section 5.2 — you need broadcast footage of the same
    # games you have SportVU tracking for.
    finder = leaguegamefinder.LeagueGameFinder(
        team_id_nullable=1610612748,      # MIA
        season_nullable="2025-26",
        season_type_nullable="Regular Season",
    )
    games = finder.get_data_frames()[0]["GAME_ID"].unique()
    print(f"Found {len(games)} games")

    for gid in games[:5]:
        n = clips_for_game(gid, limit=40)
        print(f"{gid}: saved {n} clips")
        time.sleep(2)
