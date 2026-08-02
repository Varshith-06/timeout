"""Download a working set of real 2015-16 SportVU games + shot/PBP labels."""
import io, sys, requests, py7zr
from pathlib import Path

N_GAMES = int(sys.argv[1]) if len(sys.argv) > 1 else 15
raw = Path("data/raw"); (raw/"sportvu_json").mkdir(parents=True, exist_ok=True)
(raw/"labels").mkdir(parents=True, exist_ok=True)
S = requests.Session(); S.headers["User-Agent"] = "nba-play-recommender/0.1"

# 1. Game .7z logs (tracking JSON).
api = "https://api.github.com/repos/linouk23/NBA-Player-Movements/contents/data/2016.NBA.Raw.SportVU.Game.Logs"
listing = S.get(api, timeout=60).json()
picked = [it for it in listing if it["name"].endswith(".7z")][:N_GAMES]
gameids = []
for it in picked:
    dst = raw/"sportvu_json"/it["name"]
    if not any((raw/"sportvu_json").glob("00215*.json")) or not dst.with_suffix("").exists():
        data = S.get(it["download_url"], timeout=180).content
        with py7zr.SevenZipFile(io.BytesIO(data), mode="r") as z:
            z.extractall(path=raw/"sportvu_json")
        print(f"  unpacked {it['name']} ({len(data)//1_000_000} MB)", flush=True)
for j in sorted((raw/"sportvu_json").glob("*.json")):
    gid = j.stem.split(".")[0] if j.stem.startswith("00215") else None
    # gameid is inside the json; read cheaply
    import json
    gid = json.loads(j.read_text(encoding="utf-8")).get("gameid")
    if gid: gameids.append(gid)
gameids = sorted(set(gameids))
print("gameids:", gameids, flush=True)

# 2. Shot labels (make/miss).
sf = raw/"labels"/"shots_fixed.csv"
if not sf.exists():
    print("downloading shots_fixed.csv ...", flush=True)
    sf.write_bytes(S.get("https://raw.githubusercontent.com/sealneaward/nba-movement-data/master/data/shots/shots_fixed.csv", timeout=300).content)

# 3. Per-game play-by-play events.
for gid in gameids:
    dst = raw/"labels"/f"events_{gid}.csv"
    if not dst.exists():
        r = S.get(f"https://raw.githubusercontent.com/sealneaward/nba-movement-data/master/data/events/{gid}.csv", timeout=120)
        if r.status_code == 200:
            dst.write_bytes(r.content); print(f"  events {gid}", flush=True)
print("DONE", len(gameids), "games", flush=True)
