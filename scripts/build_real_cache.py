"""Build the real-data training set in memory-safe chunks (roadmap 3.2).

polars' Arrow arena does not return each parsed game's memory to the OS, so
building many games in one process slowly grows RAM until it's killed. This
script processes a *slice* of games in a fresh process and writes compact numpy
arrays to disk; run it once per chunk (each short and light), then
``train_real.py --from-cache`` concatenates the chunks and trains.

    python scripts/build_real_cache.py --start 0  --count 12
    python scripts/build_real_cache.py --start 12 --count 12
    ...

The V(s) arrays store *raw* player-ids; the trainer builds one global vocab
across all chunks and encodes them at train time.
"""
from __future__ import annotations

import argparse
import io
import sys
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace", line_buffering=True)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from src.ingest.real_data import parsed_game_paths  # noqa: E402
from src.value.real_dataset import build_real_dataset  # noqa: E402
from src.value.state_value import build_vs_arrays  # noqa: E402

JSON_DIR = "data/raw/sportvu_json"
SHOTS_CSV = "data/raw/labels/shots_fixed.csv"
PBP_DIR = "data/raw/labels"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=int, required=True)
    ap.add_argument("--count", type=int, default=12)
    ap.add_argument("--out-dir", type=str, default="data/cache")
    ap.add_argument("--max-possessions", type=int, default=4000)
    args = ap.parse_args()

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    paths = parsed_game_paths(JSON_DIR)[args.start:args.start + args.count]
    if not paths:
        print("no games in this slice"); return 0
    print(f"chunk games {args.start}..{args.start + len(paths)}")

    ds = build_real_dataset(JSON_DIR, SHOTS_CSV, PBP_DIR, paths=paths,
                            max_possessions=args.max_possessions)
    va = build_vs_arrays(ds.possessions, vocab=None)  # raw player-ids

    np.savez_compressed(
        out / f"chunk_{args.start:03d}.npz",
        shot_X=ds.shot_X, shot_y=ds.shot_y, shot_points=ds.shot_points, shot_player=ds.shot_player,
        pass_X=ds.pass_X, pass_y=ds.pass_y, drive_X=ds.drive_X, drive_y=ds.drive_y,
        v_ents=va["ents"], v_pidx=va["pidx"], v_mask=va["mask"],
        v_glob=va["globals"], v_y=va["y"], v_w=va["w"],
    )
    print(f"  shots {ds.shot_X.shape[0]} passes {ds.pass_X.shape[0]} "
          f"drives {ds.drive_X.shape[0]} V-states {len(va['y'])} -> chunk_{args.start:03d}.npz")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
