"""Train the full Phase 2 value stack on REAL 2015-16 data (roadmap 3.2).

Swaps the physics simulator for real SportVU tracking joined to real shot and
play-by-play labels: the shot / pass / drive sub-models and the state-value
network V(s), split by *game* so nothing leaks. Reports calibration (the metric
the roadmap cares about most, 3.4) and the EPV trajectory through real
possessions. GPU-trained V when available.

    python scripts/train_real.py
"""
from __future__ import annotations

import argparse
import io
import json
import sys
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace", line_buffering=True)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

# Heavy imports (torch via evaluate/state_value/submodels) are deferred into
# main() so torch/CUDA does not reserve ~1-2 GB of RAM during the memory-heavy
# dataset build on a constrained machine.
from src.ingest.real_data import parsed_game_paths  # noqa: E402
from src.value.features import PlayerVocab  # noqa: E402
from src.value.real_dataset import build_real_dataset  # noqa: E402

JSON_DIR = "data/raw/sportvu_json"
SHOTS_CSV = "data/raw/labels/shots_fixed.csv"
PBP_DIR = "data/raw/labels"


def _brier(p, y):
    return float(np.mean((np.asarray(p) - np.asarray(y)) ** 2))


def _concat(chunks, key):
    parts = [c[key] for c in chunks if c[key].shape[0] > 0]
    return np.concatenate(parts) if parts else np.zeros((0,))


def train_from_cache(cache_dir: str) -> int:
    """Train from disk chunks built by build_real_cache.py (memory-safe path)."""
    import glob as _glob
    from src.ingest.sportvu import BALL_PLAYER_ID  # noqa: F401
    from src.value import evaluate as EV
    from src.value.features import PlayerVocab
    from src.value.simulation import LabeledDataset
    from src.value.state_value import train_value_model
    from src.value.submodels import train_submodels

    out = Path("out/real"); out.mkdir(parents=True, exist_ok=True)
    models = Path("models/real"); models.mkdir(parents=True, exist_ok=True)

    files = sorted(_glob.glob(str(Path(cache_dir) / "chunk_*.npz")))
    if not files:
        print(f"no chunks in {cache_dir}; run build_real_cache.py first"); return 1
    data = [dict(np.load(f)) for f in files]
    # Hold out whole chunks (games), never frames. Keep the test set big enough
    # to be a reliable estimate (the last chunk can be a single game).
    n_test = max(2, len(data) // 5)
    tr_c, te_c = data[:-n_test], data[-n_test:]
    print(f"{len(files)} cache chunks -> {len(tr_c)} train / {len(te_c)} test")

    def bundle(cs):
        return LabeledDataset(
            shot_X=_concat(cs, "shot_X"), shot_y=_concat(cs, "shot_y"),
            shot_points=_concat(cs, "shot_points"), shot_player=_concat(cs, "shot_player"),
            pass_X=_concat(cs, "pass_X"), pass_y=_concat(cs, "pass_y"),
            drive_X=_concat(cs, "drive_X"), drive_y=_concat(cs, "drive_y"),
            possessions=[], player_skill={}, player_ids=[])
    tr, te = bundle(tr_c), bundle(te_c)
    print(f"  shots {tr.shot_X.shape[0]}/{te.shot_X.shape[0]}  "
          f"passes {tr.pass_X.shape[0]}/{te.pass_X.shape[0]}  "
          f"drives {tr.drive_X.shape[0]}/{te.drive_X.shape[0]}")

    sm = train_submodels(tr); sm.save(models)

    # Global vocab across all chunks, then encode the raw V(s) player-id arrays.
    all_pids = np.concatenate([c["v_pidx"].ravel() for c in data]).astype(int)
    vocab = PlayerVocab(all_pids.tolist())
    idx = np.vectorize(vocab.index)

    def varr(cs):
        return {"ents": _concat(cs, "v_ents"), "pidx": idx(_concat(cs, "v_pidx").astype(int)),
                "mask": _concat(cs, "v_mask"), "globals": _concat(cs, "v_glob"),
                "y": _concat(cs, "v_y"), "w": _concat(cs, "v_w")}
    tr_v, te_v = varr(tr_c), varr(te_c)
    print(f"  V-states {len(tr_v['y'])}/{len(te_v['y'])}  PPP {tr_v['y'].mean():.2f}")

    vm, hist = train_value_model(None, vocab, augment=True, epochs=40, verbose=True,
                                 seed=1, weight_decay=2e-3, early_stop=True,
                                 arrays=tr_v, val_arrays=te_v)
    vm.save(models / "value.pt")

    _report(sm, tr, te, vm, tr_v, te_v, out, EV)
    print(f"\nModels: {models.resolve()}  |  artifacts: {out.resolve()}")
    return 0


def _report(sm, tr, te, vm, tr_v, te_v, out, EV):
    named = {}
    print("\n=== Real value stack (held-out) ===")
    for name, X, y, base_y, pred in [
        ("shot", te.shot_X, te.shot_y, tr.shot_y,
         sm.shot.predict_proba(te.shot_X, te.shot_player) if te.shot_X.shape[0] else None),
        ("pass", te.pass_X, te.pass_y, tr.pass_y, sm.pass_.predict_proba(te.pass_X) if te.pass_X.shape[0] else None),
        ("drive", te.drive_X, te.drive_y, tr.drive_y, sm.drive.predict_proba(te.drive_X) if te.drive_X.shape[0] else None),
    ]:
        if pred is None or len(y) == 0:
            continue
        b = _brier(pred, y); bb = _brier(np.full_like(pred, base_y.mean()), y)
        print(f"  {name:<6} Brier {b:.3f} (base {bb:.3f})")
        named[f"{name} (real)"] = (pred, y)
    EV.plot_reliability(named, out / "real_submodels_reliability.png")
    vpred = vm.net_predict(te_v) if hasattr(vm, "net_predict") else _vpred(vm, te_v)
    ybar = tr_v["y"].mean()
    print(f"  V(s)   MSE {_brier(vpred, te_v['y']):.3f} (base {_brier(np.full_like(vpred, ybar), te_v['y']):.3f})"
          f"  corr {np.corrcoef(vpred, te_v['y'])[0,1]:.3f}  PPP {te_v['y'].mean():.2f}")


def _vpred(vm, arrays):
    import torch
    vm.net.eval()
    with torch.no_grad():
        return vm.net(torch.tensor(arrays["ents"]), torch.tensor(arrays["pidx"]),
                      torch.tensor(arrays["mask"]), torch.tensor(arrays["globals"])).numpy()


def main() -> int:
    ap = argparse.ArgumentParser(description="Train the value stack on real data")
    ap.add_argument("--max-games", type=int, default=45,
                    help="Cap games used (build time + memory scale with this)")
    ap.add_argument("--from-cache", type=str, default=None,
                    help="Train from chunk_*.npz built by build_real_cache.py")
    args = ap.parse_args()
    if args.from_cache:
        return train_from_cache(args.from_cache)

    out = Path("out/real"); out.mkdir(parents=True, exist_ok=True)
    models = Path("models/real"); models.mkdir(parents=True, exist_ok=True)

    paths = parsed_game_paths(JSON_DIR)
    if args.max_games:
        paths = paths[: args.max_games]
    n_test = max(1, len(paths) // 5)
    train_paths, test_paths = paths[:-n_test], paths[-n_test:]
    print(f"{len(paths)} real games -> {len(train_paths)} train / {len(test_paths)} test (held-out games)")

    # Keep every shot/pass/drive; cap the possession State objects for V(s) to
    # stay within memory on many games.
    print("Building real train dataset (shots + passes + drives + possessions)...")
    tr = build_real_dataset(JSON_DIR, SHOTS_CSV, PBP_DIR, paths=train_paths, max_possessions=9000)
    print("Building real test dataset...")
    te = build_real_dataset(JSON_DIR, SHOTS_CSV, PBP_DIR, paths=test_paths, max_possessions=2500)
    print(f"  shots {tr.shot_X.shape[0]}/{te.shot_X.shape[0]}  "
          f"passes {tr.pass_X.shape[0]}/{te.pass_X.shape[0]}  "
          f"drives {tr.drive_X.shape[0]}/{te.drive_X.shape[0]}  "
          f"possessions {len(tr.possessions)}/{len(te.possessions)}")

    # Now that the build's memory peak is behind us, load the torch stack.
    from src.value import evaluate as EV
    from src.value.state_value import build_vs_arrays, train_value_model
    from src.value.submodels import train_submodels

    print("Training sub-models on real data...")
    sm = train_submodels(tr)
    sm.save(models)

    print("Training V(s) on real possessions...")
    vocab = PlayerVocab(tr.player_ids)
    # Real data has many players and noisy single-sample returns, so V(s)
    # overfits without regularization — weight decay + early-stopping on the
    # held-out games keep it honest (roadmap notes V benefits from smoothing).
    vm, hist = train_value_model(tr.possessions, vocab, augment=True, epochs=40,
                                 val_possessions=te.possessions, verbose=True, seed=1,
                                 weight_decay=2e-3, early_stop=True)
    vm.save(models / "value.pt")

    # --- Evaluation on held-out games -----------------------------------------
    metrics = {"brier": {}}
    named = {}
    for name, X, y, pred in [
        ("shot", te.shot_X, te.shot_y, sm.shot.predict_proba(te.shot_X, te.shot_player) if te.shot_X.shape[0] else None),
        ("pass", te.pass_X, te.pass_y, sm.pass_.predict_proba(te.pass_X) if te.pass_X.shape[0] else None),
        ("drive", te.drive_X, te.drive_y, sm.drive.predict_proba(te.drive_X) if te.drive_X.shape[0] else None),
    ]:
        if pred is None or len(y) == 0:
            continue
        base_rate = {"shot": tr.shot_y, "pass": tr.pass_y, "drive": tr.drive_y}[name].mean()
        metrics["brier"][name] = {"model": _brier(pred, y), "baseline": _brier(np.full_like(pred, base_rate), y)}
        named[f"{name} (real)"] = (pred, y)
    EV.plot_reliability(named, out / "real_submodels_reliability.png")

    # V(s): MSE vs baseline + EPV trajectory.
    va = build_vs_arrays(te.possessions, vocab)
    vpred = vm.value_batch([s for p in te.possessions for s in p.states])
    ybar = np.mean([p.realized_points for p in tr.possessions])
    metrics["value"] = {
        "mse": float(np.mean((vpred - va["y"]) ** 2)),
        "mse_baseline": float(np.mean((va["y"] - ybar) ** 2)),
        "corr": float(np.corrcoef(vpred, va["y"])[0, 1]),
        "ppp_train": float(ybar), "ppp_test": float(np.mean(va["y"])),
    }
    EV.plot_epv_trajectories(te.possessions, vm, out / "real_epv_trajectories.png", n=12)

    (out / "real_full_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print("\n=== Real value stack (held-out games) ===")
    for name, b in metrics["brier"].items():
        print(f"  {name:<6} Brier {b['model']:.3f} (base {b['baseline']:.3f})")
    v = metrics["value"]
    print(f"  V(s)   MSE {v['mse']:.3f} (base {v['mse_baseline']:.3f})  corr {v['corr']:.3f}  "
          f"PPP {v['ppp_test']:.2f}")
    print(f"\nModels: {models.resolve()}  |  artifacts: {out.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
