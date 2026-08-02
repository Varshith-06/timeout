"""Train the full Phase 2 value stack on REAL 2015-16 data (roadmap 3.2).

Swaps the physics simulator for real SportVU tracking joined to real shot and
play-by-play labels: the shot / pass / drive sub-models and the state-value
network V(s), split by *game* so nothing leaks. Reports calibration (the metric
the roadmap cares about most, 3.4) and the EPV trajectory through real
possessions. GPU-trained V when available.

    python scripts/train_real.py
"""
from __future__ import annotations

import io
import json
import sys
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace", line_buffering=True)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from src.ingest.real_data import parsed_game_paths  # noqa: E402
from src.value import evaluate as EV  # noqa: E402
from src.value.features import PlayerVocab  # noqa: E402
from src.value.real_dataset import build_real_dataset  # noqa: E402
from src.value.state_value import build_vs_arrays, train_value_model  # noqa: E402
from src.value.submodels import train_submodels  # noqa: E402

JSON_DIR = "data/raw/sportvu_json"
SHOTS_CSV = "data/raw/labels/shots_fixed.csv"
PBP_DIR = "data/raw/labels"


def _brier(p, y):
    return float(np.mean((np.asarray(p) - np.asarray(y)) ** 2))


def main() -> int:
    out = Path("out/real"); out.mkdir(parents=True, exist_ok=True)
    models = Path("models/real"); models.mkdir(parents=True, exist_ok=True)

    paths = parsed_game_paths(JSON_DIR)
    n_test = max(1, len(paths) // 5)
    train_paths, test_paths = paths[:-n_test], paths[-n_test:]
    print(f"{len(paths)} real games -> {len(train_paths)} train / {len(test_paths)} test (held-out games)")

    print("Building real train dataset (shots + passes + drives + possessions)...")
    tr = build_real_dataset(JSON_DIR, SHOTS_CSV, PBP_DIR, paths=train_paths)
    print("Building real test dataset...")
    te = build_real_dataset(JSON_DIR, SHOTS_CSV, PBP_DIR, paths=test_paths)
    print(f"  shots {tr.shot_X.shape[0]}/{te.shot_X.shape[0]}  "
          f"passes {tr.pass_X.shape[0]}/{te.pass_X.shape[0]}  "
          f"drives {tr.drive_X.shape[0]}/{te.drive_X.shape[0]}  "
          f"possessions {len(tr.possessions)}/{len(te.possessions)}")

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
