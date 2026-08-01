"""Train the Phase 2 value stack and run the roadmap 3.4 evaluation.

    python scripts/train_phase2.py --possessions 6000 --epochs 60

Trains the three sub-models and V(s) on simulated possessions, saves everything
under models/phase2/, and writes evaluation artifacts to out/phase2/:
  * reliability.png     — sub-model calibration (3.4)
  * epv_trajectories.png — V(s) through possessions (3.4)
  * metrics.json        — Brier scores + the week-8 gate proxy (3.4/3.5)
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

from src.value import evaluate as EV  # noqa: E402
from src.value.features import PlayerVocab  # noqa: E402
from src.value.simulation import build_dataset  # noqa: E402
from src.value.state_value import train_value_model  # noqa: E402
from src.value.submodels import train_submodels  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Train Phase 2 value stack")
    ap.add_argument("--possessions", type=int, default=6000)
    ap.add_argument("--val-possessions", type=int, default=1500)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--models", type=str, default="models/phase2")
    ap.add_argument("--out", type=str, default="out/phase2")
    args = ap.parse_args(argv)

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    models = Path(args.models); models.mkdir(parents=True, exist_ok=True)

    print(f"Simulating {args.possessions} train / {args.val_possessions} val possessions...")
    tr = build_dataset(n_possessions=args.possessions, seed=args.seed)
    va = build_dataset(n_possessions=args.val_possessions, seed=args.seed + 777)

    print("Training sub-models (shot / pass / drive)...")
    submodels = train_submodels(tr)
    submodels.save(models)

    print(f"Training V(s) for {args.epochs} epochs (with noise augmentation)...")
    vocab = PlayerVocab(tr.player_ids)
    value_model, hist = train_value_model(
        tr.possessions, vocab, augment=True, epochs=args.epochs,
        val_possessions=va.possessions, verbose=True, seed=args.seed + 1,
    )
    value_model.save(models / "value.pt")

    # --- Evaluation (3.4) ----------------------------------------------------
    print("\nEvaluating...")
    metrics = {}

    shot_p = submodels.shot.predict_proba(va.shot_X, va.shot_player)
    pass_p = submodels.pass_.predict_proba(va.pass_X)
    drive_p = submodels.drive.predict_proba(va.drive_X)
    metrics["brier"] = {
        "shot": EV.brier_score(shot_p, va.shot_y),
        "shot_baseline": EV.brier_score(np.full_like(shot_p, tr.shot_y.mean()), va.shot_y),
        "pass": EV.brier_score(pass_p, va.pass_y),
        "pass_baseline": EV.brier_score(np.full_like(pass_p, tr.pass_y.mean()), va.pass_y),
        "drive": EV.brier_score(drive_p, va.drive_y),
        "drive_baseline": EV.brier_score(np.full_like(drive_p, tr.drive_y.mean()), va.drive_y),
    }
    EV.plot_reliability(
        {"shot": (shot_p, va.shot_y), "pass": (pass_p, va.pass_y), "drive": (drive_p, va.drive_y)},
        out / "reliability.png",
    )
    EV.plot_epv_trajectories(va.possessions, value_model, out / "epv_trajectories.png", n=10)

    print("Running ordering / week-8 gate proxy...")
    metrics["recommendation"] = EV.ordering_eval(
        va.possessions[:800], submodels, value_model, va.player_skill
    )
    metrics["value_history"] = {"final_train_mse": hist["train_mse"][-1],
                                "final_val_mse": hist["val_mse"][-1] if hist["val_mse"] else None}

    (out / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    # --- Report --------------------------------------------------------------
    b = metrics["brier"]
    print("\n=== Phase 2 evaluation ===")
    print(f"Brier  shot {b['shot']:.3f} (base {b['shot_baseline']:.3f}) | "
          f"pass {b['pass']:.3f} (base {b['pass_baseline']:.3f}) | "
          f"drive {b['drive']:.3f} (base {b['drive_baseline']:.3f})")
    r = metrics["recommendation"]
    print(f"Recommendation quality (n={r['n']} decision states):")
    print(f"  {'policy':<16}{'right':>8}{'r-or-def':>10}{'wrong':>8}{'mean_regret':>13}")
    for k in ["model", "pass_most_open", "always_shoot", "random"]:
        m = r[k]
        print(f"  {k:<16}{m['right_rate']:>7.0%}{m['right_or_defensible']:>10.0%}"
              f"{m['wrong_rate']:>8.0%}{m['mean_regret']:>12.3f}")
    mdl = r["model"]
    best_base = min(r[k]["mean_regret"] for k in ["pass_most_open", "always_shoot", "random"])
    print(f"  -> model mean regret {mdl['mean_regret']:.3f} vs best baseline {best_base:.3f} "
          f"({'beats baseline' if mdl['mean_regret'] < best_base else 'does NOT beat baseline'})")
    gate = mdl["right_or_defensible"] >= 0.70 and mdl["wrong_rate"] < 0.10
    print(f"  Week-8 gate (>=70% right-or-defensible, <10% wrong): {'PASS' if gate else 'not met'}")
    print(f"\nArtifacts: {out.resolve()}\nModels: {models.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
