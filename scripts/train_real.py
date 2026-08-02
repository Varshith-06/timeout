"""Train the shot make-probability model on REAL 2015-16 data (roadmap 3.2a).

Swaps the physics simulator for real SportVU tracking joined to real shot labels.
Splits by *game* (held-out games, never frames) so nothing leaks, trains the
LightGBM shot model with its empirical-Bayes per-player prior, and reports
calibration — the metric the roadmap cares about most (3.4).

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
from src.value.real_dataset import build_real_shot_dataset  # noqa: E402
from src.value.submodels import ShotModel  # noqa: E402

JSON_DIR = "data/raw/sportvu_json"
SHOTS_CSV = "data/raw/labels/shots_fixed.csv"


def main() -> int:
    out = Path("out/real"); out.mkdir(parents=True, exist_ok=True)
    models = Path("models/real"); models.mkdir(parents=True, exist_ok=True)

    paths = parsed_game_paths(JSON_DIR)
    n_test = max(1, len(paths) // 5)
    train_paths, test_paths = paths[:-n_test], paths[-n_test:]
    print(f"{len(paths)} real games -> {len(train_paths)} train / {len(test_paths)} test (held-out games)")

    print("Extracting real shots (train)...")
    tr = build_real_shot_dataset(JSON_DIR, SHOTS_CSV, paths=train_paths)
    print("Extracting real shots (test)...")
    te = build_real_shot_dataset(JSON_DIR, SHOTS_CSV, paths=test_paths)
    print(f"shots: train {tr.shot_X.shape[0]}  test {te.shot_X.shape[0]}  "
          f"| FG% train {tr.shot_y.mean():.3f} test {te.shot_y.mean():.3f}")

    shot = ShotModel().fit(tr.shot_X, tr.shot_y, tr.shot_player)
    shot.save(models / "shot_model.pkl")

    p = shot.predict_proba(te.shot_X, te.shot_player)
    base = np.full_like(p, tr.shot_y.mean())
    brier, brier_base = EV.brier_score(p, te.shot_y), EV.brier_score(base, te.shot_y)
    try:
        from sklearn.metrics import roc_auc_score
        auc = roc_auc_score(te.shot_y, p)
    except Exception:
        auc = float("nan")

    EV.plot_reliability({"real shot model": (p, te.shot_y)}, out / "real_shot_reliability.png")

    # Calibration by distance bucket (a coach-facing sanity check).
    dist = te.shot_X[:, 0]
    buckets = [("rim <4ft", dist < 4), ("paint 4-14", (dist >= 4) & (dist < 14)),
               ("mid 14-22", (dist >= 14) & (dist < 22)), ("three >=22", dist >= 22)]

    print("\n=== Real shot model (held-out games) ===")
    print(f"n_test={te.shot_X.shape[0]}  Brier={brier:.3f} (base {brier_base:.3f})  AUC={auc:.3f}")
    print(f"  {'zone':<12}{'n':>5}{'pred%':>8}{'actual%':>9}")
    for name, m in buckets:
        if m.sum() >= 5:
            print(f"  {name:<12}{int(m.sum()):>5}{p[m].mean()*100:>7.0f}%{te.shot_y[m].mean()*100:>8.0f}%")

    metrics = {"n_train": int(tr.shot_X.shape[0]), "n_test": int(te.shot_X.shape[0]),
               "brier": brier, "brier_baseline": brier_base, "auc": auc,
               "fg_pct_test": float(te.shot_y.mean())}
    (out / "real_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"\nModel: {models/'shot_model.pkl'}  |  artifacts: {out.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
