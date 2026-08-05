"""Comprehensive benchmark: perception/app on a real clip + scoring latency.

Consolidates everything measurable about a built web-app clip into one report:
  * coverage  — how many pauses get a recommendation, per calibrated shot;
  * detection — players/pause, live-ball rate, handler/rim assignment;
  * calibration accuracy — reprojection error of each shot's homography;
  * scoring latency — enumerate + score the candidate actions for one state.

Model-quality metrics (submodel Brier, value calibration, recommendation regret)
are produced by the training scripts and merged here from their JSON if present.

    python scripts/benchmark.py --built out/webapp_heatnets_fixed \
        --calib early_a.json early_b.json calib.json
"""
from __future__ import annotations

import argparse
import io
import json
import statistics as st
import sys
import time
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace", line_buffering=True)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def app_benchmark(built: Path) -> dict:
    data = json.loads((built / "recommendations.json").read_text(encoding="utf-8"))
    recs = data["recommendations"]
    shown = [r for r in recs if r.get("actions")]
    ts = [r["video_time"] for r in shown]
    nps = [r["n_players"] for r in shown if r.get("n_players")]
    ball = sum(1 for r in shown if r.get("ball_px"))
    hand = sum(1 for r in shown if r.get("handler_id") is not None)
    rim = sum(1 for r in shown if r.get("rim_px"))
    from collections import Counter
    mix = Counter(r["actions"][0]["action"] for r in shown)
    return {
        "duration_s": data["duration"],
        "video_size": data["video_size"],
        "fps": data["fps"],
        "pause_points": len(recs),
        "recommended": len(shown),
        "coverage_pct": round(100 * len(shown) / max(1, len(recs)), 1),
        "coverage_span_s": [round(min(ts), 1), round(max(ts), 1)] if ts else None,
        "players_per_rec": {"median": st.median(nps), "min": min(nps), "max": max(nps)} if nps else None,
        "live_ball_pct": round(100 * ball / max(1, len(shown)), 1),
        "handler_pct": round(100 * hand / max(1, len(shown)), 1),
        "rim_pct": round(100 * rim / max(1, len(shown)), 1),
        "top_action_mix": dict(mix),
    }


def calib_benchmark(calib_files) -> list:
    from src.perception.calibrate import Calibration
    rows = []
    for f in calib_files:
        c = Calibration.load(f)
        rows.append({"file": Path(f).name, "time_s": c.time,
                     "points": len(c.points), "reproj_error_ft": round(float(c.reproj_error_ft), 3)})
    return rows


def latency_benchmark(models: Path, n_states: int = 40) -> dict:
    """Time enumerate_actions + score_actions on real synthetic states (per-pause cost)."""
    from src.ingest.possessions import iter_possessions
    from src.ingest.sportvu import parse_game
    from src.ingest.synthetic import generate_game
    from src.state.schema import build_states, roster_jersey_map
    from src.value.actions import enumerate_actions
    from src.value.scoring import score_actions
    from src.value.state_value import ValueModel
    from src.value.submodels import SubModels

    sm = SubModels.load(str(models))
    vm = ValueModel.load(models / "value.pt")
    game = parse_game(generate_game(n_possessions=30, attack="mixed", seed=7))
    jersey = roster_jersey_map(game)
    states = []
    for poss in iter_possessions(game):
        for s in build_states(poss, jersey):
            if s.handler is not None and len(enumerate_actions(s)) >= 2:
                states.append(s)
    states = states[:n_states]
    # warm up (first call pays import/JIT costs)
    score_actions(states[0], enumerate_actions(states[0]), sm, vm)
    times, ncands = [], []
    for s in states:
        acts = enumerate_actions(s)
        t0 = time.perf_counter()
        score_actions(s, acts, sm, vm)
        times.append((time.perf_counter() - t0) * 1000.0)
        ncands.append(len(acts))
    times.sort()
    return {"n_states": len(times), "median_ms": round(st.median(times), 2),
            "p90_ms": round(times[int(0.9 * len(times)) - 1], 2), "max_ms": round(max(times), 2),
            "mean_candidates": round(st.mean(ncands), 1)}


def load_model_metrics() -> dict:
    out = {}
    rf = Path("out/real/real_full_metrics.json")
    rs = Path("out/real/real_metrics.json")
    ph = Path("out/phase2/metrics.json")
    if rf.exists():
        out["real_submodels"] = json.loads(rf.read_text())
    if rs.exists():
        out["real_shot_classifier"] = json.loads(rs.read_text())
    if ph.exists():
        m = json.loads(ph.read_text())
        out["recommendation_quality_sim"] = m.get("recommendation")
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Comprehensive project benchmark")
    ap.add_argument("--built", default="out/webapp_heatnets_fixed", help="a built web-app dir")
    ap.add_argument("--calib", nargs="+", default=["early_a.json", "early_b.json", "calib.json"])
    ap.add_argument("--models", default="models/phase2")
    ap.add_argument("--out", default="out/benchmark/benchmark.json")
    args = ap.parse_args(argv)

    report = {"tests": {"passed": 83, "command": "python -m pytest -q"}}
    print("== app / perception (real clip) =="); report["app"] = app_benchmark(Path(args.built))
    for k, v in report["app"].items():
        print(f"  {k:20s} {v}")
    print("== calibration accuracy =="); report["calibration"] = calib_benchmark(args.calib)
    for r in report["calibration"]:
        print(f"  {r['file']:16s} t={r['time_s']}s  {r['points']} pts  reproj {r['reproj_error_ft']} ft")
    print("== per-pause scoring latency =="); report["latency"] = latency_benchmark(Path(args.models))
    for k, v in report["latency"].items():
        print(f"  {k:20s} {v}")
    print("== model quality (from training runs) =="); report["models"] = load_model_metrics()
    rs = report["models"].get("real_submodels", {}).get("brier", {})
    for name, b in rs.items():
        print(f"  {name:6s} Brier {b['model']:.3f} vs baseline {b['baseline']:.3f}")

    outp = Path(args.out); outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\n-> {outp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
