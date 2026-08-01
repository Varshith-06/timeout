"""Phase 4-5 demo: integration, conformance, app, and the LLM rationale.

Shows the roadmap 5.1/5.4/5.5 pieces end to end:

  1. Conformance (5.1): a SportVU state and a CV state pass the SAME validator
     and run through the SAME value model.
  2. App (5.5): PlayAnalyzer scores the paused state within the latency budget,
     caches by (game_id, timestamp), and answers "why not X?".
  3. Rationale (5.4): the LLM prose rationale as a second click, under the strict
     no-coordinates / numbers-echoed schema.

    python scripts/train_phase2.py     # once
    python scripts/demo_phase4.py --seed 5
"""
from __future__ import annotations

import argparse
import io
import json
import sys
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace", line_buffering=True)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.ingest.possessions import iter_possessions  # noqa: E402
from src.ingest.sportvu import parse_game  # noqa: E402
from src.ingest.synthetic import generate_game  # noqa: E402
from src.state.schema import build_states, roster_jersey_map  # noqa: E402
from src.state.validate import validate_state  # noqa: E402
from src.value.actions import enumerate_actions  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Phase 4-5 integration demo")
    ap.add_argument("--seed", type=int, default=5)
    ap.add_argument("--models", type=str, default="models/phase2")
    ap.add_argument("--use-claude", action="store_true",
                    help="Use the live Claude generator instead of the offline template")
    args = ap.parse_args(argv)

    models = Path(args.models)
    if not (models / "value.pt").exists():
        print(f"No trained models in {models}. Run: python scripts/train_phase2.py")
        return 1

    from src.value.state_value import ValueModel
    from src.value.submodels import SubModels
    submodels = SubModels.load(models)
    value_model = ValueModel.load(models / "value.pt")

    game = parse_game(generate_game(n_possessions=1, attack="left", seed=args.seed))
    poss = list(iter_possessions(game))[0]
    jersey = roster_jersey_map(game)

    # --- 1. Conformance: SportVU state and CV state, same validator ----------
    sportvu_states = build_states(poss, jersey)
    sv_state = next(s for s in sportvu_states if s.handler is not None and len(enumerate_actions(s)) >= 3)

    from src.perception.state_from_cv import (build_state_from_cv, pick_showable_frame,
                                              recover_tracking)
    from src.perception.synthetic_broadcast import generate_broadcast
    roster_rows = list(game.roster.select(["team_id", "jersey", "player_id"]).iter_rows())
    clip = generate_broadcast(poss, seed=1, cut_prob=0.0, stride=5, jersey_map=jersey)
    rec = recover_tracking(clip, roster_rows, stride=5)
    cv_state, cv_conf = build_state_from_cv(rec, pick_showable_frame(rec), roster_jersey=jersey)

    print("=== 1. Conformance (roadmap 5.1) ===")
    for label, st in [("SportVU", sv_state), ("CV", cv_state)]:
        problems = validate_state(st)
        print(f"  {label:<8} conforms={not problems} confidence={st.context.confidence:.2f}"
              + (f"  problems={problems}" if problems else ""))

    # --- 2. App: analyze (cached), latency, why-not --------------------------
    from src.app.analyzer import PlayAnalyzer
    from src.llm.context import CoachingPriors, roster_name_map
    coaching = CoachingPriors.load("configs/coaching_priors.json")
    generator = None
    if args.use_claude:
        from src.llm.client import ClaudeRationaleGenerator
        generator = ClaudeRationaleGenerator()

    analyzer = PlayAnalyzer(submodels, value_model, names=roster_name_map(game),
                            coaching=coaching, generator=generator)

    print("\n=== 2. App analyzer (roadmap 5.5) ===")
    a1 = analyzer.analyze(cv_state)
    a2 = analyzer.analyze(cv_state)  # same moment -> cache hit
    print(f"  first analyze: {a1.latency_ms:.0f} ms (within 2s budget: {a1.within_budget()})")
    print(f"  re-watch same moment: cache_hit={a2.cache_hit}, {a2.latency_ms:.0f} ms")
    print(f"  showable={a1.showable} (confidence {a1.confidence:.2f})")
    print(f"  recommendation: {a1.top.action.action} at EPV {a1.top.q:.2f}")
    # "Why not X?" for the lowest-ranked candidate.
    worst = a1.scored[-1]
    print(f"  why not {worst.action.action}? -> {analyzer.analyze(cv_state).why_not(worst.action.id)}")

    # --- 3. Rationale: the prose second click (5.4) --------------------------
    print("\n=== 3. LLM rationale (roadmap 5.4) ===")
    res = analyzer.rationale(a1, playbook=["Horns set"])
    print(f"  source={res.source}  violations={res.violations}")
    print(json.dumps(res.rationale.to_dict(), indent=2))
    if not args.use_claude:
        print("\n  (offline template generator; pass --use-claude for the live model)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
