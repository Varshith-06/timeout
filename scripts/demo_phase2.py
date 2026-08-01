"""Phase 2 demo — replace Phase 1's placeholder scores with real expected points.

Runs the Phase 1 reasoning pipeline to get states + candidate actions, then
scores each candidate with the trained value stack (sub-models + V(s)) and
renders the top recommendation and runner-up with their EPV labels.

    python scripts/train_phase2.py      # once, to produce models/phase2/
    python scripts/demo_phase2.py --seed 5

The overlay now shows *expected points*, not uniform placeholders — the Phase 1
week-3 artifact upgraded with the Phase 2 value model.
"""
from __future__ import annotations

import argparse
import io
import sys
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.ingest.possessions import iter_possessions  # noqa: E402
from src.ingest.sportvu import parse_game  # noqa: E402
from src.ingest.synthetic import generate_game  # noqa: E402
from src.render.court_renderer import save_state_png  # noqa: E402
from src.state.schema import build_states, roster_jersey_map  # noqa: E402
from src.value.actions import enumerate_actions  # noqa: E402
from src.value.scoring import score_actions, scores_by_id  # noqa: E402
from src.value.state_value import ValueModel  # noqa: E402
from src.value.submodels import SubModels  # noqa: E402


def pick_decision_frame(states):
    for i, s in enumerate(states):
        if s.handler is not None and len(enumerate_actions(s)) >= 3:
            return i
    return len(states) // 3


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Phase 2 EPV recommendation demo")
    ap.add_argument("--game", type=str, default=None)
    ap.add_argument("--models", type=str, default="models/phase2")
    ap.add_argument("--out", type=str, default="out/phase2")
    ap.add_argument("--seed", type=int, default=5)
    ap.add_argument("--possessions", type=int, default=2)
    args = ap.parse_args(argv)

    models = Path(args.models)
    if not (models / "value.pt").exists():
        print(f"No trained models in {models}. Run: python scripts/train_phase2.py")
        return 1

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    submodels = SubModels.load(models)
    value_model = ValueModel.load(models / "value.pt")

    if args.game:
        game = parse_game(args.game)
    else:
        game = parse_game(generate_game(n_possessions=args.possessions, attack="mixed", seed=args.seed))
    jersey = roster_jersey_map(game)

    for pi, poss in enumerate(iter_possessions(game)):
        states = build_states(poss, jersey)
        fi = pick_decision_frame(states)
        s = states[fi]
        actions = enumerate_actions(s)
        scored = score_actions(s, actions, submodels, value_model)

        print("=" * 74)
        h = s.handler
        print(f"POSSESSION {pi} frame {fi}: handler #{h.jersey} zone={h.zone} "
              f"dist_rim={h.dist_to_rim:.1f} pressure={h.defender_pressure:.2f}")
        print(f"  {'rank':<5}{'action':<13}{'EPV':>7}{'P(success)':>12}{'V_next':>8}")
        for rank, sc in enumerate(scored):
            v = f"{sc.v_next:.2f}" if sc.v_next is not None else "  -"
            tgt = f" ->#{_jersey(s, sc.action.target)}" if sc.action.target else \
                  (f" {sc.action.direction}" if sc.action.direction else "")
            print(f"  {rank+1:<5}{sc.action.action + tgt:<13}{sc.q:>7.3f}{sc.success_prob:>12.2f}{v:>8}")

        best = scored[0]
        png = out / f"demo_possession_{pi}_epv.png"
        save_state_png(
            s, png, actions=actions, scores=scores_by_id(scored),
            title=f"Possession {pi}: top = {best.action.action} (EPV {best.q:.2f})",
        )
        print(f"  -> {png}")

    print(f"\nEPV overlays written to {out.resolve()}")
    return 0


def _jersey(state, player_id):
    for p in state.players:
        if p.player_id == player_id:
            return p.jersey if p.jersey is not None else player_id
    return player_id


if __name__ == "__main__":
    raise SystemExit(main())
