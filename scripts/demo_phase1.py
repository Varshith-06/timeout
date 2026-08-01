"""Phase 1 end-to-end demo — the week-3 gate (roadmap 2.8).

Runs the full reasoning layer on synthetic tracking (no downloads required):

    SportVU JSON  ->  parse  ->  segment possessions  ->  build state schema
                  ->  enumerate candidate actions  ->  render court overlay

For each frozen possession it prints the state summary and the candidate list
with *placeholder uniform scores* (real scores arrive in Phase 2), and writes a
PNG overlay. This is exactly the artifact you sit down with a coach and ask:
"Is the candidate list complete? Is anything on it absurd?"

Usage:
    python scripts/demo_phase1.py                 # synthetic, 2 possessions
    python scripts/demo_phase1.py --game path.json  # a real SportVU game
    python scripts/demo_phase1.py --out out --seed 3
"""
from __future__ import annotations

import argparse
import io
import json
import sys
from pathlib import Path

# Ensure UTF-8 stdout on Windows consoles (polars/box glyphs, arrows).
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

# Make the repo importable when run as a script.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.ingest.possessions import frames_to_dataframe, iter_possessions  # noqa: E402
from src.ingest.sportvu import parse_game  # noqa: E402
from src.ingest.synthetic import generate_game  # noqa: E402
from src.render.court_renderer import save_state_png  # noqa: E402
from src.state.schema import build_states, roster_jersey_map  # noqa: E402
from src.value.actions import enumerate_actions  # noqa: E402


def uniform_scores(actions) -> dict:
    """Placeholder scores for the week-3 gate. Phase 2 replaces this with V(s')."""
    return {a.id: 1.0 for a in actions}


def pick_decision_frame(states):
    """First frame with a settled handler and a full candidate set."""
    for i, s in enumerate(states):
        if s.handler is not None and len(enumerate_actions(s)) >= 3:
            return i
    return len(states) // 3


def summarize_state(s) -> str:
    h = s.handler
    lines = [
        f"  possession {s.possession_id}  Q{s.timestamp['quarter']} "
        f"clock={s.timestamp['game_clock']:.1f} shot={s.timestamp['shot_clock']}",
        f"  offense_team={s.offense_team_id}  scheme={s.context.defense_scheme}  "
        f"spacing={s.context.spacing_area_sqft:.0f} sqft  players={s.context.n_players_observed}  "
        f"confidence={s.context.confidence}",
    ]
    if h is not None:
        lines.append(
            f"  HANDLER #{h.jersey} zone={h.zone} dist_to_rim={h.dist_to_rim:.1f}ft "
            f"pressure={h.defender_pressure:.2f} "
            f"nearest_def={h.nearest_defender.dist:.1f}ft@{h.nearest_defender.angle_deg:.0f}deg"
        )
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Phase 1 reasoning-layer demo")
    ap.add_argument("--game", type=str, default=None, help="Path to a real SportVU game JSON")
    ap.add_argument("--out", type=str, default="out", help="Output directory for PNGs/JSON")
    ap.add_argument("--seed", type=int, default=0, help="Synthetic seed (ignored with --game)")
    ap.add_argument("--possessions", type=int, default=2, help="Synthetic possession count")
    ap.add_argument("--max-render", type=int, default=5, help="Max possessions to render")
    args = ap.parse_args(argv)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    if args.game:
        print(f"Parsing real game: {args.game}")
        game = parse_game(args.game)
    else:
        print(f"Generating synthetic game (seed={args.seed}, {args.possessions} possessions)")
        game = parse_game(generate_game(n_possessions=args.possessions, attack="mixed", seed=args.seed))

    jersey = roster_jersey_map(game)
    possessions = list(iter_possessions(game))
    print(f"Segmented {len(possessions)} possession(s).\n")

    # Deliverable parquet (roadmap 2.4).
    df = frames_to_dataframe(possessions)
    parquet_path = out / "frames.parquet"
    df.write_parquet(parquet_path)
    print(f"Wrote frame table: {parquet_path}  ({df.height} rows)\n")

    for pi, poss in enumerate(possessions[: args.max_render]):
        states = build_states(poss, jersey)
        fi = pick_decision_frame(states)
        s = states[fi]
        actions = enumerate_actions(s)
        scores = uniform_scores(actions)

        print("=" * 78)
        print(f"POSSESSION {pi}  (decision frame {fi}/{len(states)})")
        print(summarize_state(s))
        print(f"  candidate actions ({len(actions)}):")
        for a in actions:
            extra = {k: v for k, v in a.to_dict().items() if k not in {"action", "actor", "id"}}
            print(f"    [{a.id:>3}] {a.action:<12} {extra}")

        png = out / f"possession_{pi}_frame_{fi}.png"
        save_state_png(s, png, actions=actions, scores=scores,
                       title=f"Possession {pi} — frame {fi} (placeholder scores)")
        # Also dump the state JSON for inspection.
        (out / f"possession_{pi}_state.json").write_text(
            json.dumps(s.to_dict(), indent=2), encoding="utf-8"
        )
        print(f"  -> {png}")

    print("\nWeek-3 gate artifacts written to", out.resolve())
    print("Ask a coach: (1) is the candidate list complete?  (2) is anything absurd?")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
