"""Phase 3 end-to-end demo: broadcast video -> state -> EPV recommendation.

Pipeline (all on synthetic broadcast, no footage required):

    Phase 1 court tracking --project--> synthetic broadcast (pixel detections,
    keypoints, clock, cuts) --perception--> recovered court State (same schema)
    --Phase 2 value model--> EPV-scored actions --render--> top-down radar +
    video overlay.

It prints the recovered state, the composite confidence, the measured domain gap
(position/identity/miss error), and the augmentation config that feeds back into
Phase 2 (roadmap 5.2 -> 3.3).

    python scripts/train_phase2.py     # once, for models/phase2/
    python scripts/demo_phase3.py --seed 5
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
from src.render.court_renderer import save_state_png  # noqa: E402
from src.state.schema import roster_jersey_map  # noqa: E402
from src.value.actions import enumerate_actions  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Phase 3 perception demo")
    ap.add_argument("--seed", type=int, default=5)
    ap.add_argument("--broadcast-seed", type=int, default=1)
    ap.add_argument("--cut-prob", type=float, default=0.05)
    ap.add_argument("--models", type=str, default="models/phase2")
    ap.add_argument("--out", type=str, default="out/phase3")
    args = ap.parse_args(argv)

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    from src.perception.synthetic_broadcast import generate_broadcast
    from src.perception.state_from_cv import (build_state_from_cv, pick_showable_frame,
                                              recover_tracking)
    from src.perception.evaluate import measure_domain_gap

    # 1. Phase 1 court tracking -> synthetic broadcast.
    game = parse_game(generate_game(n_possessions=1, attack="mixed", seed=args.seed))
    poss = list(iter_possessions(game))[0]
    jersey = roster_jersey_map(game)
    roster_rows = list(game.roster.select(["team_id", "jersey", "player_id"]).iter_rows())
    clip = generate_broadcast(poss, seed=args.broadcast_seed, cut_prob=args.cut_prob,
                              stride=5, jersey_map=jersey)
    print(f"Synthetic broadcast: {len(clip.frames)} frames "
          f"(true offense team {poss.offense_team_id})")

    # 2. Perception -> recovered court tracking.
    rec = recover_tracking(clip, roster_rows, stride=5)
    d = rec.diagnostics
    print(f"Perception: homography valid {d['homog_valid_rate']:.0%}, "
          f"mean {d['mean_players']:.1f} players, ball recall {d['ball_recall']:.0%}, "
          f"{d['n_tracklets']} tracklets")

    # 3. Pick a showable frame and build the state.
    fi = pick_showable_frame(rec)
    if fi is None:
        print("No frame passed the confidence gate — perception too degraded to recommend.")
        return 0
    state, conf = build_state_from_cv(rec, fi, roster_jersey=jersey)
    h = state.handler
    print(f"\nDecision frame {fi}: confidence {conf:.2f}, {state.context.n_players_observed} players")
    print(f"  handler #{h.jersey} zone={h.zone} dist_rim={h.dist_to_rim:.1f} "
          f"pressure={h.defender_pressure:.2f}")

    # 4. Phase 2 EPV scoring (if models are present).
    actions = enumerate_actions(state)
    scored = None
    models = Path(args.models)
    if (models / "value.pt").exists() and actions:
        from src.value.scoring import score_actions, scores_by_id
        from src.value.state_value import ValueModel
        from src.value.submodels import SubModels
        submodels = SubModels.load(models)
        value_model = ValueModel.load(models / "value.pt")
        scored = score_actions(state, actions, submodels, value_model)
        print("  top recommendations (EPV):")
        for sc in scored[:4]:
            tgt = f"->#{_jersey(state, sc.action.target)}" if sc.action.target else ""
            print(f"    {sc.action.action}{tgt:>6}  EPV {sc.q:.3f}  P {sc.success_prob:.2f}")
    else:
        print("  (no Phase 2 models found — run scripts/train_phase2.py for EPV scores)")

    # 5. Renders: top-down radar + video overlay.
    radar = out / f"phase3_radar_seed{args.seed}.png"
    save_state_png(state, radar, actions=actions,
                   scores=scores_by_id(scored) if scored else None,
                   title=f"CV-recovered state (confidence {conf:.2f})")
    print(f"  -> {radar}")

    from src.perception.state_from_cv import frame_player_pixels
    from src.perception.video_overlay import render_video_overlay
    H = rec.homographies[fi]
    if H is not None:
        player_feet, ball_px = frame_player_pixels(rec, clip, fi)
        overlay = out / f"phase3_overlay_seed{args.seed}.png"
        render_video_overlay(clip.frames[fi], state, scored, H, player_feet, ball_px, overlay)
        print(f"  -> {overlay}")

    # 6. Domain gap -> Phase 2 augmentation feedback (5.2 -> 3.3).
    gap = measure_domain_gap(rec, clip)
    print("\nDomain gap (paired vs ground truth):")
    print(f"  position error median {gap['position_error_ft_median']:.2f} ft "
          f"(mean {gap['position_error_ft_mean']:.2f}, p90 {gap['position_error_ft_p90']:.2f})")
    print(f"  player-miss rate {gap['player_miss_rate']:.0%}, "
          f"identity-error rate {gap['identity_error_rate']:.0%}")
    print(f"  -> recommended Phase 2 augmentation: {gap['augment_recommendation']}")
    (out / f"domain_gap_seed{args.seed}.json").write_text(json.dumps(gap, indent=2), encoding="utf-8")

    print(f"\nArtifacts written to {out.resolve()}")
    return 0


def _jersey(state, player_id):
    for p in state.players:
        if p.player_id == player_id:
            return p.jersey if p.jersey is not None else player_id
    return player_id


if __name__ == "__main__":
    raise SystemExit(main())
