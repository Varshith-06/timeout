"""Assemble the LLM input for rationale generation (roadmap 5.4).

Input to the LLM: the state object, the ranked candidate list with scores, any
matched playbook sets, and the coaching-priors config. This module resolves
player_ids to names (so the model never sees or emits an id), builds a compact
structured brief, and — critically — enumerates every number the model is
allowed to state, so :func:`src.llm.schema.validate_rationale` can reject any
fabricated figure.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from src.value.scoring import ScoredAction


@dataclass
class CoachingPriors:
    """Team/staff preferences (roadmap 0: the coaching-priors config surface)."""
    team_name: str = "the offense"
    emphasis: str = ""
    avoid_zones: list[str] = field(default_factory=list)   # "we don't take that shot"
    min_epv_to_recommend: float = 0.0

    @staticmethod
    def load(path) -> "CoachingPriors":
        import json
        from pathlib import Path
        p = Path(path)
        if not p.exists():
            return CoachingPriors()
        data = json.loads(p.read_text(encoding="utf-8"))
        return CoachingPriors(**data)


@dataclass
class RationaleContext:
    brief: dict
    allowed_numbers: list[float]
    allowed_player_ids: list[int]
    names: dict


def _name(names, player_id, fallback_jersey=None):
    if player_id in names:
        return names[player_id]
    return f"#{fallback_jersey}" if fallback_jersey is not None else "a teammate"


def _describe_action(state, sc: ScoredAction, names) -> dict:
    a = sc.action
    handler = state.handler
    out = {"action": a.action, "epv": round(sc.q, 2),
           "success_prob": round(sc.success_prob, 2),
           "success_pct": int(round(sc.success_prob * 100))}
    if a.action == "SHOOT":
        out["shooter"] = _name(names, handler.player_id, handler.jersey)
        out["zone"] = handler.zone
        out["points"] = 3 if "three" in handler.zone else 2
    elif a.action == "PASS_TO":
        target = next((p for p in state.offense() if p.player_id == a.target), None)
        out["target"] = _name(names, a.target, target.jersey if target else None)
        if target is not None:
            out["target_zone"] = target.zone
            if target.nearest_defender:
                out["target_separation_ft"] = round(target.nearest_defender.dist, 1)
    elif a.action == "DRIVE":
        out["direction"] = a.direction
    return out


def build_context(state, scored_actions, names, coaching: CoachingPriors | None = None,
                  playbook: list[str] | None = None, top_k: int = 3) -> RationaleContext:
    coaching = coaching or CoachingPriors()
    handler = state.handler
    top = scored_actions[0]
    runner_up = scored_actions[1] if len(scored_actions) > 1 else None

    brief = {
        "team": coaching.team_name,
        "handler": _name(names, handler.player_id, handler.jersey),
        "handler_zone": handler.zone,
        "shot_clock": state.timestamp.get("shot_clock"),
        "defense_scheme": state.context.defense_scheme,
        "confidence": state.context.confidence,
        "recommendation": _describe_action(state, top, names),
        "alternative": _describe_action(state, runner_up, names) if runner_up else None,
        "coaching_emphasis": coaching.emphasis,
        "matched_plays": playbook or [],
    }
    # The defender who is "the reason" for the recommendation.
    if handler.nearest_defender:
        brief["handler_defender_separation_ft"] = round(handler.nearest_defender.dist, 1)

    # Numbers the model may echo: EPVs, probs (fraction + percent), separations,
    # point values, and the shot clock.
    allowed: set[float] = {2.0, 3.0}
    for sc in scored_actions[:top_k]:
        allowed.add(round(sc.q, 2))
        allowed.add(round(sc.success_prob, 2))
        allowed.add(float(int(round(sc.success_prob * 100))))
    for p in state.players:
        if p.nearest_defender:
            allowed.add(round(p.nearest_defender.dist, 1))
    if state.timestamp.get("shot_clock") is not None:
        allowed.add(round(state.timestamp["shot_clock"], 1))
        allowed.add(float(int(state.timestamp["shot_clock"])))

    allowed_ids = [p.player_id for p in state.players]
    return RationaleContext(brief=brief, allowed_numbers=sorted(allowed),
                            allowed_player_ids=allowed_ids, names=names)


def roster_name_map(game) -> dict:
    """player_id -> 'First Last' from a parsed Game's roster frame."""
    out = {}
    for pid, fn, ln in game.roster.select(["player_id", "firstname", "lastname"]).iter_rows():
        out[pid] = f"{fn} {ln}".strip()
    return out
