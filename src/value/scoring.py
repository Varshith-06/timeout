"""Score candidate actions in expected points (roadmap 3.1).

    Q(s, a) = P(success | s, a) * V(s'_a) + (1 - P(success | s, a)) * V_turnover

For SHOOT the special case collapses to P(make) * point_value. For deterministic
repositioning (SCREEN_WITH, RESET) there is no success model, so Q = V(s').

This is the function that replaces the Phase 1 placeholder uniform scores with
calibrated expected points.
"""
from __future__ import annotations

from dataclasses import dataclass

from src.value import features as F
from src.value.actions import Action
from src.value.state_value import ValueModel
from src.value.submodels import SubModels
from src.value.transition import apply_action

# Value conceded on a turnover: ~0 plus the (negative) transition value you give
# up. Kept explicit and tunable (3.1).
V_TURNOVER = -0.10


@dataclass
class ScoredAction:
    action: Action
    q: float                 # expected points
    success_prob: float      # P(make) for SHOOT, P(complete)/P(reach) otherwise
    v_next: float | None     # V(s'_a); None for SHOOT
    kind: str

    def to_dict(self) -> dict:
        d = self.action.to_dict()
        d.update(q=round(self.q, 3), success_prob=round(self.success_prob, 3))
        if self.v_next is not None:
            d["v_next"] = round(self.v_next, 3)
        return d


def _shoot_q(state, handler, submodels: SubModels):
    feats = F.shot_features(state, handler)  # prior column overwritten by the model
    p = float(submodels.shot.predict_proba(feats, [handler.player_id])[0])
    points = 3 if "three" in handler.zone else 2
    return p * points, p


def leaf_value(state, submodels: SubModels, value_model: ValueModel, allow_shot: bool = True) -> float:
    """Value of a resulting state s' for the one-step lookahead.

    A blurry on-policy V(s') alone cannot tell an open receiver from a covered
    one, so it systematically undervalues passes. For actions that hand the ball
    to a player with a *fresh scoring look* (PASS to a catch, DRIVE to the rim),
    we take the better of (a) that handler's calibrated immediate shot and (b)
    the learned continuation value V(s'). For a RESET or a SCREEN there is no
    fresh look — the handler just relocated against his man — so we use V(s')
    alone (allow_shot=False); otherwise the immediate-shot leaf inflates them.
    Either way the lookahead stays one step deep (3.1).
    """
    v_learned = value_model.value(state)
    handler = state.handler
    if handler is None or not allow_shot:
        return v_learned
    shot_q, _ = _shoot_q(state, handler, submodels)
    return max(v_learned, shot_q)


def score_action(state, action: Action, submodels: SubModels, value_model: ValueModel) -> ScoredAction:
    handler = state.handler
    if action.action == "SHOOT":
        q, p = _shoot_q(state, handler, submodels)
        return ScoredAction(action, q, p, None, "shoot")

    if action.action == "PASS_TO":
        receiver = next(pp for pp in state.offense() if pp.player_id == action.target)
        p = float(submodels.pass_.predict_proba(F.pass_features(state, handler, receiver))[0])
        v = leaf_value(apply_action(state, action), submodels, value_model)
        return ScoredAction(action, p * v + (1 - p) * V_TURNOVER, p, v, "pass")

    if action.action == "DRIVE":
        p = float(submodels.drive.predict_proba(F.drive_features(state, handler, action.direction))[0])
        v = leaf_value(apply_action(state, action), submodels, value_model)
        return ScoredAction(action, p * v + (1 - p) * V_TURNOVER, p, v, "drive")

    # SCREEN_WITH / RESET: deterministic reposition, no fresh shot -> continuation V.
    v = leaf_value(apply_action(state, action), submodels, value_model, allow_shot=False)
    return ScoredAction(action, v, 1.0, v, action.action.lower())


def score_actions(state, actions, submodels: SubModels, value_model: ValueModel) -> list[ScoredAction]:
    """Score and rank a candidate list, best expected-points first."""
    scored = [score_action(state, a, submodels, value_model) for a in actions]
    scored.sort(key=lambda s: s.q, reverse=True)
    return scored


def scores_by_id(scored: list[ScoredAction]) -> dict:
    """{action_id: q} for the renderer overlay."""
    return {s.action.id: s.q for s in scored}
