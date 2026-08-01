"""State-schema conformance validation (roadmap 5.1).

The whole architecture rests on one invariant: the state produced from SportVU
(Phase 1) and the state produced from broadcast CV (Phase 3) are the *same
object*. This validator is the executable form of that contract — run both
sources through it (and through the same value model) and they must agree in
shape. The only field that legitimately differs is ``context.confidence``: 1.0
for perfect tracking, < 1.0 for perception.
"""
from __future__ import annotations

from src.state.schema import State

# Required top-level keys and the sub-keys each must contain.
REQUIRED = {
    "timestamp": {"quarter", "game_clock", "shot_clock"},
    "ball": {"x", "y", "z", "vx", "vy", "in_flight"},
    "context": {"n_players_observed", "spacing_area_sqft", "defense_scheme",
                "active_screen", "confidence"},
}
PLAYER_FIELDS = {
    "player_id", "team_id", "side", "x", "y", "vx", "vy", "speed",
    "orientation_deg", "has_ball", "dist_to_rim", "angle_to_rim_deg", "zone",
    "nearest_defender", "defender_pressure", "seconds_since_touch",
}


def validate_state(state: State) -> list[str]:
    """Return a list of conformance violations; empty means the state conforms."""
    problems: list[str] = []
    d = state.to_dict()

    for key, subkeys in REQUIRED.items():
        if key not in d:
            problems.append(f"missing top-level key: {key}")
            continue
        missing = subkeys - set(d[key].keys())
        if missing:
            problems.append(f"{key} missing sub-keys: {sorted(missing)}")

    if not d.get("players"):
        problems.append("players list is empty")
    else:
        for i, p in enumerate(d["players"]):
            missing = PLAYER_FIELDS - set(p.keys())
            if missing:
                problems.append(f"player[{i}] missing fields: {sorted(missing)}")
                break  # one report is enough

    # Semantic checks shared by both sources.
    conf = d["context"]["confidence"]
    if not (0.0 <= conf <= 1.0):
        problems.append(f"confidence out of range: {conf}")
    sides = {p["side"] for p in d["players"]}
    if not sides <= {"offense", "defense"}:
        problems.append(f"unexpected side values: {sides}")
    if d["context"]["defense_scheme"] not in {"man", "zone", "unknown"}:
        problems.append(f"unexpected defense_scheme: {d['context']['defense_scheme']}")

    return problems


def is_conformant(state: State) -> bool:
    return not validate_state(state)
