"""Rationale orchestration (roadmap 5.4).

Assemble the brief, generate, validate against the hard constraints, and
regenerate on any violation. If the live model keeps violating (or is
unavailable), fall back to the deterministic template generator, which is
constraint-valid by construction — the product never shows an unvalidated,
coordinate-leaking, or number-fabricating rationale.
"""
from __future__ import annotations

from dataclasses import dataclass

from src.llm.client import RationaleGenerator, TemplateRationaleGenerator
from src.llm.context import CoachingPriors, build_context
from src.llm.schema import Rationale, validate_rationale


@dataclass
class RationaleResult:
    rationale: Rationale
    violations: list[str]        # violations on the FINAL accepted output (empty if clean)
    source: str                  # "model" | "template_fallback"
    attempts: int


def generate_rationale(
    state,
    scored_actions,
    names: dict,
    generator: RationaleGenerator | None = None,
    coaching: CoachingPriors | None = None,
    playbook: list[str] | None = None,
    max_attempts: int = 2,
) -> RationaleResult:
    ctx = build_context(state, scored_actions, names, coaching=coaching, playbook=playbook)
    gen = generator or TemplateRationaleGenerator()
    template = TemplateRationaleGenerator()

    last_violations: list[str] = []
    for attempt in range(1, max_attempts + 1):
        try:
            r = gen.generate(ctx)
        except Exception:
            break  # generator unavailable -> fall back
        last_violations = validate_rationale(r, ctx.allowed_numbers, ctx.allowed_player_ids)
        if not last_violations:
            source = "template_fallback" if isinstance(gen, TemplateRationaleGenerator) else "model"
            return RationaleResult(r, [], source, attempt)

    # Fallback: the template output is validated too (should always pass).
    r = template.generate(ctx)
    violations = validate_rationale(r, ctx.allowed_numbers, ctx.allowed_player_ids)
    return RationaleResult(r, violations, "template_fallback", max_attempts)
