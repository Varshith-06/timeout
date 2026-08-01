"""Rationale generators (roadmap 5.4).

Two implementations behind one interface:

* :class:`ClaudeRationaleGenerator` — the production path. Calls Claude
  (``claude-opus-4-8``) under a strict JSON schema so the output shape is
  guaranteed. The system prompt forbids coordinates and any number not in the
  provided brief; the orchestrator still validates and regenerates on violation.

* :class:`TemplateRationaleGenerator` — the default, offline path. Deterministic
  prose assembled only from resolved names and the value model's own numbers, so
  it *cannot* fabricate a figure or leak a coordinate — the roadmap's constraints
  hold by construction. Also what the test suite runs (no network, no API key).
"""
from __future__ import annotations

import json

from src.llm.context import RationaleContext
from src.llm.schema import Rationale

MODEL = "claude-opus-4-8"

_SYSTEM = """You are a basketball assistant coach explaining one play recommendation.
You are given a structured brief with a recommendation and its numbers.

Hard rules:
- Reference players by the names in the brief. Never invent a name or use a number/ID.
- Every number you write MUST appear in the brief. Do not compute or estimate new numbers.
- Never write court coordinates (no feet, no pixels, no "(x, y)").
- Be concise: one or two sentences per field, in a coach's voice.
Return only the requested JSON."""

_SCHEMA = {
    "type": "object",
    "properties": {
        "headline": {"type": "string"},
        "rationale": {"type": "string"},
        "risk": {"type": "string"},
        "alternative": {"type": "string"},
    },
    "required": ["headline", "rationale", "risk", "alternative"],
    "additionalProperties": False,
}


class RationaleGenerator:
    def generate(self, ctx: RationaleContext) -> Rationale:  # pragma: no cover
        raise NotImplementedError


class ClaudeRationaleGenerator(RationaleGenerator):
    """Generates the rationale with Claude under a strict output schema."""

    def __init__(self, model: str = MODEL, max_tokens: int = 512, client=None):
        self.model = model
        self.max_tokens = max_tokens
        self._client = client

    def _get_client(self):
        if self._client is None:
            import anthropic  # lazy: only needed for the live path
            self._client = anthropic.Anthropic()
        return self._client

    def generate(self, ctx: RationaleContext) -> Rationale:
        client = self._get_client()
        user = (
            "Brief (all numbers you may use are in here):\n"
            + json.dumps(ctx.brief, indent=2)
            + "\n\nWrite the recommendation rationale as JSON with keys "
            "headline, rationale, risk, alternative."
        )
        resp = client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=_SYSTEM,
            output_config={"format": {"type": "json_schema", "schema": _SCHEMA}},
            messages=[{"role": "user", "content": user}],
        )
        text = next(b.text for b in resp.content if b.type == "text")
        data = json.loads(text)
        return Rationale(**data)


class TemplateRationaleGenerator(RationaleGenerator):
    """Deterministic, offline rationale — numbers echoed, names resolved, no coords."""

    def generate(self, ctx: RationaleContext) -> Rationale:
        b = ctx.brief
        rec = b["recommendation"]
        alt = b.get("alternative")
        handler = b["handler"]
        epv = f"{rec['epv']:.2f}"

        if rec["action"] == "SHOOT":
            headline = f"Let {rec['shooter']} shoot from the {rec['zone'].replace('_', ' ')}"
            reason = (f"{rec['shooter']} has the best look on the floor — a {epv} "
                      f"expected-points shot with a {rec['success_pct']} percent make chance")
            risk = f"It is a {rec['points']}-point attempt; the make probability is {rec['success_prob']:.2f}."
        elif rec["action"] == "PASS_TO":
            sep = rec.get("target_separation_ft")
            headline = f"{handler} should pass to {rec['target']}"
            reason = (f"{rec['target']} is the higher-value option at {epv} expected points"
                      + (f", with {sep} feet of separation" if sep is not None else ""))
            risk = f"The pass completion probability is {rec['success_prob']:.2f}."
        elif rec["action"] == "DRIVE":
            headline = f"{handler} should drive {rec['direction']}"
            reason = f"Attacking {rec['direction']} is worth {epv} expected points here"
            risk = f"The drive succeeds with probability {rec['success_prob']:.2f}."
        else:
            headline = f"{handler} should reset the offense"
            reason = f"No current look beats a reset at {epv} expected points"
            risk = "Low risk, but it burns clock."

        if b.get("handler_defender_separation_ft") is not None and rec["action"] != "SHOOT":
            reason += (f"; {handler} is being guarded at "
                       f"{b['handler_defender_separation_ft']} feet")
        if b.get("coaching_emphasis"):
            reason += f". This fits the emphasis on {b['coaching_emphasis']}"
        rationale = reason + "."

        if alt:
            alt_epv = f"{alt['epv']:.2f}"
            if alt["action"] == "PASS_TO":
                alternative = f"If it is not there, {alt['target']} is the fallback at {alt_epv} expected points."
            elif alt["action"] == "SHOOT":
                alternative = f"Otherwise {alt['shooter']} can shoot for {alt_epv} expected points."
            else:
                alternative = f"The fallback is to {alt['action'].lower().replace('_', ' ')} at {alt_epv} expected points."
        else:
            alternative = "No strong second option; run the set again."

        return Rationale(headline=headline, rationale=rationale, risk=risk, alternative=alternative)
