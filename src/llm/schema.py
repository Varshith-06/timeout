"""The rationale schema + constraint validator (roadmap 5.4).

The LLM produces a one-paragraph rationale under a strict schema. Three hard
constraints are enforced on parse (5.4):

  * No coordinates in the output — not pixels, not feet.
  * Player references resolved to names by our code; the model never emits raw
    player_ids.
  * All numeric claims are passed *in* from the value model and echoed, never
    generated — a number not present in the input is rejected.

The validator is what makes those constraints real: :func:`validate_rationale`
returns violations, and the orchestrator rejects-and-regenerates on any.
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass

# A coordinate-looking token: "(12.3, 45.6)", "x=12", "y = 3.4", "pixel".
_COORD_PATTERNS = [
    re.compile(r"\(\s*-?\d+(\.\d+)?\s*,\s*-?\d+(\.\d+)?\s*\)"),
    re.compile(r"\b[xy]\s*=\s*-?\d+(\.\d+)?", re.IGNORECASE),
    re.compile(r"\bpixel", re.IGNORECASE),
]
# Any number (int or decimal), optionally trailing % — the claims we audit.
_NUMBER = re.compile(r"-?\d+(?:\.\d+)?")


@dataclass
class Rationale:
    headline: str
    rationale: str
    risk: str
    alternative: str

    def to_dict(self) -> dict:
        return asdict(self)

    def all_text(self) -> str:
        return " ".join([self.headline, self.rationale, self.risk, self.alternative])


def extract_numbers(text: str) -> list[float]:
    return [float(m) for m in _NUMBER.findall(text)]


def validate_rationale(
    rationale: Rationale,
    allowed_numbers: list[float],
    allowed_player_ids: list[int],
    number_tol: float = 0.02,
) -> list[str]:
    """Return constraint violations; empty means the rationale is acceptable.

    allowed_numbers: every numeric value the model is permitted to state (EPVs,
    probabilities as percents and fractions, separation feet), pre-expanded by
    the context builder. allowed_player_ids: ids that must NOT appear literally.
    """
    problems: list[str] = []
    text = rationale.all_text()

    # 1. No coordinates.
    for pat in _COORD_PATTERNS:
        if pat.search(text):
            problems.append(f"coordinate-like token found: {pat.pattern}")

    # 2. No raw player_ids.
    for pid in allowed_player_ids:
        if re.search(rf"\b{pid}\b", text):
            problems.append(f"raw player_id {pid} leaked into prose")

    # 3. Every number must echo an allowed input number (within tolerance).
    allowed = list(allowed_numbers)
    for n in extract_numbers(text):
        if not any(abs(n - a) <= max(number_tol, abs(a) * number_tol) for a in allowed):
            problems.append(f"fabricated number in output: {n}")

    # 4. Non-empty fields.
    for field, val in rationale.to_dict().items():
        if not val or not val.strip():
            problems.append(f"empty field: {field}")

    return problems
