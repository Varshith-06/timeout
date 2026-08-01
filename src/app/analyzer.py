"""Pause-to-overlay analyzer (roadmap 5.5).

The app layer's core: given a state (from SportVU or CV — the schema is the same,
roadmap 5.1), score the candidate actions, gate on confidence, and return the
recommendation fast. The prose rationale is deliberately a *second click*
(roadmap 5.4/5.5): ``analyze()`` renders the overlay immediately from the value
model; ``rationale()`` calls the LLM only on demand.

Results are cached by (game_id, timestamp) because coaches re-watch the same
moments (5.5), and each analysis is timed against the sub-2-second budget.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

from src.llm.context import CoachingPriors
from src.llm.rationale import RationaleResult, generate_rationale
from src.perception.state_from_cv import CONFIDENCE_GATE, MIN_PLAYERS_SHOW
from src.value.actions import enumerate_actions
from src.value.scoring import ScoredAction, score_actions

LATENCY_BUDGET_MS = 2000.0


@dataclass
class Analysis:
    state: object
    scored: list[ScoredAction]
    confidence: float
    showable: bool
    latency_ms: float
    cache_hit: bool = False
    _by_id: dict = field(default_factory=dict)

    @property
    def top(self) -> ScoredAction | None:
        return self.scored[0] if self.scored else None

    def why_not(self, action_id: str) -> dict:
        """Compare any candidate to the top recommendation (the 'why not X?' button)."""
        if action_id not in self._by_id or not self.scored:
            return {}
        sc, top = self._by_id[action_id], self.scored[0]
        delta = round(top.q - sc.q, 3)
        return {
            "action": sc.action.action,
            "epv": round(sc.q, 3),
            "top_epv": round(top.q, 3),
            "epv_gap": delta,
            "success_prob": round(sc.success_prob, 3),
            "verdict": "recommended" if delta <= 1e-6 else f"{delta:.2f} EPV worse than the top option",
        }

    def within_budget(self) -> bool:
        return self.latency_ms <= LATENCY_BUDGET_MS


class PlayAnalyzer:
    def __init__(self, submodels, value_model, names: dict | None = None,
                 coaching: CoachingPriors | None = None, generator=None):
        self.submodels = submodels
        self.value_model = value_model
        self.names = names or {}
        self.coaching = coaching
        self.generator = generator
        self._cache: dict = {}

    def _key(self, state, key):
        if key is not None:
            return key
        ts = state.timestamp
        return (state.possession_id, ts.get("quarter"), round(ts.get("game_clock") or 0.0, 1))

    def analyze(self, state, key=None) -> Analysis:
        """Score + gate a paused state. Cached by (game_id, timestamp)."""
        ck = self._key(state, key)
        if ck in self._cache:
            cached = self._cache[ck]
            return Analysis(cached.state, cached.scored, cached.confidence,
                            cached.showable, 0.0, cache_hit=True, _by_id=cached._by_id)

        t0 = time.perf_counter()
        actions = enumerate_actions(state)
        scored = score_actions(state, actions, self.submodels, self.value_model) if actions else []
        conf = state.context.confidence
        showable = bool(scored) and conf >= CONFIDENCE_GATE and \
            state.context.n_players_observed >= MIN_PLAYERS_SHOW
        latency = (time.perf_counter() - t0) * 1000.0

        result = Analysis(state, scored, conf, showable, latency,
                          _by_id={sc.action.id: sc for sc in scored})
        self._cache[ck] = result
        return result

    def rationale(self, analysis: Analysis, playbook: list[str] | None = None) -> RationaleResult:
        """Generate the prose rationale on demand (the 'explain this' second click)."""
        return generate_rationale(
            analysis.state, analysis.scored, self.names,
            generator=self.generator, coaching=self.coaching, playbook=playbook,
        )

    def cache_size(self) -> int:
        return len(self._cache)
