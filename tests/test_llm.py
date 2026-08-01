"""Phase 5.4 tests: rationale schema, constraint validation, generators."""
import json

import pytest

from src.ingest.possessions import iter_possessions
from src.ingest.sportvu import parse_game
from src.ingest.synthetic import generate_game
from src.state.schema import build_states, roster_jersey_map
from src.value.actions import enumerate_actions
from src.value.features import PlayerVocab
from src.value.scoring import score_actions
from src.value.simulation import build_dataset
from src.value.state_value import train_value_model
from src.value.submodels import train_submodels


@pytest.fixture(scope="module")
def scored_state():
    ds = build_dataset(n_possessions=400, seed=3)
    submodels = train_submodels(ds)
    vocab = PlayerVocab(ds.player_ids)
    vm, _ = train_value_model(ds.possessions, vocab, epochs=6, seed=1)
    game = parse_game(generate_game(n_possessions=1, attack="left", seed=5))
    poss = list(iter_possessions(game))[0]
    states = build_states(poss, roster_jersey_map(game))
    s = next(st for st in states if st.handler is not None and len(enumerate_actions(st)) >= 3)
    scored = score_actions(s, enumerate_actions(s), submodels, vm)
    return game, s, scored


# --- Validator ---------------------------------------------------------------
def test_validator_catches_violations():
    from src.llm.schema import Rationale, validate_rationale
    bad = Rationale(
        headline="Pass to player 2547",
        rationale="Open at (24.1, 18.7) for a 1.42 look",
        risk="completion 0.90",
        alternative="reset",
    )
    v = validate_rationale(bad, allowed_numbers=[0.90, 2.0, 3.0], allowed_player_ids=[2547])
    assert any("coordinate" in x for x in v)
    assert any("fabricated" in x for x in v)
    assert any("player_id" in x for x in v)


def test_validator_passes_clean():
    from src.llm.schema import Rationale, validate_rationale
    good = Rationale(
        headline="Kick it out",
        rationale="A 1.18 look beats the 0.94 pull-up",
        risk="completion is 0.81",
        alternative="reset if help recovers",
    )
    assert validate_rationale(good, [1.18, 0.94, 0.81], allowed_player_ids=[123]) == []


# --- Template generator (offline default) ------------------------------------
def test_template_rationale_is_constraint_valid(scored_state):
    from src.llm.context import build_context, roster_name_map
    from src.llm.client import TemplateRationaleGenerator
    from src.llm.schema import validate_rationale
    game, s, scored = scored_state
    ctx = build_context(s, scored, roster_name_map(game))
    r = TemplateRationaleGenerator().generate(ctx)
    assert validate_rationale(r, ctx.allowed_numbers, ctx.allowed_player_ids) == []
    # A resolved name (with a space) appears; no raw id does.
    assert any(nm in r.all_text() for nm in ctx.names.values())


def test_generate_rationale_returns_clean(scored_state):
    from src.llm.context import roster_name_map
    from src.llm.rationale import generate_rationale
    game, s, scored = scored_state
    res = generate_rationale(s, scored, roster_name_map(game))
    assert res.violations == []
    assert res.source == "template_fallback"
    assert res.rationale.headline


# --- Claude generator parsing path (no network; injected fake client) --------
def test_claude_generator_parses_schema(scored_state):
    from src.llm.client import ClaudeRationaleGenerator
    from src.llm.context import build_context, roster_name_map
    game, s, scored = scored_state
    ctx = build_context(s, scored, roster_name_map(game))

    payload = {"headline": "h", "rationale": "r", "risk": "k", "alternative": "a"}

    class _Block:
        type = "text"
        text = json.dumps(payload)

    class _Resp:
        content = [_Block()]

    class _Msgs:
        def create(self, **kwargs):
            # Sanity: correct model + strict schema requested.
            assert kwargs["model"] == "claude-opus-4-8"
            assert kwargs["output_config"]["format"]["type"] == "json_schema"
            return _Resp()

    class _Client:
        messages = _Msgs()

    gen = ClaudeRationaleGenerator(client=_Client())
    r = gen.generate(ctx)
    assert r.headline == "h" and r.alternative == "a"
