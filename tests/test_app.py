"""Phase 5.1 / 5.5 tests: conformance validator and the pause-to-overlay app."""
import pytest

from src.ingest.possessions import iter_possessions
from src.ingest.sportvu import parse_game
from src.ingest.synthetic import generate_game
from src.state.schema import build_states, roster_jersey_map
from src.state.validate import is_conformant, validate_state
from src.value.actions import enumerate_actions
from src.value.features import PlayerVocab
from src.value.simulation import build_dataset
from src.value.state_value import train_value_model
from src.value.submodels import train_submodels


@pytest.fixture(scope="module")
def env():
    ds = build_dataset(n_possessions=400, seed=3)
    submodels = train_submodels(ds)
    vocab = PlayerVocab(ds.player_ids)
    vm, _ = train_value_model(ds.possessions, vocab, epochs=6, seed=1)
    game = parse_game(generate_game(n_possessions=1, attack="left", seed=5))
    poss = list(iter_possessions(game))[0]
    jersey = roster_jersey_map(game)
    sportvu_states = build_states(poss, jersey)
    return submodels, vm, game, poss, jersey, sportvu_states


# --- Conformance (5.1) -------------------------------------------------------
def test_sportvu_state_conforms(env):
    _, _, _, _, _, states = env
    s = next(st for st in states if st.handler is not None)
    assert is_conformant(s), validate_state(s)


def test_cv_state_conforms_same_validator(env):
    submodels, vm, game, poss, jersey, _ = env
    from src.perception.state_from_cv import (build_state_from_cv, pick_showable_frame,
                                              recover_tracking)
    from src.perception.synthetic_broadcast import generate_broadcast
    roster_rows = list(game.roster.select(["team_id", "jersey", "player_id"]).iter_rows())
    clip = generate_broadcast(poss, seed=1, cut_prob=0.0, stride=5, jersey_map=jersey)
    rec = recover_tracking(clip, roster_rows, stride=5)
    cv_state, conf = build_state_from_cv(rec, pick_showable_frame(rec), roster_jersey=jersey)
    assert is_conformant(cv_state), validate_state(cv_state)
    # The one field that legitimately differs from SportVU.
    assert cv_state.context.confidence < 1.0


# --- App analyzer (5.5) ------------------------------------------------------
@pytest.fixture(scope="module")
def analyzer_and_state(env):
    submodels, vm, game, poss, jersey, states = env
    from src.app.analyzer import PlayAnalyzer
    from src.llm.context import roster_name_map
    analyzer = PlayAnalyzer(submodels, vm, names=roster_name_map(game))
    s = next(st for st in states if st.handler is not None and len(enumerate_actions(st)) >= 3)
    return analyzer, s


def test_analyze_scores_and_gates(analyzer_and_state):
    analyzer, s = analyzer_and_state
    a = analyzer.analyze(s)
    assert a.top is not None
    assert a.showable is True                 # Phase-1 confidence is 1.0
    assert a.within_budget()                  # under 2 s
    assert a.scored == sorted(a.scored, key=lambda x: x.q, reverse=True)


def test_analyze_caches_by_moment(analyzer_and_state):
    analyzer, s = analyzer_and_state
    analyzer.analyze(s)
    before = analyzer.cache_size()
    a2 = analyzer.analyze(s)
    assert a2.cache_hit and a2.latency_ms == 0.0
    assert analyzer.cache_size() == before     # no new entry


def test_why_not(analyzer_and_state):
    analyzer, s = analyzer_and_state
    a = analyzer.analyze(s)
    worst = a.scored[-1]
    info = a.why_not(worst.action.id)
    assert info["epv_gap"] >= 0
    assert "worse" in info["verdict"] or info["verdict"] == "recommended"
    # The top action reports itself as recommended.
    assert a.why_not(a.top.action.id)["verdict"] == "recommended"


def test_rationale_second_click_is_clean(analyzer_and_state):
    analyzer, s = analyzer_and_state
    a = analyzer.analyze(s)
    res = analyzer.rationale(a, playbook=["Horns set"])
    assert res.violations == []
    assert res.rationale.headline and res.rationale.risk


def test_low_confidence_state_not_showable(analyzer_and_state):
    analyzer, s = analyzer_and_state
    # Force a degraded state: drop confidence below the gate.
    a = analyzer.analyze(s, key="degraded_probe")
    s2 = a.state
    s2.context.confidence = 0.2
    a2 = analyzer.analyze(s2, key="degraded_probe_2")
    assert a2.showable is False
