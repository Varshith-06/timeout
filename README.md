# NBA Play Recommender

A pause-and-analyze system for NBA game footage: freeze the video, get a ranked
set of actions the offense could take, scored in expected points, drawn on
screen. See [`nba-play-recommender-roadmap.md`](nba-play-recommender-roadmap.md)
for the full six-month plan.

## Status: Phase 1 complete — reasoning layer on perfect tracking

Following the roadmap's most important sequencing decision (§2), **Phase 1 is
built before any computer vision.** It proves the recommendation logic produces
basketball-sane output using perfect (SportVU) tracking, so a conceptually empty
reasoning layer would surface in weeks with zero GPU spend.

Everything runs today on **synthetic tracking generated in the real SportVU JSON
shape** — no downloads, no GPU. Drop in real 2015-16 logs (`setup_data.sh`) and
the identical code path runs unchanged.

```
SportVU JSON ──▶ parse ──▶ segment possessions ──▶ state schema
             ──▶ enumerate candidate actions ──▶ court render (PNG + JSON)
```

### What's implemented

| Roadmap | Module | What it does |
|---|---|---|
| §2.3 | `src/state/court.py` | NBA court constants; half-court flip; basket-relative polar; shot-zone assignment |
| §2.2 | `src/ingest/sportvu.py` | Parse the real game JSON; dedupe overlapping events; drop empty moments |
| §2.2 | `src/ingest/synthetic.py` | Generate schema-valid tracking so the pipeline runs with no data |
| §2.4 | `src/ingest/possessions.py` | Possession segmentation; ball-handler assignment + 5-frame smoothing; parquet frame table |
| §2.5 | `src/state/schema.py` | Deterministic state object: pressure, spacing hull, zones, nearest defender, defense scheme |
| §2.6 | `src/value/actions.py` | Enumerate the ≤13 legal candidate actions with pruning |
| §2.7 | `src/render/court_renderer.py` | Top-down matplotlib court; players, ball, velocity; action overlays resolved to geometry |
| §2.8 | `scripts/demo_phase1.py` | The week-3 gate: freeze possessions, print state + candidates, render |

### Design rules honored from the roadmap

- **Coordinates live in one place** (`court.py`) — never hand-rolled elsewhere.
- **The state schema is produced by code, deterministically** — nothing is inferred by an LLM.
- **Actions are structured objects, never coordinates** — the renderer is the only component that turns an action into geometry, and every endpoint comes from a tracked position.
- **`confidence` is always 1.0 in Phase 1**; it will carry perception uncertainty in Phase 4.
- **`orientation_deg` is the velocity heading, explicitly marked** — real torso normal arrives in Phase 3.

## Quick start

```bash
python -m pip install polars pyarrow numpy matplotlib   # Phase 1 deps only
python scripts/demo_phase1.py --seed 3 --possessions 2  # synthetic, no downloads
python -m pytest -q                                     # 19 tests
```

Outputs land in `out/`: `frames.parquet` (the §2.4 deliverable), one PNG overlay
and one `*_state.json` per possession.

Run against a real game once you've fetched data:

```bash
bash setup_data.sh                                      # see prerequisites in the script
python scripts/demo_phase1.py --game data/raw/sportvu_json/0021500431.json
```

## Layout

```
src/
  ingest/    SportVU parsing, synthetic generator, possession segmentation
  state/     court geometry + the state schema
  value/     candidate actions  (value model = Phase 2)
  render/    court renderer
  perception/ llm/ app/          scaffolded, empty until later phases
tests/       unit + end-to-end pipeline tests
scripts/     demo_phase1.py
```

## Next: the week-3 gate

`demo_phase1.py` produces the artifact you take to a coach and ask:
1. Is the candidate list complete? (Did we miss an obvious option?)
2. Is anything on the list absurd?

If the action space is right, Phase 2 (`src/value/`) builds the value model that
replaces the placeholder uniform scores with calibrated expected points.
