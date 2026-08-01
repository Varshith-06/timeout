# NBA Play Recommender

A pause-and-analyze system for NBA game footage: freeze the video, get a ranked
set of actions the offense could take, scored in expected points, drawn on
screen. See [`nba-play-recommender-roadmap.md`](nba-play-recommender-roadmap.md)
for the full six-month plan.

## Status: Phases 1–2 complete

- **Phase 1** — the reasoning layer on perfect tracking (state schema, candidate actions, renderer).
- **Phase 2** — the value model: calibrated sub-models + a permutation-invariant V(s), scoring every candidate action in expected points. Trains on the GPU when available.

## Phase 1 — reasoning layer on perfect tracking

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

## Phase 2 — the value model

A calibrated function mapping a state to expected points, and a one-step
lookahead that scores each candidate action:

    Q(s, a) = P(success | s, a) · V(s'_a) + (1 − P(success | s, a)) · V_turnover

| Roadmap | Module | What it does |
|---|---|---|
| §3.2 | `src/value/features.py` | Shot/pass/drive tabular features + permutation-invariant entity tensor for V(s) |
| — | `src/value/simulation.py` | Ground-truth possession simulator: labeled shots/passes/drives + realized points, so calibration is *verifiable* without downloads |
| §3.2a-c | `src/value/submodels.py` | LightGBM shot / pass / drive models; empirical-Bayes per-player shot prior |
| §3.2d | `src/value/state_value.py` | Permutation-invariant deep-set V(s) in PyTorch (GPU-trained), MSE to realized points |
| §3.3 | `src/value/augment.py` | Noise augmentation: position jitter, player dropout, id-swap, velocity lag |
| §3.1 | `src/value/transition.py`, `scoring.py` | Approximate s'→a and the Q-scoring above |
| §3.4 | `src/value/evaluate.py` | Reliability/Brier, EPV trajectories, recommendation quality vs naive baselines |

### Train and demo

```bash
python scripts/train_phase2.py --possessions 10000 --epochs 60   # trains on the GPU if available
python scripts/demo_phase2.py --seed 5                           # EPV-scored overlays
```

`train_phase2.py` writes models to `models/phase2/` and evaluation artifacts to
`out/phase2/` (reliability diagram, EPV trajectories, `metrics.json`).

### Results on simulated possessions (week-8 gate, §3.5)

| Metric | Result |
|---|---|
| Shot / pass / drive Brier | 0.226 / 0.103 / 0.197 — all beat base rate |
| Recommendation right-or-defensible | **86%** (gate: ≥70%) |
| Recommendation wrong | **4%** (gate: <10%) |
| Mean regret vs best naive baseline | **0.058 vs 0.274** (beats baseline ~5×) |

Calibration is validatable because the simulator knows the true probabilities;
the learned models never see them. **GATE: PASS.**

## Layout

```
src/
  ingest/    SportVU parsing, synthetic generator, possession segmentation
  state/     court geometry + the state schema
  value/     candidate actions + the Phase 2 value model
  render/    court renderer
  perception/ llm/ app/          scaffolded, empty until later phases
tests/       unit + end-to-end pipeline tests (30 tests)
scripts/     demo_phase1.py, train_phase2.py, demo_phase2.py
```

## Next: Phase 3 — perception

Turn broadcast video into the *same* state schema (detection, tracking,
homography, jersey OCR). The Phase 1/2 stack then runs unchanged on CV-derived
states — which is why the state schema is the contract everything is built on.
