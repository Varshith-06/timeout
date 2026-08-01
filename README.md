# NBA Play Recommender

A pause-and-analyze system for NBA game footage: freeze the video, get a ranked
set of actions the offense could take, scored in expected points, drawn on
screen. See [`nba-play-recommender-roadmap.md`](nba-play-recommender-roadmap.md)
for the full six-month plan.

## Status: Phases 1–5 complete (end-to-end)

- **Phase 1** — the reasoning layer on perfect tracking (state schema, candidate actions, renderer).
- **Phase 2** — the value model: calibrated sub-models + a permutation-invariant V(s), scoring every candidate action in expected points. Trains on the GPU when available.
- **Phase 3** — perception: turn a broadcast into the *same* state schema (homography, tracking, team/identity, clock), with confidence gating and a domain-gap measurement that feeds back into Phase 2's augmentation.
- **Phases 4–5** — integration, the LLM rationale layer (strict no-coordinates / numbers-echoed schema), and the pause-to-overlay app with caching, a sub-2s latency budget, and "why not X?".

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

## Phase 3 — perception

Turn a broadcast into the *same* state schema, so Phases 1–2 run on it unchanged.
The learned front-end (YOLO detection, court keypoints, jersey OCR, pose) needs
annotated frames that don't exist publicly — so it sits behind a documented
interface with a **synthetic broadcast** stand-in: the court-space tracking we
already have, projected through a pinhole camera to pixel detections, keypoints,
a clock, and camera cuts. That gives paired pixel↔court data, which makes the
whole geometry/tracking spine real, testable, and *measurable*.

| Roadmap | Module | What it does |
|---|---|---|
| §4.1 | `perception/camera.py`, `synthetic_broadcast.py` | Pinhole camera; projects tracking → noisy pixel detections + keypoints + clock + cuts, with paired ground truth |
| §4.6 | `perception/homography.py` | `cv2` RANSAC pixel→court, reprojection-error gate, temporal corner smoothing, foot projection — the load-bearing piece |
| §4.2 | `perception/detection.py` | Detection schema + `Detector` interface (real YOLO plugs in here) |
| §4.3 | `perception/tracking.py`, `cuts.py` | Foot-distance tracker with tracklets + histogram cut detection (hard reset on cuts) |
| §4.4/4.5/4.8 | `perception/teams.py`, `identity.py`, `clock.py` | KMeans team clustering, jersey tracklet-voting + roster map, temporally-validated clock |
| §4.9/5.1 | `perception/state_from_cv.py` | Builds the Phase 1 `State` with composite `confidence` + gating; never imputes a missing defender |
| §5.2→3.3 | `perception/evaluate.py`, `augment_feedback.py` | Measures position/identity/miss error on paired data → the augmentation config to retrain V(s) |
| §5.3 | `perception/video_overlay.py` | Projects court→pixel via H⁻¹; draws the recommendation on the frame between tracked positions |

### Demo

```bash
python scripts/demo_phase3.py --seed 5     # broadcast -> state -> EPV -> overlay + domain gap
```

### Results on the synthetic broadcast (paired vs ground truth)

| Metric | Result | Roadmap target |
|---|---|---|
| Homography median reprojection error | ~0.1–0.2 ft | < 1.5 ft |
| Recovered player position error (median) | **0.66 ft** | < 1.5 ft |
| Team-assignment accuracy | **1.00** | > 0.97 |
| Jersey identity accuracy | **1.00** | > 0.90 |
| Shot-clock accuracy (after validation) | ~0.97 | > 0.99 |
| Confidence gating | withholds below 8 players / 0.6 confidence (§4.9) | — |

The measured gap is fed straight back into Phase 2's noise augmentation
(`augment_feedback.py`), closing the §5.2 loop.

## Phases 4–5 — integration, rationale, and the app

| Roadmap | Module | What it does |
|---|---|---|
| §5.1 | `src/state/validate.py` | Conformance validator — SportVU and CV states pass the *same* checks (only `confidence` differs) |
| §5.4 | `src/llm/schema.py` | Strict `Rationale` schema + constraint validator: no coordinates, player-ids never leaked, every number echoed from the value model |
| §5.4 | `src/llm/context.py` | Resolves player-ids to names, enumerates the numbers the model may state, folds in coaching priors + matched plays |
| §5.4 | `src/llm/client.py` | `ClaudeRationaleGenerator` (`claude-opus-4-8`, strict JSON) + a deterministic offline `TemplateRationaleGenerator` |
| §5.4 | `src/llm/rationale.py` | Generate → validate → regenerate, with a constraint-valid template fallback |
| §5.5 | `src/app/analyzer.py` | Pause-to-overlay: score + gate, cache by (game, timestamp), sub-2s budget, "why not X?", rationale as a second click |

### Demo

```bash
python scripts/demo_phase4.py --seed 5                 # conformance + app + rationale (offline)
python scripts/demo_phase4.py --seed 5 --use-claude    # live Claude rationale (needs API creds)
```

The rationale is a *second click* (§5.4): the overlay renders instantly from the
value model; the LLM is called only on demand. The offline generator is the
default so the whole layer runs and tests without a network or API key — and by
construction it can't fabricate a number or leak a coordinate.

## Layout

```
src/
  ingest/     SportVU parsing, synthetic generator, possession segmentation
  state/      court geometry + the state schema + conformance validator
  value/      candidate actions + the Phase 2 value model
  perception/ Phase 3: camera, homography, tracking, teams, identity, clock, state-from-CV
  render/     court renderer + video overlay
  llm/        Phase 5.4: strict rationale schema, context, Claude + offline generators
  app/        Phase 5.5: pause-to-overlay analyzer with caching + latency budget
tests/        unit + end-to-end tests (54 tests)
scripts/      demo_phase1.py, train_phase2.py, demo_phase2.py, demo_phase3.py, demo_phase4.py
configs/      coaching_priors.json
```

## The full pipeline

```
broadcast video ─(Phase 3)─▶ State ◀─(Phase 1)─ SportVU tracking
                              │  (same schema — validated in Phase 4)
                              ▼
                   (Phase 2) candidate actions scored in expected points
                              ▼
              (Phase 5) overlay + confidence gate + LLM rationale, in the app
```

Everything runs today on synthetic data with no downloads; drop in real SportVU
logs and broadcast footage and the same code path runs unchanged.
