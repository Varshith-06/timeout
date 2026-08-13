# Timeout — NBA Play Recommender

Pause an NBA game at any moment and Timeout draws the **best thing the offense could do
next** on the frame — an arrow, a highlighted player, and a number in **expected points** —
then explains why in a sentence. It's not a play *predictor* (what will happen); it's an
action *evaluator* (what's worth the most).

**[Live demo](http://timeout-webapp-516633645639.s3-website.ap-south-1.amazonaws.com)** ·
**[`timeout.pdf`](timeout.pdf)** — the complete walkthrough: design, methods, every result,
and the deployment. Read that for anything this page summarises.

---

## Results

Every number here comes from `python scripts/benchmark.py`. **100 tests pass** end to end.

### Scoring quality — real 2015-16 SportVU, held out by game

Trained on ~240 games (≈25k shots, 45k passes, 49k drives, 627k possession states). The
possession parser independently recovers NBA-real rates (**≈1.05 points/possession**).

| Sub-model | Brier | Baseline | |
|---|---|---|---|
| Shot make | **0.200** | 0.248 | **beats** |
| Drive success | **0.035** | 0.090 | **beats** |
| Pass completion | 0.068 | 0.068 | ties — completion is ~90%, the base rate is hard to beat |
| V(s) MSE | 1.468 | 1.468 | ties — the honest weak point (corr 0.02) |

### Recommendation quality — vs. naive strategies

Graded against a ground-truth generator by regret in expected points:

| Strategy | Right-or-defensible | Wrong | Mean regret |
|---|---|---|---|
| **Timeout** | **86%** | **4%** | **0.058** |
| Always pass to most open | 40% | 48% | 0.274 |
| Always shoot | 30% | 55% | 0.337 |
| Random | 20% | 55% | 0.352 |

≈5× fewer expected points lost than the best simple rule.

### On real broadcast footage — Heat/Nets clip, 130 s

| Measured | Result |
|---|---|
| Pause points with a recommendation | **111 / 146 (76%)** across three calibrated shots |
| On-court players per pause | **median 10** (range 7–13) — the action, not the crowd |
| Live ball detection | **90%** (the rest coast on the velocity tracker) |
| Resolved handler / located rim | **100% / 100%** |
| Calibration reprojection error | **0.14–1.3 ft** (target < 1.5 ft) |
| Per-pause scoring latency | **~38 ms** median, 13 candidates (2 s budget) |

A median of exactly 10 players is the headline of the detector work: a purpose-built
basketball detector's `player` class excludes the crowd, where a generic "person" detector
boxed 30–40.

### Per-frame calibration — trained court-keypoint model

A click calibration is solved on one frame and carried across the shot by optical flow, so
it drifts. A keypoint model that finds the landmarks in *every* frame fixes that — trained
on 683 frames auto-labelled from the existing calibrations across seven camera shots on three
broadcasts (no hand-labelling) — 635 for training, one whole shot (48 frames) held out.

| | propagated (default) | per-frame model |
|---|---|---|
| within a trained shot, 11 s | drifts **0.18 → 16.8 ft** | flat **1.55 → 1.74 ft** |
| frames passing the acceptance gate | — | 30% in-domain, **0% held-out** |

**The drift is real and severe** — the propagated calibration is catastrophically wrong ~7 s
after its anchor, and the model is stable where it isn't. But at this data volume the model
does not generalise to an unseen camera shot, so `--keypoint-model` is **opt-in and off by
default**. Training loss keeps falling while held-out error plateaus: the bottleneck is
calibrated shots, not epochs. More `calibrate.py` sessions feed the same pipeline unchanged.

The lesson worth carrying: **reprojection error measures self-consistency, not correctness.**
A model that hallucinates a consistent set of landmarks scores well while projecting a court
nowhere near the real one, so acceptance also requires a physical check
(`plausible_court_homography`) — a foot of court spans 17–56 px on real calibrations versus
1.5–5 px on hallucinated ones.

---

## Quick start

```bash
pip install -e .        # numpy, polars, pyarrow, matplotlib, torch, lightgbm, scikit-learn, opencv
python -m pytest -q     # 100 tests, no data or GPU required
```

Runs on synthetic data with no downloads:

```bash
python scripts/demo_phase1.py --seed 3               # state + candidate actions + court render
python scripts/demo_phase3.py --seed 5               # broadcast → state → EPV → overlay
python scripts/demo_phase4.py --seed 5               # conformance + app + rationale
```

On real broadcast video:

```bash
pip install -e ".[realvideo]"
python scripts/fetch_youtube.py "https://youtu.be/..." --start 1:05 --end 1:20
python scripts/calibrate.py    --video clip.mp4 --time 39 --out calib.json
python scripts/build_webapp.py --video clip.mp4 --shots calib.json --detector roboflow
python scripts/serve_webapp.py                       # http://localhost:8000
```

Or `python scripts/studio.py` for the setup wizard (upload a clip or paste a YouTube link,
calibrate in the browser, build). Deployment to AWS lives in [`deploy/aws/`](deploy/aws/).

---

## How it works

```
broadcast video ──(perception)──▶  State  ◀──(parser)── SportVU tracking
                                     │   one schema, validated identical from both sources
                                     ▼
                     (value model)  candidate actions scored in expected points
                                     ▼
                (app)  overlay + confidence gate + LLM rationale, on pause
```

The **state schema** is the contract: a deterministic, code-produced object (nothing inferred
by an LLM) describing every player and the ball, defender pressure, spacing, and a
`confidence` field. A SportVU possession and a broadcast-CV possession produce a
*byte-identical* state, so every downstream component — all trained on clean official data —
runs on messy video unchanged.

Two rules keep it honest: **actions are structured objects, never coordinates** (the renderer
is the only thing that turns an action into geometry, and every endpoint is a tracked
position), and the perception path **never imputes a missing defender** — it drops
`confidence` and, below a threshold, withholds the recommendation entirely.

| Component | Modules |
|---|---|
| Court + state | `src/state/` — geometry, possession segmentation, schema + conformance validator |
| Ingest | `src/ingest/` — SportVU parsing, synthetic generator, real shot/play-by-play labels |
| Value model | `src/value/` — ≤13 candidate actions, `Q(s,a) = P(success)·V(s') + (1−P)·V_turnover`; LightGBM sub-models + permutation-invariant deep-set V(s) |
| Perception | `src/perception/` — homography, tracking, cut detection, team clustering, jersey OCR, keypoint model, confidence gate |
| Rationale | `src/llm/` — strict schema, every number echoed from the value model |
| App | `src/app/` — pause-to-overlay, cache, chat assistant, studio wizard + web player |

---

## Honest limitations

- **V(s) is weak on real data.** Single-sample returns give little per-state signal; it needs
  TD(λ) smoothing and more data. Ranking still works because it leans on the three strong
  sub-models.
- **Calibration is still manual.** The keypoint model is built and gated but doesn't yet
  generalise to unseen camera shots (above). Automatic *from-scratch* line detection was tried
  and set aside — broadcast wood mixes light and dark strokes with logo/jersey interference.
- **Coverage has gaps.** Recommendations exist only inside calibrated shots; elsewhere the app
  shows an honest "no coverage" note rather than a stale arrow.
- **The action mix can skew.** On the test clip the top pick is "set a screen" most of the
  time — a scoring question to investigate, not a detection one.

Full detail, method, and reasoning for all of the above: **[`timeout.pdf`](timeout.pdf)**.
