# Timeout — NBA Play Recommender

Pause an NBA game at any moment and Timeout draws the **best thing the offense could
do next** on the frame — an arrow, a highlighted player, and a number in **expected
points** — then explains why in a sentence. It's not a play *predictor* (what will
happen); it's an action *evaluator* (what's worth the most): a second opinion with a
number attached.

> **A complete, plain-language walkthrough of the whole project is in
> [`timeout.pdf`](timeout.pdf)** — read that if you want to understand it from scratch.

Given a paused frame it produces:

1. a **court-space state** — all ten players and the ball in feet, with velocities;
2. a **ranked candidate action list** for the ball handler, each scored in expected points;
3. an **overlay** drawing the top action (alternatives on click); and
4. a **plain-English rationale**, generated under a strict schema (no invented numbers).

The reasoning core is trained and validated on real NBA data; the front-end runs on
real broadcast video through a purpose-built basketball detector and a one-time court
calibration. **83 tests pass** end to end.

---

## The web app — watch, pause, ask, adjust

The product is a local single-page app: **play the footage, pause anywhere, and the
best play is drawn on the frame**; ask **why**; **adjust** the shown play from a ranked
list or by chatting with an assistant. There are two parts.

**The studio (setup wizard).** Upload a clip from your device or paste a YouTube link.
It fetches the footage, opens a court-calibration step (click the highlighted
landmarks; you can also click where the ball is to seed tracking), then builds the
recommendations. Calibrate several camera angles to cover more of the game.

```bash
python scripts/studio.py           # open the printed URL, then upload/paste a clip
```

**The player.** Pause and the top action is drawn with an arrow, a highlighted player,
and its EPV. A ranked list of alternatives sits beside the video (click any to draw it).
Ask **why** for a one-paragraph rationale. A **chat assistant** answers free-form
questions ("why not shoot?", "show the drive", "pass to #7") and can change the drawn
play — it runs through **Groq or Claude** when a key is set and falls back to built-in
rules otherwise. The app does **no live inference**: all work is pre-computed and cached,
so pausing is instant. Each overlay is computed for its own frame, so the app never
reuses a recommendation on a far-away moment — outside a calibrated segment it shows an
honest "no coverage here" note instead of a stale arrow.

If you already have a clip + calibration, you can build and serve directly:

```bash
python scripts/calibrate.py   --video clip.mp4 --time 39 --out calib.json
python scripts/build_webapp.py --video clip.mp4 --shots calib.json --detector roboflow
python scripts/serve_webapp.py            # http://localhost:8000
```

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

The **state schema** is the contract the whole system is built on: a deterministic,
code-produced object (nothing inferred by an LLM) describing every player and the ball,
defender pressure, spacing, zones, and a `confidence` field. A SportVU possession and a
broadcast-CV possession produce a *byte-identical* state, so every downstream component —
all trained on clean official data — runs on messy video unchanged.

| Component | Modules | What it does |
|---|---|---|
| **Court + state** | `src/state/` | NBA court geometry (one source of truth for coordinates), possession segmentation, and the deterministic state schema + conformance validator |
| **Ingest** | `src/ingest/` | Parses real SportVU JSON, segments possessions with ball-handler smoothing, generates schema-valid synthetic tracking, loads real shot / play-by-play labels |
| **Value model** | `src/value/` | Enumerates ≤13 legal candidate actions and scores each with `Q(s,a) = P(success)·V(s') + (1−P)·V_turnover`. LightGBM shot/pass/drive sub-models + a permutation-invariant deep-set **V(s)** in PyTorch (GPU) |
| **Perception** | `src/perception/` | Turns a broadcast into the same state: `cv2` homography (reprojection-gated, optical-flow propagated), tracking with camera-cut resets, KMeans team clustering, jersey OCR, temporally-validated shot clock, composite confidence + gating |
| **Rationale** | `src/llm/` | A strict `Rationale` schema (no coordinates, player-ids never leaked, every number echoed from the value model) via Claude (`claude-opus-4-8`) or a deterministic offline template, validate-and-regenerate |
| **App** | `src/app/` | Pause-to-overlay: score + gate, cache, "why not X?", chat assistant (Groq/Claude), the studio wizard + web player |
| **Render** | `src/render/` | Top-down court renderer and the video overlay that projects court→pixel to draw on the frame |

Two design rules keep it honest: **actions are structured objects, never coordinates**
— the renderer is the only thing that turns an action into geometry, and every endpoint
comes from a tracked position; and the perception path **never imputes a missing
defender** — it drops `confidence` and, below a threshold, withholds the recommendation
entirely.

---

## Benchmark

Run the whole thing yourself: **`python scripts/benchmark.py`** (writes
`out/benchmark/benchmark.json`), and the figures with `python scripts/benchmark_charts.py`.
Numbers below are that script's output; the charts also feed `timeout.pdf`.

### Scoring quality on real 2015-16 data

Trained on **~240 games** of SportVU tracking joined to real shot + play-by-play labels
(≈25k shots, 45k passes, 49k drives, 627k possession states), **split by game** (held-out
games, never frames). The possession parser recovers NBA-real rates (**≈1.05
points/possession**).

| Sub-model | Real Brier | Baseline | |
|---|---|---|---|
| Shot make | **0.200** | 0.248 | **beats** |
| Drive success | **0.035** | 0.090 | **beats** |
| Pass completion | 0.068 | 0.068 | ties — completion is ~90%, the base rate is hard to beat |
| V(s) MSE | 1.468 | 1.468 | ties — weak per-state signal from single-sample returns (corr 0.02) |

The shot and drive sub-models clearly beat their base rates; V(s) is the honest weak
point (it needs TD(λ) smoothing and more data). The action *ranking* still works because
it leans mostly on the three strong sub-models. Reliability diagrams are in `out/real/`.

### Recommendation quality (simulated possessions, vs. naive strategies)

Scored against a ground-truth generative model, each pick is graded by its regret in
expected points versus the best available action:

| Strategy | Right-or-defensible | Wrong | Mean regret |
|---|---|---|---|
| **Timeout** | **86%** | **4%** | **0.058** |
| Always pass to most open | 40% | 48% | 0.274 |
| Always shoot | 30% | 55% | 0.337 |
| Random | 20% | 55% | 0.352 |

Timeout loses ≈5× fewer expected points than the best simple rule.

### On real footage (the Heat/Nets test clip, 130 s)

| What was measured | Result |
|---|---|
| Pause points with a recommendation | **111 / 146 (76%)**, spanning three calibrated shots |
| On-court players detected per pause | **median 10** (range 7–13) — the five-on-five action, not the crowd |
| Pauses with a live ball detection | **90%** (the rest coast on the velocity tracker) |
| Pauses with a resolved handler / located rim | **100% / 100%** |
| Court calibration reprojection error | **0.14–1.3 ft** (target < 1.5 ft) |
| Per-pause scoring latency | **~38 ms** median (13 candidates; 2 s budget) |

A median of exactly 10 players is the headline of the detector work: the purpose-built
basketball detector's `player` class excludes the crowd/bench, where a generic "person"
detector boxed 30–40.

---

## Quick start

```bash
pip install -e .        # core: numpy, polars, pyarrow, matplotlib, torch, lightgbm, scikit-learn, opencv
python -m pytest -q     # 83 tests, no data or GPU required
```

Everything below runs on synthetic data with no downloads:

```bash
python scripts/demo_phase1.py --seed 3               # state + candidate actions + court render
python scripts/train_phase2.py --possessions 10000   # train the value model (GPU if available)
python scripts/demo_phase2.py  --seed 5              # EPV-scored overlays
python scripts/demo_phase3.py  --seed 5              # broadcast → state → EPV → overlay + domain gap
python scripts/demo_phase4.py  --seed 5              # conformance + app + rationale (offline)
python scripts/demo_phase4.py  --seed 5 --use-claude # live Claude rationale (needs API creds)
```

Train on real data (games + weights are gitignored):

```bash
python scripts/fetch_real_data.py 240                # download games + shot/play-by-play labels
for s in $(seq 0 12 240); do \
  python scripts/build_real_cache.py --start $s --count 12; done   # memory-safe chunks
python scripts/train_real.py --from-cache data/cache # train the full stack + write metrics
```

For a quick run, `python scripts/train_real.py --max-games 15`. For GPU training install a
CUDA build of torch, e.g. `pip install torch --index-url https://download.pytorch.org/whl/cu126`.
The real dataset is built in **memory-safe chunks** (each a fresh process) because many
games in one process leak memory through polars' Arrow arena.

---

## Running on real broadcast video

The real-video path uses **pretrained** models (no annotation/training): a basketball
detector plus a **one-time manual court calibration** in place of the (untrained)
court-keypoint model.

```bash
pip install -e ".[realvideo]"                              # nba_api + yt-dlp + ultralytics
python scripts/fetch_youtube.py "https://youtu.be/..." --start 1:05 --end 1:20
python scripts/calibrate.py --video clip.mp4 --time 2.0 --out calib.json
python scripts/build_webapp.py --video clip.mp4 --shots calib.json --detector roboflow
python scripts/serve_webapp.py
```

**Calibration** is a ~45-second click step per camera shot: a court diagram highlights a
landmark, you click the matching spot (5–8 well-spread line intersections), and it draws
the projected court lines so you can verify the fit. Event-driven controls let you undo a
point, skip a landmark, or reject and redo a shot. Optical flow then propagates the
homography across the shot. This is the reliable substitute for a trained keypoint model.

### Detectors (dropped in by class name — no code change)

The detector maps its own class *names* onto the schema, so any detector works:

- **Hosted Roboflow workflow** (recommended) — `--detector roboflow` calls a serverless
  **basketball player-detection** workflow whose `player` class detects the ~10 on-court
  players and **not the crowd**, with a separate `referee` class (excluded) and reliable
  `ball` detection. This is what fixed "players/ball not detected properly." The key is
  read from `$ROBOFLOW_API_KEY` and never stored. One network call per frame, so use a
  larger `--stride` and `--detect-workers` for concurrency; transient serverless timeouts
  are retried and a single unreachable frame is skipped rather than crashing the build.

  ```bash
  export ROBOFLOW_API_KEY=...   # your key; never committed
  python scripts/build_webapp.py --video clip.mp4 --shots calib.json \
      --detector roboflow --stride 8 --detect-workers 8 --live-ball 4
  ```

- **Local YOLO** (default, offline) — COCO `yolo11l.pt`, or a fine-tuned basketball model
  at `models/basketball.pt` (auto-detected). Runs on GPU when a matching CUDA
  torch+torchvision is installed.

### Robustness on messy broadcast pixels

Several steps keep the front-end honest, taking a raw frame to a clean 10-player state
with the handler on the ball-carrier:

- **Referee exclusion** — the detector's `referee` class is dropped; a conservative
  gray-uniform appearance check is a backup (deliberately tuned not to remove real players).
- **Roster gate** — a track is kept only if its *median* court position is an interior
  on-floor player, so anything projected onto the sidelines is removed.
- **Temporal ball tracking** — a constant-velocity motion model coasts the ball's position
  through the frames the detector misses, so every pause has a ball and thus a handler.
- **Handler voting** — possession is mode-filtered over neighbouring frames so it can't flicker.
- **Camera-cut truncation** — each shot's homography is valid only until the first cut, so
  detection stops there; multi-shot builds anchor a fresh calibration in each shot.

### Coverage — multiple camera shots

One calibration covers one camera shot (from its click-time to the next cut). To cover
more, calibrate a frame in each shot (the click-time is saved into the file) and pass
them all; the build processes each shot bounded by the next and merges the timeline:

```bash
python scripts/calibrate.py --video game.mp4 --time 4   --out shot1.json
python scripts/calibrate.py --video game.mp4 --time 18  --out shot2.json
python scripts/calibrate.py --video game.mp4 --time 39  --out shot3.json
python scripts/build_webapp.py --video game.mp4 --shots shot1.json shot2.json shot3.json
```

**Jersey OCR** (EasyOCR, tracklet-voted over the shot) names the players it can read, so
the overlay shows `#7`, `#34`… and the rationale/assistant reference them by number
(back-facing/blurred players correctly stay unnamed).

### Per-frame calibration — the trained keypoint model

A click calibration is solved on **one** frame and carried across the shot by optical
flow, so its accuracy decays with distance from that frame: within-shot drift. The fix
is a model that finds the court landmarks in *every* frame independently.

```bash
# 1. train from the calibrations you already have (no new labelling)
python scripts/train_keypoints.py \
    --pair game.mp4 shot1.json shot2.json shot4.json \
    --pair clip.mp4 early_a.json early_b.json calib.json

# 2. check it against the propagated calibration on a real shot
python scripts/eval_keypoints.py --video clip.mp4 --calib calib.json --render out/kp.png

# 3. build with per-frame homography
python scripts/build_webapp.py --video clip.mp4 --shots early_a.json early_b.json --keypoint-model
```

**The labels come from the calibrations, not from hand-labelling.** A solved calibration
*is* a pixel↔court correspondence, so inverting its `H` places all 26 canonical landmarks
— including the ones nobody clicked. Walking the shot with the flow tracker labels every
frame it passes (gated on the tracker's own reprojection error, so drifted frames are
dropped before they rot the labels), and warping the verified anchor frames by random
perspective transforms manufactures new camera geometry with *exact* labels, because the
warp is the label transform.

Heatmap regression, not coordinate regression: the peak height is a per-point confidence,
which is precisely what `solve_homography` already consumes — so low-confidence landmarks
are dropped by the same RANSAC + reprojection gate the synthetic path uses. The two
methods are used **together**: a frame takes the model's keypoints only when they solve on
their own *and* the resulting homography is geometrically believable, and otherwise falls
back to the propagated calibration.

That second condition is not redundant, and it is the main lesson from building this.
Reprojection error only measures whether a homography agrees with the correspondences it
was fitted to, so a model that hallucinates a *self-consistent* set of landmarks reports a
low error while projecting a court that is nowhere near the real one. `plausible_court_
homography` checks the geometry independently: on the measured click calibrations a foot
of court spans 17–56 px, whereas hallucinated homographies put it at 1.5–5 px — the camera
has effectively been placed in the next building. Scale is sampled only where the court is
actually *visible*, because on a zoomed shot the far baseline sits past the vanishing point
where scale legitimately explodes.

#### Where it stands — measured, not claimed

Trained on **683 auto-labelled frames from 6 usable calibrated shots** across 3 games (two
calibrations were excluded: 4 clicked points fit a homography with zero residual by
construction, so they constrain nothing). Validation holds out a whole camera shot.

| | propagated (today) | per-frame model |
|---|---|---|
| within a trained shot, over 11 s | drifts **0.18 → 16.8 ft** | flat **1.55 → 1.74 ft** |
| frames passing the full acceptance gate | — | 30% in-domain, **0% held-out** |
| held-out keypoint error | — | 53 px median (~5–10 ft of court) |

The honest reading: **the propagated calibration demonstrably drifts** — it is catastrophically
wrong ~7 s after its anchor — and the model is stable where the calibration is not. But at
this data volume the model does not generalise to a camera shot it has never seen, and the
plausibility gate correctly rejects all of its held-out predictions rather than letting them
corrupt the overlay. So `--keypoint-model` is **opt-in and off by default**, and the honest
bottleneck is labelled shots, not architecture or epochs: training loss keeps falling while
held-out error plateaus, which is overfitting to six camera angles. Every additional
calibrated shot feeds the same harvesting pipeline unchanged.

> **The honest boundary.** The reasoning core is validated (86% on simulated, beats
> baselines on ~240 real games); the remaining gap is the *pretrained* front-end on real
> pixels. Automatic *from-scratch* court calibration was attempted and set aside — on
> broadcast wood the court lines mix light and dark strokes with logo/jersey interference,
> so classical line detection can't correspond them reliably. The keypoint model above is
> the real path, but it is trained on the handful of camera shots that were calibrated
> across two games: it is a model for *these broadcasts*, not a general NBA court model,
> and widening it means calibrating more shots (the harvesting code does not change). The
> ~45-second manual click calibration remains the reliable stand-in and the fallback. On
> clean half-court frames the app draws the recommendation; on replays/close-ups/occluded
> frames it (correctly) withholds via the confidence gate.

---

## Layout

```
src/
  ingest/     SportVU parsing, synthetic generator, possession segmentation, real-data labels
  state/      court geometry, the state schema, conformance validator
  value/      candidate actions, sub-models, V(s), scoring, simulator, real-data builder
  perception/ camera, homography, tracking, teams, identity, clock, state-from-CV, overlay,
              + real video: calibrate, video (YOLO/Roboflow), jersey OCR, realvideo adapter,
              + per-frame calibration: keypoints (canonical), keypoint_data (auto-labeller),
                keypoint_model (heatmap net + detector)
  render/     top-down court renderer + video overlay
  llm/        strict rationale schema, context, Claude + offline generators
  app/        pause-to-overlay analyzer, webexport (overlay JSON), chat_backend (Groq/Claude),
              studio/ (setup wizard) + webapp/ (player UI)
tests/        unit + end-to-end tests (96)
scripts/      demo_phase{1..4}.py, demo_realvideo.py, calibrate.py, studio.py, fetch_youtube.py,
              build_webapp.py, serve_webapp.py, benchmark.py, benchmark_charts.py, make_pdf.py,
              train_phase2.py, fetch_real_data.py, build_real_cache.py, train_real.py,
              train_keypoints.py, eval_keypoints.py
configs/      coaching_priors.json
data/ models/ out/   gitignored — real games, cache, weights, artifacts
timeout.pdf   complete plain-language project description
```

## Notes

- **Scope:** half-court offense, ball-handler decisions, one recommendation set per pause.
- **Team-specific layer:** nothing in perception is team-specific; personnel priors,
  playbook, and coaching priors are swappable config surfaces (`configs/coaching_priors.json`).
- **Secrets:** `ROBOFLOW_API_KEY`, `GROQ_API_KEY`, `ANTHROPIC_API_KEY` are read from the
  environment only and never committed.
- **The learned perception front-end** (detection, court keypoints, jersey OCR, pose) needs
  annotated broadcast frames that don't exist publicly; it sits behind documented interfaces
  with a synthetic-broadcast stand-in that makes the geometry/tracking spine real, testable,
  and measurable. Plug real detectors in without touching the rest.
