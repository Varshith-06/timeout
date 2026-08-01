# NBA Play Recommender — Build Roadmap

A pause-and-analyze system for NBA game footage. Freeze the video at any moment, get a ranked set of actions the offense could take, scored in expected points, drawn on screen as arrows and highlights.

Built natively for basketball — no forking a soccer pipeline. Same underlying component libraries (Ultralytics, supervision, OpenCV), but basketball court geometry, basketball keypoints, and basketball action taxonomy from line one.

---

## 0. Scope

### What this is

Given a video timestamp, the system produces:

1. A **court-space state** — all 10 players and the ball in feet, with velocities, over the preceding ~5 seconds.
2. A **ranked list of candidate actions** for the ball handler, each with an expected-points estimate.
3. An **overlay** drawing the top action, with the runner-up available on toggle.
4. A **one-paragraph rationale** in plain English.

### What this is not

- Not a play *predictor* (what will happen). It's an action *evaluator* (what's worth the most).
- Not real-time during live play. It runs on pause. Target latency: under 2 seconds from pause to overlay.
- Not a replacement for coaching judgment. It's a second opinion with a number attached.

### The team-specific layer

Nothing in perception is team-specific. A jump shot looks the same in every arena. Team specificity lives in three config surfaces, all of which can be swapped without retraining anything:

| Surface | What it holds | Where it lives |
|---|---|---|
| Personnel priors | Per-player shooting rates by court zone, finishing rate, turnover rate under pressure | Lookup tables, `data/priors/` |
| Playbook | Annotated sets, indexed by offensive formation | Vector index, `data/playbook/` |
| Coaching priors | Staff preferences, emphasis, "we don't take that shot" | YAML config + LLM prompt |

Build with a generic NBA baseline first. Add a team layer once the baseline works. If you start with the team layer you'll never know whether a bad recommendation came from the model or the config.

### Hard scope decisions to make now

- **Half-court offense only for v1.** Transition and inbounds are separate problems with different state spaces. Filter them out during possession segmentation.
- **Ball-handler decisions only for v1.** Off-ball recommendations ("Player 4 should relocate to the weak-side corner") are a natural v2 and require a different action space.
- **One recommendation set per pause.** No multi-step play trees. The compounding error over a 3-step sequence makes the third step meaningless.

---

## 1. Environment and repo

### Structure

```
nba-play-recommender/
├── data/
│   ├── raw/sportvu/           # downloaded .7z game logs
│   ├── processed/possessions/ # segmented, parquet
│   ├── priors/                # per-player shooting tables
│   ├── playbook/              # annotated sets
│   └── video/                 # broadcast clips (gitignored)
├── src/
│   ├── ingest/                # SportVU parsing, possession segmentation
│   ├── state/                 # state schema + builders
│   ├── value/                 # value model, candidate actions
│   ├── perception/            # detection, tracking, homography, pose, OCR
│   ├── render/                # court renderer, video overlay
│   ├── llm/                   # rationale generation
│   └── app/                   # UI
├── models/                    # trained weights (gitignored, DVC or S3)
├── notebooks/                 # exploration only, nothing load-bearing
├── tests/
└── configs/
```

### Stack

- Python 3.11
- `polars` or `pandas` for tracking data (polars is meaningfully faster on the 2.6M-row-per-game scale)
- `pytorch` for the value model
- `ultralytics` for detection and keypoints
- `supervision` for tracking, annotation, and geometry utilities
- `opencv-python` for homography and video I/O
- `lightgbm` for the tabular sub-models
- `fastapi` + a web frontend for the app layer

### Hardware

- Phase 1–2 run on CPU. A laptop is fine.
- Phase 3 needs a GPU. A single 4090 or a rented A10G is enough for fine-tuning YOLO on a few thousand frames.
- Storage: one SportVU game unpacks to roughly 2.6 million rows. Budget ~50 GB if you pull a meaningful slice of the season.

---

## 2. Phase 1 — Reasoning layer on perfect tracking (weeks 1–3)

**Goal:** prove the recommendation logic produces basketball-sane output before you write a single line of computer vision.

This is the most important sequencing decision in the whole project. If the reasoning layer turns out to be conceptually empty, you'll find out in three weeks with no GPU spend instead of in four months.

### 2.1 Acquire the data

The 2015-16 season is the last with publicly released NBA tracking. Sources, in order of convenience:

- HuggingFace: `dcayton/nba_tracking_data_15_16` — SportVU merged with play-by-play, already cleaned.
- GitHub: `sealneaward/nba-movement-data` — mirror of the original `neilmj/BasketballData` logs, plus conversion scripts (full-court to half-court, corrected shot timestamps in `data/shots/fixed_shots.csv`).
- GitHub: `linouk23/NBA-Player-Movements` for a reference visualizer to sanity-check your parsing.

Coverage runs roughly late October 2015 through 23 January 2016. Not a full season — plan around it.

Two things to internalize about this data:

- **It's a decade stale.** Different rosters, different league-wide shot distribution (three-point rate has climbed substantially since). Phase 1 is about validating *logic*, not shipping a model you'll deploy. You'll recalibrate personnel priors from current public shot-chart data later.
- **The tracking for every season since is private.** If you end up working with an actual team, they have a live Second Spectrum / Sportradar / Hawk-Eye feed. That feed makes Phase 3 unnecessary. Ask before you build.

### 2.2 Parse the format

Each game is one JSON file. Structure:

```
{
  "gameid": "0021500431",
  "gamedate": "2015-12-23",
  "events": [
    {
      "eventId": "303",
      "home":    { "teamid": ..., "name": ..., "abbreviation": ...,
                   "players": [ {"playerid", "firstname", "lastname", "jersey", "position"} ] },
      "visitor": { ... },
      "moments": [
        [ quarter, unix_ms, game_clock, shot_clock, null,
          [ [team_id, player_id, x, y, z], ... ] ]
      ]
    }
  ]
}
```

- Sampled at 25 Hz.
- Each moment's position array has 11 entries: 10 players plus the ball.
- The ball is `team_id = -1, player_id = -1`. Its `z` is height in feet; players' `z` is radius and can be ignored.
- Court coordinates are in feet: `x ∈ [0, 94]`, `y ∈ [0, 50]`, origin at a baseline corner.

**Known landmines, all of which will bite you:**

- Events overlap. Coordinates for event N often start before and end after the labeled event, bleeding into event N+1. De-duplicate on `(quarter, game_clock)` and treat the event boundaries as advisory.
- Some moments have no coordinates at all. Rare enough to drop; don't try to interpolate across them.
- `shot_clock` is sometimes `null` (typically when the clock is off). Forward-fill within a possession, and flag possessions where it's null for more than ~10 frames.
- Player IDs are stable across the season; team IDs are stable. Jersey numbers are in the event metadata, which is how you'll later bridge to OCR output.

### 2.3 Normalize geometry

Write this once, in `src/state/court.py`, and never hand-roll a coordinate again.

NBA court constants, all in feet:

```
COURT_LENGTH   = 94.0
COURT_WIDTH    = 50.0
BASKET_LEFT    = (5.25, 25.0)     # rim center, 63" from baseline
BASKET_RIGHT   = (88.75, 25.0)
BACKBOARD_INSET = 4.0             # from baseline
PAINT_WIDTH    = 16.0             # y ∈ [17, 33]
FT_LINE_DIST   = 19.0             # from baseline
FT_CIRCLE_R    = 6.0
THREE_ARC_R    = 23.75            # from rim center
THREE_CORNER_Y = 3.0, 47.0        # straight segments
CORNER_BREAK_X = 14.0             # from baseline, where arc meets corner line
RESTRICTED_R   = 4.0
CENTER_CIRCLE  = (47.0, 25.0), r = 6.0
```

Required transforms:

1. **Half-court flip.** Mirror every possession so the attacking basket is always at `BASKET_LEFT`. This doubles your effective sample size and halves the model's job. `sealneaward/nba-movement-data` ships a script for this; verify it against a few known plays rather than trusting it.
2. **Basket-relative polar coordinates.** For each player: distance to rim, angle from the baseline. Most shooting behavior is far cleaner in polar than Cartesian.
3. **Zone assignment.** Bucket the court into the standard shot zones (restricted area, paint non-RA, mid-range left/center/right, corner three left/right, above-break three left/center/right). You'll key personnel priors off these.

### 2.4 Segment possessions

You need clean possession boundaries or every downstream number is contaminated.

A possession starts on: made-basket inbound, defensive rebound, steal, or turnover recovery. It ends on: shot attempt (made or missed and rebounded by the defense), turnover, or shot-clock violation. Merge offensive rebounds into the same possession — a putback is not a new possession.

Deriving these:

- Join to play-by-play on `(quarter, game_clock)` with a ±0.5s tolerance. Play-by-play gives you the event types directly and is far more reliable than inferring from coordinates.
- Cross-check against the shot clock: a reset from a low value to 24 (or 14 after an offensive rebound) is a strong possession-boundary signal.
- Filter to half-court: drop any possession where the ball crosses half-court less than ~4 seconds before the terminal event. Those are transition and you're excluding them by scope.

**Ball handler assignment**, per frame:

```
handler = argmin_p  distance(p, ball)  subject to:
    distance(p, ball) < 4.0 ft
    ball.z < 9.0 ft                          # not a shot or a lob in flight
    |velocity(p) - velocity(ball)| < 6 ft/s  # ball is moving with the player
```

Then smooth: a handler must hold for ≥5 consecutive frames (0.2s) to count. Without smoothing you'll get handler assignment flickering during every hand-off. Frames with no valid handler are "ball in flight" — a pass or a shot — and are their own state category.

**Deliverable for 2.4:** a parquet table, one row per frame, columns: `possession_id, frame_idx, game_clock, shot_clock, handler_player_id, ball_in_flight, [10 × (player_id, x, y, vx, vy)], ball_x, ball_y, ball_z, terminal_event, points_scored`.

### 2.5 Build the state schema

This is the JSON object you originally wanted an LLM to produce. It is produced by code, deterministically, from tracking data. Every field is either measured or computed — none is inferred by a language model.

```json
{
  "timestamp": {"quarter": 3, "game_clock": 412.4, "shot_clock": 11.2},
  "possession_id": "0021500431_p0417",
  "offense_team_id": 1610612748,
  "attacking_basket": [5.25, 25.0],
  "ball": {"x": 24.1, "y": 18.7, "z": 4.2, "vx": -3.1, "vy": 1.4, "in_flight": false},
  "players": [
    {
      "player_id": 2547,
      "jersey": 13,
      "team_id": 1610612748,
      "side": "offense",
      "x": 24.9, "y": 18.2,
      "vx": -3.0, "vy": 1.5, "speed": 3.35,
      "orientation_deg": 212.0,
      "has_ball": true,
      "dist_to_rim": 20.4,
      "angle_to_rim_deg": 161.0,
      "zone": "mid_range_left",
      "nearest_defender": {"player_id": 201142, "dist": 3.1, "angle_deg": 44.0},
      "defender_pressure": 0.71,
      "seconds_since_touch": 1.8
    }
  ],
  "context": {
    "n_players_observed": 10,
    "spacing_area_sqft": 412.0,
    "defense_scheme": "man",
    "active_screen": {"screener_id": 202710, "phase": "approach"},
    "confidence": 1.0
  }
}
```

Field notes:

- `orientation_deg` — in Phase 1 this is unavailable (SportVU has no pose). Set it to velocity heading and mark it as such. Phase 3 replaces it with a real torso normal.
- `defender_pressure` — a scalar you define, e.g. `exp(-dist / 4.0)` weighted by whether the defender is between the player and the rim. Keep the formula in one place; you'll tune it.
- `spacing_area_sqft` — convex hull area of the five offensive players. A well-established spacing proxy and a useful sanity metric.
- `defense_scheme` — rule-based for v1: cluster defender-to-nearest-offensive-player distances. Tight and consistent pairing means man; defenders anchored to court zones regardless of offensive movement means zone. Don't over-invest here; NBA defenses are overwhelmingly man with switches.
- `confidence` — always 1.0 in Phase 1. In Phase 4 this carries perception uncertainty and gates whether you show a recommendation at all.

### 2.6 Enumerate candidate actions

For a given state, generate the legal action set. Keep it small and unambiguous.

```
PASS_TO(teammate_id)        × 4     — one per teammate
DRIVE(direction)            × 3     — left, right, middle
SHOOT                       × 1
SCREEN_WITH(teammate_id)    × 4     — request a ball screen
RESET                       × 1     — dribble out, re-run the set
```

Thirteen candidates maximum. Prune illegal ones before scoring: no `SHOOT` beyond ~32 ft unless shot clock < 2; no `DRIVE(left)` if the player is already on the left sideline; no `SCREEN_WITH` a teammate more than ~25 ft away.

Each candidate is a structured object, never a coordinate:

```json
{"action": "PASS_TO", "actor": 2547, "target": 201609, "id": "a3"}
```

The renderer resolves it to geometry. The model never emits pixels or feet.

### 2.7 Build the court renderer

Matplotlib, top-down, 94×50. Draw the court lines from the constants in 2.3. Plot players as circles colored by team, the ball as a smaller marker, velocity as a short arrow. Overlay candidate actions as dashed arrows with their scores as labels.

You will look at this thousands of times. Make it good early. Add a frame slider and the ability to step through a possession.

### 2.8 The week-3 gate

Freeze five possessions. For each, print the state and the candidate list with placeholder uniform scores. Sit down with someone who actually coaches basketball — a high-school assistant is enough — and ask two questions:

1. Is the candidate list complete? (Did we miss an obvious option?)
2. Is anything on the list absurd?

If the answer to (1) is "you're missing the most important thing," fix the action space now. Everything downstream is built on it.

---

## 3. Phase 2 — The value model (weeks 4–8)

**Goal:** a calibrated function that maps a state to expected points, and a way to score each candidate action with it.

This is the intellectual core. It's also where the published research is, and you should read it before writing code: Cervone, D'Amour, Bornn, and Goldsberry, *A Multiresolution Stochastic Process Model for Predicting Basketball Possession Outcomes* (arXiv:1408.0777). The `dcervone/EPVDemo` repo has runnable code and a walkthrough PDF. It's in R; you're not going to port it, but the decomposition is the thing to steal.

### 3.1 The architecture: state value plus one-step lookahead

The clean, buildable version:

**Train a state-value function** `V(s) → expected points`, on every frame of every possession, with the target being the actual points that possession ended up scoring (0, 2, or 3; handle free throws as their expected value from the and-one or shooting-foul outcome).

**Score a candidate action** by predicting the state it produces and evaluating `V` there, discounted by the probability the action succeeds:

```
Q(s, a) = P(success | s, a) · V(s'_a)  +  (1 − P(success | s, a)) · V_turnover
```

where `V_turnover ≈ 0` plus the (negative) transition value you're conceding. For `SHOOT`, the special case collapses to `P(make) × point_value`.

This is deliberately a one-step lookahead. Deeper trees compound prediction error until the leaves are noise.

### 3.2 The sub-models

Build them in this order. Each is independently testable.

**(a) Shot make probability** — `P(make | shooter, x, y, closest_defender_dist, defender_angle, shot_clock, off_dribble, catch_and_shoot)`

- LightGBM, binary target. Roughly 60k–100k shots in the available data.
- Player identity via a per-player prior blended with league average by empirical Bayes — you will not have enough shots per player per zone to fit individual effects directly. Shrink hard.
- Output is calibrated probability. Check with a reliability diagram, not just AUC.

**(b) Pass completion probability** — `P(complete | passer, receiver, all 10 positions)`

Expect this to be your hardest sub-model. Cervone's team said the same: fitting how likely a player is to pass to a teammate given both their spatial positions, per-player to capture individual tendencies, was the hardest part of EPV.

- Features: pass distance, number of defenders whose perpendicular distance to the pass line is under ~3 ft, the receiver's separation from his defender, whether the passer is facing the receiver (needs orientation — in Phase 1, use velocity heading).
- LightGBM. Target from tracking: did the ball reach the intended receiver.
- Getting the *intended* receiver for completed passes is easy; for turnovers it's ambiguous. Label deflections and steals conservatively, or train only on the completion side and calibrate the turnover rate separately.

**(c) Drive success** — `P(reach a rim-zone shot | driver, defenders, direction)`

- Simpler: does the ball handler get within 6 ft of the rim within 3 seconds of starting the drive.
- Features: help-defender positions on the drive side, the primary defender's stance and lateral position, the driver's speed.

**(d) The state-value function** `V(s)`

- Architecture: a small permutation-invariant network over 11 entities. Each entity gets a feature vector (position relative to rim, velocity, one-hot for offense/defense/ball, a learned player embedding), passed through a shared MLP, pooled by attention or mean, concatenated with global features (shot clock, spacing area), then a head to a scalar.
- Roughly 200k–500k parameters. This does not need to be large.
- Train with the possession's realized points as the target, MSE loss. Optionally use TD(λ) bootstrapping for smoother values through the possession.
- Sample weight: down-weight frames in the first two seconds of a possession, where the value is near-constant and uninformative.

### 3.3 The step you cannot skip: noise augmentation

Your value model is being trained on near-perfect optical tracking. In Phase 4 you will feed it output from a broadcast CV pipeline that has position error of several feet, occasionally loses players entirely, and mixes up identities. A model trained on clean data and served noisy data degrades in ways that don't show up in your validation metrics.

Fix it during training, not after:

- Add Gaussian jitter to positions, σ ≈ 1.5–2.5 ft. Match this to whatever reprojection error you actually measure in Phase 3, then retrain.
- Randomly drop 1–3 players from the state entirely (broadcast frames routinely miss them) and train the model to handle a variable entity count. The permutation-invariant architecture in 3.2(d) gives you this for free — make sure you exercise it.
- Randomly swap two same-team player identities at ~2% rate, matching expected re-ID error.
- Add lag jitter to velocity estimates.

Train two checkpoints: clean and augmented. Compare them on clean validation data. If the augmented model is meaningfully worse on clean data, you've over-augmented.

### 3.4 Evaluation for Phase 2

- **Calibration.** Bin predicted EPV into deciles; plot mean prediction against mean realized points. The line should be diagonal. This matters more than any accuracy number, because a coach comparing 1.12 against 0.94 needs those to mean something.
- **Brier score and log loss** on each sub-model.
- **EPV trajectory sanity.** Plot the EPV curve through a possession. It should rise when a player gets open at the rim and fall when the shot clock runs down under pressure — this is the plot from the Cervone paper, and if yours doesn't look like theirs, something is wrong.
- **Ordering test.** On possessions where the offense took an obviously bad shot (contested, long two, early clock), your model should rank `SHOOT` below at least two alternatives. Hand-label 50 such possessions and measure.

### 3.5 The week-8 gate

Same coach, same room. Show 20 frozen possessions with the full ranked candidate list and scores. Ask them to mark each recommendation as: right, defensible, or wrong.

Target before you touch computer vision: **at least 70% right-or-defensible, and under 10% wrong.** If you're above 20% wrong, the value model isn't ready and adding a noisy perception front-end will only obscure why.

---

## 4. Phase 3 — Perception (weeks 9–18)

**Goal:** turn a broadcast clip into the same court-space state schema you built in Phase 1, at acceptable fidelity.

Everything here is built basketball-native. The court keypoint set, the detection classes, the annotation schema — all defined for a 94×50 court from the start.

### 4.1 Assemble the video dataset

- Source: full-game NBA broadcasts. For a personal or research project this is fine under fair use; for anything commercial or for actual team deployment you need footage rights. Address this before it's a problem, not after.
- Sample frames at 1 fps from a diverse set: at least 8 different arenas (court paint schemes and lighting differ enormously), day and night games, national and local broadcasts.
- **Annotation budget:** 2,000–3,000 frames for detection, 800–1,200 frames for court keypoints. Use Roboflow or CVAT. This is 3–5 days of real work; don't underestimate it.
- Hold out two entire games — never sampled from — as your test set. Frame-level splits leak badly because adjacent frames are nearly identical.

### 4.2 Detection

Classes: `player`, `ball`, `referee`, `rim`.

- Fine-tune a YOLO detection model (Ultralytics) from COCO weights. Players are close enough to the `person` class that you'll converge fast.
- **The ball is the hard part.** It is small, fast, motion-blurred, and occluded roughly half the time. Mitigations:
  - Train a **separate, dedicated ball model** at higher input resolution. Don't make one model handle both a 200px player and a 12px ball.
  - Use sliced inference (SAHI-style tiling) on the region around the last known ball position.
  - Accept that per-frame recall will be 60–80% and rely on temporal interpolation. A Kalman filter over ball position with a ballistic motion model fills gaps well.
  - `rim` detection gives you a strong prior on where shots terminate and helps disambiguate ball-in-flight.
- **Referees must be excluded** or they'll contaminate team clustering. The `referee` class handles this; striped jerseys are visually distinctive enough that this trains easily.

Target metrics on held-out games: player mAP@50 > 0.90, ball mAP@50 > 0.55.

### 4.3 Tracking

- ByteTrack via `supervision`, with Kalman filtering. ByteTrack's low-confidence association step is exactly what you want for basketball, where players are constantly partially occluded by each other.
- **Camera cut detection is mandatory.** Broadcasts cut to replays, close-ups, and crowd shots constantly. Detect cuts with a frame-to-frame color histogram distance threshold, and reset the tracker on every cut. Without this, track IDs persist across a replay and your state becomes nonsense.
- **Re-identification across occlusion.** Basketball has severe mutual occlusion during screens and post play. Add an appearance embedding (a small re-ID model, or SigLIP crops) to the association step so a player emerging from a screen re-attaches to his old track.

Target: IDF1 > 0.75 on continuous half-court segments. Expect it to be much lower across cuts, which is why you reset instead.

### 4.4 Team assignment

- Crop each player detection to the torso region (top ~40% of the box, center 60% horizontally — this avoids the floor and the shorts).
- Embed crops with SigLIP.
- Reduce with UMAP, cluster with KMeans, k=2.
- **Cluster per-possession, not per-frame.** Team assignment should be stable over a tracklet — take a majority vote across all frames of the tracklet.
- Refs are already filtered by the detector. Also filter by track length: anyone tracked for under ~15 frames is probably a coach, a photographer, or noise.

Target: >97% team accuracy on tracklets longer than 1 second. Below that, everything downstream breaks.

### 4.5 Player identity

Team assignment gives you five bodies per side. You need names to look up personnel priors.

- Crop the jersey number region (upper back or chest, depending on camera angle).
- Upscale 4× with a lightweight super-resolution model — jersey numbers are frequently under 20px tall in broadcast.
- Recognize with a text recognition model (PARSeq or TrOCR fine-tuned on jersey crops), or train a direct 0–99 classifier. The classifier is more robust because the label space is closed.
- **Vote across the entire tracklet.** Any individual frame is unreliable; 60 frames of the same player produce a confident answer.
- Map `(team_id, jersey_number)` to a player via the game's roster, pulled from the NBA stats API.
- Fall back gracefully: if identity is uncertain, use position-average priors instead of player-specific ones and drop `confidence` in the state object.

### 4.6 Court keypoints and homography

This is the load-bearing component and it will consume the most time of anything in Phase 3.

**Keypoint set** — define ~26 points, symmetric across the court so a single model covers both halves:

| Group | Points |
|---|---|
| Court corners | 4 baseline/sideline intersections |
| Half-court | 2 sideline/midline intersections, 2 center-circle top/bottom |
| Paint (each end) | 4 corners of the 16-ft lane, ×2 ends = 8 |
| Free-throw circle (each end) | top and bottom of the circle, ×2 = 4 |
| Three-point corner breaks (each end) | 2 points where the arc meets the straight corner line, ×2 = 4 |
| Three-point arc apex (each end) | 1 top-of-key point, ×2 = 2 |

Not all are visible in any given frame — that's expected and fine.

**Model:** a YOLO keypoint-detection model trained to output all 26 with per-point confidence. Annotate 800–1,200 frames. Include frames where the camera is zoomed tight and only a handful of points are visible; those are the ones that break naive implementations.

**Solving the homography:**

```python
# keep only high-confidence detected points
src = detected_pixel_coords[conf > 0.5]
dst = CANONICAL_COURT_COORDS[conf > 0.5]

if len(src) < 4:
    return None  # cannot solve — mark frame low-confidence

H, mask = cv2.findHomography(src, dst, cv2.RANSAC, ransacReprojThreshold=5.0)
```

Then, and this is not optional:

1. **Reprojection error check.** Project the detected points through `H` and measure error in feet against canonical positions. Reject any frame with median error above ~2 ft. A bad homography silently poisons the state; you want it to fail loudly.
2. **Temporal smoothing.** Broadcast cameras pan and zoom smoothly, so `H` should change smoothly. Decompose to camera parameters (pan, tilt, zoom, position) if you can, smooth those, and recompose — smoothing the raw 3×3 matrix elements produces geometric artifacts. A simpler alternative that works acceptably: smooth the projected positions of the four court corners, then re-solve.
3. **Fill gaps with optical flow.** When keypoint detection fails on a frame but succeeded on the previous one, propagate `H` using sparse optical flow on court-line features (Lucas-Kanade on Shi-Tomasi corners, masked to the court region).

The earlier generation of this work hard-coded three separate homography transforms for court regions and manually specified which frames used which. You are avoiding that entirely by using a learned keypoint model — but understand that's the failure mode you're engineering around.

**Applying it:** project each player's **foot position** (bottom-center of the bounding box), not the box center. The homography maps the court *plane*; a player's torso is five feet above it and will project badly.

Target: median reprojection error under 1.5 ft on held-out games, with a valid homography on >85% of half-court frames.

### 4.7 Body orientation

- Run a pose model (RTMPose or ViTPose) on each player crop.
- Take left/right shoulder and left/right hip keypoints. The vector perpendicular to the shoulder line, projected onto the court plane through your homography, gives torso facing.
- Hips are more reliable than shoulders for actual body orientation; shoulders swing during a shot or a pass. Use hips as primary, shoulders as fallback.
- Smooth over ~5 frames. Raw per-frame orientation is jittery.
- This is what finally replaces the velocity-heading placeholder from Phase 1. Once it's available, retrain the pass-completion sub-model with real orientation as a feature — it should improve noticeably, since whether the passer is facing the receiver matters a great deal.

### 4.8 Clock extraction

Shot clock is a required state field and you can read it directly off the broadcast.

- The scoreboard graphic sits in a fixed screen region per broadcast package. Detect the region once per game (template match or a small detector), then crop the same ROI every frame.
- OCR with PaddleOCR or EasyOCR restricted to a digit charset. Seven-segment renderings are unusually easy for these models.
- **Validate temporally.** The shot clock decreases monotonically at 1.0/sec and resets to 24 or 14. Any reading that violates this is an OCR error — discard and interpolate. This constraint makes clock extraction near-perfect in practice.
- Game clock and score come from the same ROI and are useful for the LLM context layer.

### 4.9 The occlusion problem

Broadcast footage frequently shows fewer than all ten players. Your value model needs full defensive positioning.

Options, in order of preference:

1. **Get better footage.** Fixed wide or full-court cameras. This is why arenas mount six cameras high above the court for optical tracking. If you have any relationship with a team, ask for their all-22 equivalent.
2. **Restrict to tight half-court sets** where the camera framing naturally includes all ten. Filter at the possession level.
3. **Degrade gracefully.** Track `n_players_observed`. Below 10, drop `confidence`. Below 8, refuse to show a recommendation and tell the user why. This is the option that keeps the product honest, and it's why you trained with player dropout in 3.3.

Never silently impute a missing defender's position. A hallucinated help defender in the wrong place inverts the recommendation.

---

## 5. Phase 4 — Integration (weeks 19–22)

### 5.1 Wire perception into the state schema

The output of Phase 3 must be byte-identical in shape to the output of Phase 1. Write a conformance test: run both a SportVU possession and a CV possession through the same schema validator and the same value model.

Populate `confidence` as a composite:

```
confidence = w1 · (n_players_observed / 10)
           + w2 · homography_quality
           + w3 · mean(identity_confidence)
           + w4 · ball_detection_recall_over_window
```

Gate the UI on it. Below a threshold you tune empirically, show the top-down radar view but withhold the recommendation.

### 5.2 Close the domain gap

Measure your actual perception error, then retrain the value model with augmentation matched to it.

To measure it without ground-truth tracking on modern footage: run your CV pipeline over broadcast video **of a 2015-16 game you have SportVU for**. This gives you paired data — CV output and optical tracking of the same moment. Align them on the game clock, and you get a direct measurement of position error, identity error rate, and player-miss rate.

This is a genuinely valuable trick and it's only possible because the public tracking data and the corresponding broadcasts both exist. Spend the time here.

Feed the measured error distribution back into 3.3 and retrain. Re-run the Phase 2 calibration check on CV-derived states.

### 5.3 The renderer

Two views, side by side.

**Top-down radar.** The Phase 1 matplotlib renderer, upgraded. Players as numbered circles, ball, recommended action as a solid arrow, runner-up as a dashed arrow, target space as a shaded region.

**Video overlay.** Project court coordinates *back* through `H⁻¹` into pixel space and draw on the paused frame.

```python
H_inv = np.linalg.inv(H)
pixel_pt = cv2.perspectiveTransform(court_pt.reshape(1,1,2), H_inv)
```

Drawing primitives, resolved from the action object — never from model-emitted coordinates:

| Action | Overlay |
|---|---|
| `PASS_TO` | Arrow from passer's tracked position to receiver's tracked position |
| `DRIVE` | Arrow from handler toward the rim, curved to the specified side |
| `SHOOT` | Ring highlight on the shooter, arc to the rim |
| `SCREEN_WITH` | Bracket connecting screener and handler, arrow showing the screen angle |
| Defensive context | Translucent shading on the defender who is the reason for the recommendation |

Every endpoint of every drawn element comes from the tracker. The model chose *which* players; the tracker supplies *where they are*. This is the single design rule that prevents arrows drifting off bodies.

### 5.4 The LLM layer

Input: the state object, the ranked candidate list with scores, any matched playbook sets, and the coaching-priors config.

Output, under a strict schema:

```json
{
  "headline": "Skip pass to the weak-side corner",
  "rationale": "The low man has committed to the drive, leaving the corner shooter with 6 feet of separation. That's a 1.18 EPV look against 0.94 for the pull-up, and it's a shot he takes at above his season average from that spot.",
  "risk": "The pass crosses two defenders; completion probability is 0.81.",
  "alternative": "If the help recovers, the reset to the top keeps 9 seconds on the clock."
}
```

Constraints, enforced in the schema and validated on parse:

- **No coordinates in the output.** Not pixels, not feet. If the schema contains a numeric position field, remove it.
- Player references by `player_id` only, resolved to names by your code.
- All numeric claims are passed *in* from the value model and echoed, never generated. If the LLM produces a number not present in the input, reject and regenerate.

**Consider whether this runs in the live path.** A coach who pauses and waits four seconds for a token stream has a worse experience than an instant overlay with a one-line label. Recommended: render the overlay immediately from the value model output, and make the prose rationale a second click.

### 5.5 The app

- Video player with frame-accurate seek, spacebar to pause.
- On pause: seek back 5 seconds, run the clip through perception, build state, score actions, render. Budget: under 2 seconds total.
- Precompute aggressively. If the user is watching linearly, run perception on a rolling buffer ahead of the playhead so a pause is nearly instant.
- Controls: toggle runner-up action, toggle radar view, "explain this" button, "why not X?" for any other candidate.
- Cache by `(game_id, timestamp)` — coaches re-watch the same moments.

---

## 6. Evaluation

Define these before you build, and run them continuously.

### Perception

| Metric | Target |
|---|---|
| Player detection mAP@50 | > 0.90 |
| Ball detection mAP@50 | > 0.55 |
| Tracking IDF1 (within a continuous shot) | > 0.75 |
| Team assignment accuracy (tracklets > 1s) | > 0.97 |
| Jersey identity accuracy (tracklets > 2s) | > 0.90 |
| Homography median reprojection error | < 1.5 ft |
| Frames with a valid homography | > 0.85 |
| Shot clock OCR accuracy (after temporal validation) | > 0.99 |

### Value model

- Calibration curve, predicted EPV vs realized points, by decile. Diagonal.
- Brier score on each sub-model against a league-average baseline.
- EPV trajectories that visually match the published curves for known possessions.

### Recommendation quality

This is the one that decides whether anyone uses it, and it's the one people skip.

1. **Backtest.** On held-out possessions, compare the realized EPV change when the offense's actual action matched your top-1 recommendation against when it didn't. If your recommendations are good, matched possessions should show a higher mean EPV delta. This is observational and confounded — a coach's actual choice isn't random — but a *negative* result here is a hard stop.
2. **Blind coach rating.** 100 frozen possessions. For each, present your top recommendation and a baseline recommendation (e.g., "always pass to the most open player") in random order, unlabeled. Ask the coach to pick the better one. You need to beat the baseline by a wide, obvious margin.
3. **Adversarial review.** Ask a coach to find possessions where the system is confidently wrong. These are worth more than a hundred cases where it's right, because they tell you which state features you're missing.

---

## 7. Risk register

| Risk | Likelihood | Mitigation |
|---|---|---|
| Homography unreliable on tight-zoom broadcast frames | High | Optical-flow gap filling; reject and skip frames rather than serve bad states |
| Ball detection recall too low to determine possession | High | Dedicated high-res ball model; Kalman interpolation; possession from player proximity when ball is lost |
| Fewer than 10 players visible in most frames | High | Confidence gating; refuse to recommend below 8; pursue fixed-camera footage |
| Value model trained on 2015-16 doesn't transfer to current NBA | Medium | Recalibrate personnel priors from current public shot data; validate calibration on modern possessions via the paired trick in 5.2 |
| Coaches find recommendations obvious or useless | Medium | The week-8 gate exists specifically to catch this before CV spend |
| Footage rights block deployment | Medium | Resolve early; team-provided footage sidesteps it entirely |
| Identity confusion between similar-build teammates | Medium | Tracklet-level voting; position-average priors as fallback |
| Scope creep into off-ball recommendations | Medium | Explicitly deferred to v2 in section 0 |

---

## 8. Timeline summary

| Weeks | Phase | Exit criterion |
|---|---|---|
| 1–3 | Reasoning layer on SportVU | Coach confirms the action space is complete |
| 4–8 | Value model | ≥70% right-or-defensible, <10% wrong on 20 possessions |
| 9–18 | Perception | All Phase 3 metric targets met on two held-out games |
| 19–22 | Integration and UI | End-to-end pause-to-overlay under 2 seconds |
| 23+ | Evaluation, team layer, iteration | Beats the naive baseline in blind coach rating |

Roughly six months at a serious part-time pace, four at full time. The two schedule risks that actually bite are annotation (section 4.1) and homography (section 4.6). Both are grindy rather than uncertain, which means they're plannable — budget generously and they won't surprise you.

---

## 9. The shortcut worth asking about first

Everything in Phase 3 exists to reconstruct what a team already receives every night. NBA teams get an optical tracking feed with all ten players and the ball in court coordinates, already solved. If there is any path to a conversation with a team's analytics staff, have it before week 9. Phase 3 is ten weeks of work that a data-sharing agreement makes unnecessary, and Phases 1, 2, 4, and 5 — the parts that are actually novel — get you a working product on their feed.
