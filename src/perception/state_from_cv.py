"""Assemble perception output into the Phase 1 State schema (roadmap 5.1, 4.9).

The whole point of Phase 3: produce a state object *byte-identical in shape* to
the one Phase 1 builds from SportVU, so the Phase 2 value model runs on it
unchanged. This module runs the full pipeline over a broadcast clip —
homography (temporally smoothed), foot-position projection, team clustering,
jersey identity — recovers court-space tracking with velocities, and builds a
State via the same :func:`src.state.schema.build_state`.

It also computes the composite ``confidence`` (5.1) and gates on it: below a
threshold we withhold the recommendation rather than serve a state we do not
trust. Missing defenders are never imputed (4.9) — n_players simply drops and
confidence with it.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from src.ingest.possessions import Frame, PlayerFrame
from src.state import court
from src.state.court import on_court
from src.state.schema import State, build_state
from src.perception.homography import (TemporalHomography, project_feet_to_court,
                                       solve_homography)
from src.perception.identity import assign_identities
from src.perception.teams import assign_teams
from src.perception.tracking import run_tracker

# Composite-confidence weights (roadmap 5.1). Sum to 1.
W_PLAYERS, W_HOMOG, W_IDENTITY, W_BALL = 0.35, 0.35, 0.15, 0.15
CONFIDENCE_GATE = 0.6      # below this, withhold the recommendation (4.9)
MIN_PLAYERS_SHOW = 8       # below this, refuse regardless (4.9)


@dataclass
class Recovery:
    """Per-frame recovered court tracking + the models needed to build states."""
    frames_court: list       # per frame: {track_id: (x, y)}
    velocities: list         # per frame: {track_id: (vx, vy)}
    homographies: list       # per frame: H (pixel->court) or None
    ball_court: list         # per frame: (x, y) or None
    ball_possessor: list     # per frame: track_id nearest the ball in PIXEL space
    homog_ok: list           # per frame: bool
    tracklets: list
    team_labels: dict
    identities: dict
    stride: int
    diagnostics: dict = field(default_factory=dict)


def recover_tracking(clip, roster_rows, stride: int = 5) -> Recovery:
    """Run detection->tracking->homography->teams->identity over a clip."""
    tracker = run_tracker(clip.frames)
    tracklets = tracker.all_tracklets(min_len=3)
    team_labels = assign_teams(tracklets)
    identities = assign_identities(tracklets, team_labels, roster_rows)

    # Index each track's detection per frame.
    per_frame_track_det: list[dict] = [dict() for _ in clip.frames]
    for t in tracklets:
        for fidx, det in t.history:
            if 0 <= fidx < len(per_frame_track_det):
                per_frame_track_det[fidx][t.track_id] = det

    temporal = TemporalHomography(window=5)
    frames_court, homographies, ball_court, homog_ok, ball_possessor = [], [], [], [], []
    dt = stride / 25.0

    for i, bf in enumerate(clip.frames):
        res = solve_homography(bf.kp_pixels, bf.kp_court, bf.kp_conf)
        res = temporal.update(res)
        H = res.H
        homographies.append(H)
        homog_ok.append(bool(res.ok))

        tdets = per_frame_track_det[i]
        balls = bf.detections.by_class("ball")

        # Ball possessor by PIXEL proximity (avoids the off-plane projection error
        # that corrupts court-space ball distance). Nearest player box to the ball.
        possessor = None
        if balls and tdets:
            bx, by = balls[0].center
            possessor = min(
                tdets, key=lambda t: (tdets[t].center[0] - bx) ** 2 + (tdets[t].center[1] - by) ** 2
            )
        ball_possessor.append(possessor)

        court = {}
        if H is not None:
            if tdets:
                tids = list(tdets)
                boxes = np.array([tdets[t].bbox for t in tids])
                pts = project_feet_to_court(H, boxes)
                for tid, xy in zip(tids, pts):
                    # Drop off-court detections (crowd/bench/refs project outside
                    # the lines) so only real on-floor players enter the state.
                    if on_court(float(xy[0]), float(xy[1])):
                        court[tid] = (float(xy[0]), float(xy[1]))
            if balls:
                bp = project_feet_to_court(H, np.array([balls[0].bbox]))[0]
                ball_court.append((float(bp[0]), float(bp[1])))
            else:
                ball_court.append(None)
        else:
            ball_court.append(None)
        frames_court.append(court)

    # Velocities via finite difference of court positions across frames.
    velocities = [dict() for _ in clip.frames]
    for i in range(1, len(frames_court)):
        for tid, (x, y) in frames_court[i].items():
            if tid in frames_court[i - 1]:
                px, py = frames_court[i - 1][tid]
                velocities[i][tid] = ((x - px) / dt, (y - py) / dt)

    diagnostics = {
        "homog_valid_rate": float(np.mean(homog_ok)) if homog_ok else 0.0,
        "mean_players": float(np.mean([len(c) for c in frames_court])) if frames_court else 0.0,
        "ball_recall": float(np.mean([b is not None for b in ball_court])) if ball_court else 0.0,
        "n_tracklets": len(tracklets),
    }
    return Recovery(frames_court, velocities, homographies, ball_court, ball_possessor,
                    homog_ok, tracklets, team_labels, identities, stride, diagnostics)


def _fallback_pid(tid: int) -> int:
    """Stable synthetic player_id for an unidentified track (negative -> 'unknown')."""
    return -(1000 + tid)


def _track_team_id(recovery, tid):
    ident = recovery.identities.get(tid)
    if ident is None:
        return None
    return ident.team_id if ident.team_id is not None else ident.team_label


def infer_offense_team(recovery: Recovery):
    """Offense = the team that possesses the ball across the clip (majority vote).

    Possession is read in PIXEL space (recovery.ball_possessor) rather than court
    space, because the ball projects off the court plane and its recovered court
    distance is biased. Voting over every frame recovers the offense robustly.
    """
    from collections import Counter
    votes = Counter()
    for tid in recovery.ball_possessor:
        team = _track_team_id(recovery, tid) if tid is not None else None
        if team is not None:
            votes[team] += 1
    return votes.most_common(1)[0][0] if votes else None


def composite_confidence(n_players, homog_ok, id_confs, ball_recent) -> float:
    """Blend the perception-quality signals into one confidence (roadmap 5.1)."""
    return float(
        W_PLAYERS * min(n_players / 10.0, 1.0)
        + W_HOMOG * (1.0 if homog_ok else 0.0)
        + W_IDENTITY * (np.mean(id_confs) if id_confs else 0.0)
        + W_BALL * (1.0 if ball_recent else 0.0)
    )


def build_state_from_cv(recovery: Recovery, frame_idx: int, offense_team_id=None,
                        roster_jersey=None):
    """Build a Phase 1 State for one broadcast frame. Returns (state, confidence).

    offense_team_id: if known, forces which side is offense; otherwise inferred
    as the team of the player nearest the ball. roster_jersey: player_id -> jersey
    for display.
    """
    court_pos = recovery.frames_court[frame_idx]
    vel = recovery.velocities[frame_idx]
    ball = recovery.ball_court[frame_idx]
    ids = recovery.identities

    # Ball fallback: last known within a small window.
    ball_recent = ball is not None
    if ball is None:
        for j in range(frame_idx, max(-1, frame_idx - 4), -1):
            if recovery.ball_court[j] is not None:
                ball = recovery.ball_court[j]
                break
    if ball is None:
        # Ball never detected (COCO YOLO misses the small fast ball). It is far
        # likelier to be among the players than at center court, so fall back to
        # the detected-players centroid rather than (47, 25).
        if court_pos:
            cx = float(np.mean([p[0] for p in court_pos.values()]))
            cy = float(np.mean([p[1] for p in court_pos.values()]))
            ball = (cx, cy)
        else:
            ball = (47.0, 25.0)

    # Assemble players; team_id from identity cluster mapping.
    players_meta = []
    for tid, (x, y) in court_pos.items():
        ident = ids.get(tid)
        team_id = ident.team_id if ident and ident.team_id is not None else ident.team_label if ident else 0
        pid = ident.player_id if (ident and ident.player_id is not None) else _fallback_pid(tid)
        vx, vy = vel.get(tid, (0.0, 0.0))
        players_meta.append((tid, pid, team_id, x, y, vx, vy))

    if not players_meta:
        return None, 0.0

    # Offense inferred robustly across the whole clip (see infer_offense_team).
    if offense_team_id is None:
        offense_team_id = infer_offense_team(recovery)
        if offense_team_id is None:
            nearest = min(players_meta, key=lambda m: (m[3] - ball[0]) ** 2 + (m[4] - ball[1]) ** 2)
            offense_team_id = nearest[2]

    # Half-court flip to canonical (attack always at BASKET_LEFT), exactly as
    # Phase 1 segmentation does — the schema computes zones/polar off the left
    # rim, so a right-attacking possession must be mirrored first.
    off_xs = [m[3] for m in players_meta if m[2] == offense_team_id]
    if off_xs and float(np.mean(off_xs)) > court.HALFCOURT_X:
        flipped = []
        for tid, pid, team_id, x, y, vx, vy in players_meta:
            fx, fy = court.flip_to_left(x, y)
            fvx, fvy = court.flip_velocity(vx, vy)
            flipped.append((tid, pid, team_id, float(fx), float(fy), float(fvx), float(fvy)))
        players_meta = flipped
        bx, by = court.flip_to_left(ball[0], ball[1])
        ball = (float(bx), float(by))

    # Handler = the pixel-space ball possessor if it is an offensive track;
    # otherwise fall back to nearest offensive player to the (noisy) ball court xy.
    handler_pid = None
    possessor_tid = recovery.ball_possessor[frame_idx]
    meta_by_tid = {m[0]: m for m in players_meta}
    if possessor_tid in meta_by_tid and meta_by_tid[possessor_tid][2] == offense_team_id:
        handler_pid = meta_by_tid[possessor_tid][1]
    else:
        best_d = 8.0
        for _, pid, team_id, x, y, *_ in players_meta:
            if team_id != offense_team_id:
                continue
            d = np.hypot(x - ball[0], y - ball[1])
            if d < best_d:
                best_d, handler_pid = d, pid

    player_frames = [
        PlayerFrame(pid, team_id, "offense" if team_id == offense_team_id else "defense",
                    x, y, vx, vy)
        for (_, pid, team_id, x, y, vx, vy) in players_meta
    ]
    frame = Frame(
        possession_id=f"cv_frame{frame_idx}", frame_idx=frame_idx, quarter=1,
        game_clock=0.0, shot_clock=None,
        ball_x=ball[0], ball_y=ball[1], ball_z=3.5, ball_vx=0.0, ball_vy=0.0,
        handler_player_id=handler_pid, ball_in_flight=handler_pid is None,
        players=player_frames, offense_team_id=offense_team_id,
    )
    state = build_state(frame, roster_jersey=roster_jersey, last_touch_time=None)

    # Composite confidence overrides the Phase-1 placeholder 1.0.
    id_confs = [ids[t].confidence for t in court_pos if t in ids and ids[t].player_id is not None]
    conf = composite_confidence(len(players_meta), recovery.homog_ok[frame_idx], id_confs, ball_recent)
    state.context.confidence = round(conf, 3)
    return state, conf


def frame_player_pixels(recovery: Recovery, clip, frame_idx: int):
    """Return ({player_id: foot_pixel}, ball_pixel|None) for a frame.

    Uses tracked *pixel* positions directly (roadmap 5.3: the tracker supplies
    where things are), keyed by the resolved player_id so the overlay can draw
    arrows on bodies regardless of any court-space flip.
    """
    feet, handler_key = {}, {}
    for t in recovery.tracklets:
        for fidx, det in t.history:
            if fidx != frame_idx:
                continue
            ident = recovery.identities.get(t.track_id)
            pid = ident.player_id if (ident and ident.player_id is not None) else _fallback_pid(t.track_id)
            feet[pid] = det.foot
    balls = clip.frames[frame_idx].detections.by_class("ball")
    ball_px = balls[0].center if balls else None
    return feet, ball_px


def is_showable(state, confidence: float) -> bool:
    """Gate: enough players and enough confidence to show a recommendation (4.9)."""
    if state is None:
        return False
    return confidence >= CONFIDENCE_GATE and state.context.n_players_observed >= MIN_PLAYERS_SHOW


def pick_showable_frame(recovery: Recovery, offense_team_id=None) -> int | None:
    """Pick the best broadcast frame to analyze: showable, with a ball handler.

    Emulates a coach pausing on a clean look — prefers frames with a valid
    homography, the most players recovered, and a settled ball handler.
    """
    best_idx, best_score = None, -1.0
    for i in range(len(recovery.frames_court)):
        state, conf = build_state_from_cv(recovery, i, offense_team_id)
        if not is_showable(state, conf) or state.handler is None:
            continue
        score = conf + 0.05 * state.context.n_players_observed
        if score > best_score:
            best_score, best_idx = score, i
    return best_idx
