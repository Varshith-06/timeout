"""Export a scored frame as a JSON overlay spec for the web app.

The web UI can't run the Python pipeline, so this turns one analyzed pause into a
self-contained, JSON-serialisable description in **video-pixel coordinates**: the
players, the ball, and every candidate action's drawable geometry (a circle on the
handler, an arrow to the pass target / rim / screener) plus a per-action rationale.
The browser draws it on a canvas scaled to the displayed video size.

Keeps the project's rule intact: the model chose the action, the tracker supplies
the pixels — nothing here invents a coordinate the perception layer didn't produce.
"""
from __future__ import annotations

import numpy as np

from src.llm.context import CoachingPriors
from src.llm.rationale import generate_rationale
from src.perception.state_from_cv import frame_player_pixels
from src.state import court


def _project_court_to_px(H_pixel_to_court, pts):
    import cv2
    Hinv = np.linalg.inv(H_pixel_to_court)
    pts = np.asarray(pts, dtype=np.float64).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(pts, Hinv).reshape(-1, 2)


def _roster_boxes(recovery, frame_idx: int) -> tuple[dict, dict]:
    """(pid -> pixel bbox, pid -> jersey number) for roster players at this frame."""
    from src.perception.state_from_cv import _fallback_pid
    boxes, jerseys = {}, {}
    for t in recovery.tracklets:
        if t.track_id not in recovery.roster_tids:
            continue
        ident = recovery.identities.get(t.track_id)
        pid = ident.player_id if (ident and ident.player_id is not None) else _fallback_pid(t.track_id)
        # Tracklet-voted jersey (stamped on every detection by assign_jerseys).
        votes = [d.jersey_read for _, d in t.history if d.jersey_read is not None]
        if votes:
            jerseys[pid] = max(set(votes), key=votes.count)
        for fidx, det in t.history:
            if fidx == frame_idx:
                boxes[pid] = [float(v) for v in det.bbox]
    return boxes, jerseys


def _rim_px(state, H_pixel_to_court, handler_px):
    """Pixel of the rim the handler is attacking (the nearer of the two)."""
    rims = _project_court_to_px(H_pixel_to_court, [court.BASKET_LEFT, court.BASKET_RIGHT])
    if handler_px is None:
        return [float(rims[0][0]), float(rims[0][1])]
    i = int(np.argmin(np.linalg.norm(rims - np.asarray(handler_px), axis=1)))
    return [float(rims[i][0]), float(rims[i][1])]


def _action_geometry(sc, handler_px, feet, rim_px):
    """Drawable geometry (pixels) for one scored action."""
    a = sc.action
    geo = {"circle": handler_px}
    if a.action in ("PASS_TO", "SCREEN_WITH") and a.target in feet:
        geo["arrow"] = [handler_px, feet[a.target]]
        geo["target"] = feet[a.target]
    elif a.action in ("DRIVE", "SHOOT"):
        geo["arrow"] = [handler_px, rim_px]
    else:  # RESET
        geo["arrow"] = None
    return geo


def _label(sc) -> str:
    a = sc.action
    if a.action == "PASS_TO":
        return f"PASS (EPV {sc.q:.2f})"
    if a.action == "SCREEN_WITH":
        return f"SCREEN (EPV {sc.q:.2f})"
    if a.action == "DRIVE":
        return f"DRIVE {a.direction} (EPV {sc.q:.2f})"
    return f"{a.action} (EPV {sc.q:.2f})"


def build_overlay_spec(recovery, clip, frame_idx: int, state, scored_actions,
                       names: dict | None = None, coaching: CoachingPriors | None = None,
                       rationale_top_k: int = 6, video_time: float | None = None,
                       roster=None) -> dict:
    """Build the JSON overlay spec for one analyzed frame.

    ``rationale_top_k`` full rationales are generated (the actions a user is likely
    to ask about); the rest carry a short auto-description so the payload stays small.
    """
    feet, ball_px = frame_player_pixels(recovery, clip, frame_idx)
    handler = state.handler
    handler_px = feet.get(handler.player_id) if handler is not None else None
    rim_px = _rim_px(state, recovery.homographies[frame_idx], handler_px)
    boxes, jerseys = _roster_boxes(recovery, frame_idx)

    # Jersey numbers become player names the rationale echoes — a roster upgrades
    # "#7" to the real name ("Curry") when the number is on it.
    names = dict(names or {})
    for pid, j in jerseys.items():
        names.setdefault(pid, roster.label(j) if roster is not None else f"#{j}")

    players = []
    for p in state.players:
        j = jerseys.get(p.player_id)
        players.append({
            "id": int(p.player_id),
            "team": "offense" if p.team_id == state.offense_team_id else "defense",
            "is_handler": handler is not None and p.player_id == handler.player_id,
            "jersey": j,
            "name": (roster.label(j) if (roster is not None and j is not None) else None),
            "foot": feet.get(p.player_id),
            "box": boxes.get(p.player_id),
        })

    actions = []
    for rank, sc in enumerate(scored_actions):
        entry = {
            "id": sc.action.id,
            "action": sc.action.action,
            "label": _label(sc),
            "epv": round(sc.q, 3),
            "success_pct": int(round(sc.success_prob * 100)),
            "target_id": sc.action.target,
            "direction": sc.action.direction,
            "kind": sc.kind,
            "geometry": _action_geometry(sc, handler_px, feet, rim_px),
        }
        if rank < rationale_top_k:
            # Reorder so THIS action is the "recommendation" the rationale explains.
            reordered = [sc] + [o for o in scored_actions if o is not sc]
            r = generate_rationale(state, reordered, names, coaching=coaching).rationale
            entry["rationale"] = r.to_dict()
        actions.append(entry)

    return {
        "video_time": video_time,
        "frame_idx": int(frame_idx),
        "confidence": round(float(state.context.confidence), 3),
        "n_players": int(state.context.n_players_observed),
        "handler_id": int(handler.player_id) if handler is not None else None,
        "ball_px": list(ball_px) if ball_px is not None else None,
        "rim_px": rim_px,
        "players": players,
        "actions": actions,
    }
