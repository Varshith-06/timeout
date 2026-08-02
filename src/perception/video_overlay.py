"""Video overlay — draw the recommendation on the broadcast frame (roadmap 5.3).

Project court coordinates *back* through the homography into pixel space and draw
there. Every drawn endpoint comes from a tracked position; the model chose which
players, the tracker supplies where they are (roadmap 5.3) — the rule that keeps
arrows on bodies.

The synthetic broadcast has no rendered pixels, so we reconstruct a broadcast-like
frame from the projected court lines and the detections. On real footage the same
code draws on the actual paused video frame.
"""
from __future__ import annotations

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

from src.state import court

OFFENSE_COLOR = "#1f77b4"
DEFENSE_COLOR = "#d62728"
BALL_COLOR = "#ff7f0e"
LINE_COLOR = "#888888"
TOP_COLOR = "#2ca02c"


def _arc_points(center, radius, a0, a1, n=40):
    a = np.linspace(np.radians(a0), np.radians(a1), n)
    return np.column_stack([center[0] + radius * np.cos(a), center[1] + radius * np.sin(a)])


def court_polylines() -> list:
    """Court lines as polylines in court feet (both halves)."""
    L, W = court.COURT_LENGTH, court.COURT_WIDTH
    lines = [
        np.array([[0, 0], [L, 0], [L, W], [0, W], [0, 0]]),          # boundary
        np.array([[47, 0], [47, W]]),                                  # half court
        np.array([[0, 17], [19, 17], [19, 33], [0, 33]]),             # left paint
        np.array([[L, 17], [75, 17], [75, 33], [L, 33]]),             # right paint
    ]
    for rim in (court.BASKET_LEFT, court.BASKET_RIGHT):
        b = 90 - np.degrees(np.arctan2(25 - court.THREE_CORNER_Y[0],
                                       court.THREE_ARC_R * 0 + 8.95))
        theta = np.degrees(np.arctan2(25 - 3, 8.95))
        sign = 1 if rim[0] < 47 else -1
        lines.append(_arc_points(rim, court.THREE_ARC_R,
                                 (-theta if sign > 0 else 180 + theta),
                                 (theta if sign > 0 else 180 - theta)))
    lines.append(_arc_points(court.CENTER_CIRCLE, court.CENTER_CIRCLE_R, 0, 360))
    return lines


def _project(H_court_to_px, pts):
    pts = np.asarray(pts, dtype=np.float64).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(pts, H_court_to_px).reshape(-1, 2)


def render_video_overlay(broadcast_frame, state, scored_actions, H_pixel_to_court,
                         player_feet, ball_px, path, title=None):
    """Draw the top recommendation on a reconstructed broadcast frame.

    Court lines are projected court->pixel via inv(H). The recommendation is drawn
    between *tracked pixel positions* (player_feet: player_id -> foot pixel), so
    arrows land on bodies regardless of any court-space flip (roadmap 5.3).
    """
    bf = broadcast_frame
    cam = bf.camera
    H_c2p = np.linalg.inv(H_pixel_to_court)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xlim(0, cam.img_w)
    ax.set_ylim(cam.img_h, 0)          # image coords: y increases downward
    ax.set_facecolor("#20301f")
    ax.set_xticks([]); ax.set_yticks([])

    # On real footage, draw the actual paused frame underneath; the synthetic
    # broadcast has no pixels, so fall back to the dark reconstructed canvas.
    on_real = getattr(bf, "image", None) is not None
    if on_real:
        ax.imshow(bf.image, extent=(0, cam.img_w, cam.img_h, 0), zorder=0)

    line_alpha = 0.9 if on_real else 0.7
    line_color = "#ffe14d" if on_real else LINE_COLOR   # yellow reads on footage
    for poly in court_polylines():
        px = _project(H_c2p, poly)
        ax.plot(px[:, 0], px[:, 1], color=line_color, lw=1.4, alpha=line_alpha, zorder=1)

    for d in bf.detections.by_class("player"):
        # On real footage, only box people who project onto the court (skip crowd/bench).
        if on_real:
            fx, fy = _project(H_pixel_to_court, [d.foot])[0]
            if not court.on_court(float(fx), float(fy)):
                continue
        x1, y1, x2, y2 = d.bbox
        ax.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False,
                               ec="#cccccc", lw=1.0, alpha=0.7))
    if ball_px is not None:
        ax.add_patch(plt.Circle(ball_px, 7, color=BALL_COLOR, zorder=5))

    if scored_actions and state.handler is not None:
        # Both rims in pixels; pick the one the handler is attacking (nearer).
        rims_px = _project(H_c2p, np.array([court.BASKET_LEFT, court.BASKET_RIGHT]))
        _draw_top_action(ax, state, scored_actions[0], player_feet, rims_px)

    ts = state.context
    ax.set_title(title or f"CV overlay — confidence {ts.confidence:.2f}, "
                          f"{ts.n_players_observed} players", fontsize=10, color="white")
    fig.patch.set_facecolor("#101510")
    fig.tight_layout()
    fig.savefig(path, dpi=130, facecolor=fig.get_facecolor())
    plt.close(fig)
    return path


def _draw_top_action(ax, state, scored, player_feet, rims_px):
    h = state.handler
    hx = player_feet.get(h.player_id)
    if hx is None:
        return
    hx = np.asarray(hx, dtype=float)
    action = scored.action
    ax.add_patch(plt.Circle(hx, 16, fill=False, ec=TOP_COLOR, lw=2.5, zorder=6))
    label = f"{action.action} (EPV {scored.q:.2f})"

    if action.action == "PASS_TO":
        tp = player_feet.get(action.target)
        if tp is not None:
            ax.annotate("", xy=np.asarray(tp, float), xytext=hx,
                        arrowprops=dict(arrowstyle="-|>", color=TOP_COLOR, lw=3), zorder=6)
    elif action.action in ("DRIVE", "SHOOT"):
        rim = rims_px[int(np.argmin(np.linalg.norm(rims_px - hx, axis=1)))]
        ax.annotate("", xy=rim, xytext=hx,
                    arrowprops=dict(arrowstyle="-|>", color=TOP_COLOR, lw=3,
                                    connectionstyle="arc3,rad=-0.2"), zorder=6)
    ax.text(hx[0], hx[1] - 24, label, color=TOP_COLOR, fontsize=10, fontweight="bold",
            ha="center", bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=TOP_COLOR), zorder=7)
