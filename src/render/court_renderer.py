"""Top-down court renderer (roadmap 2.7).

Matplotlib, 94x50, drawn from the constants in :mod:`src.state.court`. Players as
circles colored by side, the ball as a small marker, velocity as a short arrow,
and candidate actions as dashed arrows labelled with their scores.

The renderer is the *only* component that turns an action object into geometry
(roadmap 2.6, 5.3): every drawn endpoint comes from a tracked position, never
from a model-emitted coordinate.

Uses the non-interactive Agg backend so it runs headless.
"""
from __future__ import annotations

import math

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Arc, Circle, Rectangle

from src.state import court
from src.state.schema import State
from src.value.actions import Action

OFFENSE_COLOR = "#1f77b4"
DEFENSE_COLOR = "#d62728"
BALL_COLOR = "#ff7f0e"
COURT_LINE = "#555555"
TOP_ACTION_COLOR = "#2ca02c"
ALT_ACTION_COLOR = "#9467bd"


def draw_court(ax, half: bool = True):
    """Draw NBA court lines for the attacking (left) half from court constants."""
    rim = court.BASKET_LEFT
    lw = 1.4
    # Boundary.
    xmax = court.HALFCOURT_X if half else court.COURT_LENGTH
    ax.add_patch(Rectangle((0, 0), xmax, court.COURT_WIDTH, fill=False, ec=COURT_LINE, lw=lw))

    # Paint / lane.
    ax.add_patch(Rectangle((0, court.PAINT_Y[0]), court.FT_LINE_DIST, court.PAINT_WIDTH,
                           fill=False, ec=COURT_LINE, lw=lw))
    # Free-throw circle.
    ax.add_patch(Arc((court.FT_LINE_DIST, 25), 2 * court.FT_CIRCLE_R, 2 * court.FT_CIRCLE_R,
                     theta1=-90, theta2=90, ec=COURT_LINE, lw=lw))
    ax.add_patch(Arc((court.FT_LINE_DIST, 25), 2 * court.FT_CIRCLE_R, 2 * court.FT_CIRCLE_R,
                     theta1=90, theta2=270, ec=COURT_LINE, lw=lw, ls=":"))

    # Backboard + rim.
    ax.plot([court.BACKBOARD_INSET, court.BACKBOARD_INSET], [22, 28], color=COURT_LINE, lw=lw)
    ax.add_patch(Circle(rim, 0.75, fill=False, ec=BALL_COLOR, lw=lw))
    # Restricted area.
    ax.add_patch(Arc(rim, 2 * court.RESTRICTED_R, 2 * court.RESTRICTED_R,
                     theta1=-90, theta2=90, ec=COURT_LINE, lw=lw))

    # Three-point line: corner straights + arc.
    break_x = rim[0] + math.sqrt(max(0.0, court.THREE_ARC_R ** 2 - (25 - court.THREE_CORNER_Y[0]) ** 2))
    ax.plot([0, break_x], [court.THREE_CORNER_Y[0], court.THREE_CORNER_Y[0]], color=COURT_LINE, lw=lw)
    ax.plot([0, break_x], [court.THREE_CORNER_Y[1], court.THREE_CORNER_Y[1]], color=COURT_LINE, lw=lw)
    break_angle = math.degrees(math.atan2(25 - court.THREE_CORNER_Y[0], break_x - rim[0]))
    ax.add_patch(Arc(rim, 2 * court.THREE_ARC_R, 2 * court.THREE_ARC_R,
                     theta1=-break_angle, theta2=break_angle, ec=COURT_LINE, lw=lw))

    # Half-court line + center circle.
    ax.plot([court.HALFCOURT_X, court.HALFCOURT_X], [0, court.COURT_WIDTH], color=COURT_LINE, lw=lw)
    ax.add_patch(Arc(court.CENTER_CIRCLE, 2 * court.CENTER_CIRCLE_R, 2 * court.CENTER_CIRCLE_R,
                     theta1=90, theta2=270, ec=COURT_LINE, lw=lw))

    ax.set_xlim(-2, xmax + 2)
    ax.set_ylim(-2, court.COURT_WIDTH + 2)
    ax.set_aspect("equal")
    ax.axis("off")


def _draw_players(ax, state: State):
    # Draw defense first, then offense, then the ball handler last, so a tightly
    # guarded handler is never hidden under his defender.
    order = sorted(state.players, key=lambda p: (p.has_ball, p.side == "offense"))
    for p in order:
        color = OFFENSE_COLOR if p.side == "offense" else DEFENSE_COLOR
        edge = "black" if p.has_ball else color
        z = 8 if p.has_ball else 3
        ax.add_patch(Circle((p.x, p.y), 1.4, fc=color, ec=edge,
                            lw=2.4 if p.has_ball else 1.0, zorder=z))
        label = str(p.jersey) if p.jersey is not None else str(p.player_id)[-2:]
        ax.text(p.x, p.y, label, color="white", ha="center", va="center",
                fontsize=7, fontweight="bold", zorder=z + 1)
        # Velocity arrow (scaled).
        if p.speed > 0.5:
            ax.arrow(p.x, p.y, p.vx * 0.35, p.vy * 0.35, head_width=0.5,
                     fc=color, ec=color, alpha=0.6, zorder=2, length_includes_head=True)


def _draw_ball(ax, state: State):
    b = state.ball
    ax.add_patch(Circle((b["x"], b["y"]), 0.8, fc=BALL_COLOR, ec="black", lw=0.8, zorder=5))


def _player_xy(state: State, player_id: int):
    for p in state.players:
        if p.player_id == player_id:
            return p.x, p.y
    return None


def _draw_action(ax, state: State, action: Action, color, score=None, ls="--"):
    """Resolve one action object to geometry and draw it (5.3 table)."""
    handler = state.handler
    if handler is None:
        return
    hx, hy = handler.x, handler.y
    rim = court.BASKET_LEFT
    label_xy = None

    if action.action == "PASS_TO":
        tgt = _player_xy(state, action.target)
        if tgt:
            ax.annotate("", xy=tgt, xytext=(hx, hy),
                        arrowprops=dict(arrowstyle="-|>", color=color, ls=ls, lw=2.2), zorder=6)
            label_xy = ((hx + tgt[0]) / 2, (hy + tgt[1]) / 2)
    elif action.action == "DRIVE":
        off = {"left": 6.0, "right": -6.0, "middle": 0.0}[action.direction]
        mid = ((hx + rim[0]) / 2, (hy + rim[1]) / 2 + off)
        ax.annotate("", xy=rim, xytext=(hx, hy),
                    arrowprops=dict(arrowstyle="-|>", color=color, ls=ls, lw=2.2,
                                    connectionstyle=f"arc3,rad={0.25 if off > 0 else (-0.25 if off < 0 else 0)}"),
                    zorder=6)
        label_xy = mid
    elif action.action == "SHOOT":
        ax.add_patch(Circle((hx, hy), 2.2, fill=False, ec=color, lw=2.2, zorder=6))
        ax.annotate("", xy=rim, xytext=(hx, hy),
                    arrowprops=dict(arrowstyle="-|>", color=color, ls=ls, lw=2.0,
                                    connectionstyle="arc3,rad=-0.2"), zorder=6)
        label_xy = ((hx + rim[0]) / 2, (hy + rim[1]) / 2 - 3)
    elif action.action == "SCREEN_WITH":
        tgt = _player_xy(state, action.target)
        if tgt:
            ax.plot([hx, tgt[0]], [hy, tgt[1]], color=color, ls=ls, lw=2.2, zorder=6)
            ax.add_patch(Circle(tgt, 2.0, fill=False, ec=color, lw=1.8, zorder=6))
            label_xy = ((hx + tgt[0]) / 2, (hy + tgt[1]) / 2)
    elif action.action == "RESET":
        ax.annotate("", xy=(hx + 5, hy), xytext=(hx, hy),
                    arrowprops=dict(arrowstyle="-|>", color=color, ls=ls, lw=2.0,
                                    connectionstyle="arc3,rad=0.5"), zorder=6)
        label_xy = (hx + 3, hy + 3)

    if label_xy and score is not None:
        ax.text(label_xy[0], label_xy[1], f"{score:.2f}", color=color, fontsize=8,
                fontweight="bold", ha="center", va="center",
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec=color, alpha=0.85), zorder=7)


def render_state(
    state: State,
    actions: list[Action] | None = None,
    scores: dict[str, float] | None = None,
    top_k: int = 2,
    title: str | None = None,
    ax=None,
):
    """Render a state, optionally overlaying the top-k scored candidate actions.

    scores: {action_id: score}. The best is drawn solid green, the runner-up
    dashed purple (5.3), the rest omitted to keep the picture legible.
    """
    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=(7, 7))
    draw_court(ax)
    _draw_players(ax, state)
    _draw_ball(ax, state)

    if actions and scores:
        ranked = sorted(actions, key=lambda a: scores.get(a.id, float("-inf")), reverse=True)
        for rank, action in enumerate(ranked[:top_k]):
            color = TOP_ACTION_COLOR if rank == 0 else ALT_ACTION_COLOR
            ls = "-" if rank == 0 else "--"
            _draw_action(ax, state, action, color, scores.get(action.id), ls=ls)

    ts = state.timestamp
    sub = f"Q{ts['quarter']}  clock {ts['game_clock']:.1f}  shot {ts['shot_clock']}"
    ax.set_title(title or f"{state.possession_id}\n{sub}", fontsize=9)
    if own_fig:
        fig.tight_layout()
        return fig, ax
    return ax


def save_state_png(state, path, **kwargs):
    fig, _ = render_state(state, **kwargs)
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return path
