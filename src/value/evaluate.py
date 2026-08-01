"""Phase 2 evaluation (roadmap 3.4) + an automated proxy for the week-8 gate (3.5).

  * Calibration — reliability diagram, predicted vs realized, the metric that
    matters more than any accuracy number (3.4).
  * Brier score on each sub-model against a base-rate baseline.
  * EPV trajectory — V(s) through a possession should rise as the offense gets a
    better look and fall under late-clock pressure.
  * Ordering / recommendation quality — because the simulator knows the true
    expected points of every action, we can automate the coach gate: how often is
    the model's top pick right-or-defensible, and how often is it flat wrong.
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.value import simulation as SIM
from src.value.actions import enumerate_actions
from src.value.scoring import score_actions


# --- Calibration -------------------------------------------------------------
def brier_score(p, y) -> float:
    p, y = np.asarray(p, float), np.asarray(y, float)
    return float(np.mean((p - y) ** 2))


def reliability_curve(p, y, n_bins: int = 10):
    """Return (bin_mean_pred, bin_mean_true, bin_count) over equal-width bins."""
    p, y = np.asarray(p, float), np.asarray(y, float)
    edges = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, n_bins - 1)
    mp, mt, cnt = [], [], []
    for b in range(n_bins):
        m = idx == b
        if m.sum() == 0:
            continue
        mp.append(p[m].mean()); mt.append(y[m].mean()); cnt.append(int(m.sum()))
    return np.array(mp), np.array(mt), np.array(cnt)


def plot_reliability(named, path, n_bins: int = 10):
    """named: {label: (p, y)}. Saves a reliability diagram."""
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="perfect")
    for label, (p, y) in named.items():
        mp, mt, _ = reliability_curve(p, y, n_bins)
        ax.plot(mp, mt, "o-", label=f"{label} (Brier={brier_score(p, y):.3f})")
    ax.set_xlabel("predicted probability")
    ax.set_ylabel("observed frequency")
    ax.set_title("Reliability diagram — sub-models")
    ax.legend(fontsize=8)
    ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


# --- EPV trajectory ----------------------------------------------------------
def epv_curve(states, value_model) -> np.ndarray:
    return value_model.value_batch(states)


def plot_epv_trajectories(possessions, value_model, path, n: int = 8):
    fig, ax = plt.subplots(figsize=(7, 5))
    for poss in possessions[:n]:
        v = epv_curve(poss.states, value_model)
        color = "tab:green" if poss.realized_points > 0 else "tab:red"
        ax.plot(range(len(v)), v, "-o", color=color, alpha=0.6)
    ax.set_xlabel("possession step")
    ax.set_ylabel("V(s) — expected points")
    ax.set_title("EPV trajectories (green=scored, red=empty)")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


# --- Recommendation quality (automated week-8 gate proxy) --------------------
def _baseline_pick(kind, state, actions, rng):
    """A naive policy's chosen action (roadmap 6.2 baselines)."""
    if kind == "always_shoot":
        return next((a for a in actions if a.action == "SHOOT"), actions[0])
    if kind == "pass_most_open":
        passes = [a for a in actions if a.action == "PASS_TO"]
        if not passes:
            return actions[0]
        openness = {p.player_id: 1 - p.defender_pressure for p in state.offense()}
        return max(passes, key=lambda a: openness.get(a.target, 0))
    return actions[rng.integers(len(actions))]  # random


def ordering_eval(possessions, submodels, value_model, skills,
                  defensible_margin: float = 0.15, wrong_margin: float = 0.30, seed: int = 0):
    """Compare the model's top pick against the ground-truth-best action, and
    against naive baselines (roadmap 6.2).

    Regret = true_value(best) - true_value(pick). Thresholds are coach-scale EPV
    margins: 'defensible' = within ~0.15 pts of optimal (the sub-models' own
    estimation-noise floor on a three), 'wrong' = >=0.30 pts worse (a clearly
    inferior choice). The headline result is that the model's mean regret is far
    below every naive baseline's — the blind-baseline comparison the roadmap asks
    for (3.4/6.2) — which a fixed cutoff alone does not capture.
    """
    rng = np.random.default_rng(seed)
    policies = ["model", "always_shoot", "pass_most_open", "random"]
    agg = {k: {"right": 0, "defensible": 0, "wrong": 0, "regrets": []} for k in policies}
    n = 0
    for poss in possessions:
        for state in poss.states:
            if state.handler is None:
                continue
            actions = enumerate_actions(state)
            if len(actions) < 2:
                continue
            true_vals = {a.id: SIM._true_action_value(state, a, skills) for a in actions}
            best_true = max(true_vals.values())
            scored = score_actions(state, actions, submodels, value_model)
            picks = {
                "model": scored[0].action,
                "always_shoot": _baseline_pick("always_shoot", state, actions, rng),
                "pass_most_open": _baseline_pick("pass_most_open", state, actions, rng),
                "random": _baseline_pick("random", state, actions, rng),
            }
            for k, a in picks.items():
                regret = best_true - true_vals[a.id]
                agg[k]["regrets"].append(regret)
                if regret <= 1e-6:
                    agg[k]["right"] += 1
                if regret <= defensible_margin:
                    agg[k]["defensible"] += 1
                if regret >= wrong_margin:
                    agg[k]["wrong"] += 1
            n += 1
    if n == 0:
        return {}
    out = {"n": n}
    for k in policies:
        r = agg[k]
        out[k] = {
            "right_rate": r["right"] / n,
            "right_or_defensible": r["defensible"] / n,
            "wrong_rate": r["wrong"] / n,
            "mean_regret": float(np.mean(r["regrets"])),
        }
    return out
