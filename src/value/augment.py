"""Noise augmentation for training V(s) — the step you cannot skip (roadmap 3.3).

V(s) trains on near-perfect SportVU tracking but in Phase 4 is served output from
a broadcast CV pipeline with several feet of position error, dropped players, and
identity swaps. A model trained clean and served noisy degrades in ways that do
not show up in clean validation. So we inject the perturbations at train time,
matched (later) to measured Phase 3 error:

  * Gaussian position jitter, sigma ~1.5-2.5 ft.
  * Random dropout of 1-3 players (broadcast frames routinely miss them).
  * Same-team identity swaps at ~2% (re-ID error).
  * Lag jitter on velocity estimates.

Operates on the entity-tensor representation from :mod:`src.value.features`, so
the permutation-invariant net exercises variable entity counts for free.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.state import court

X_NORM = court.HALFCOURT_X
Y_NORM = court.COURT_WIDTH / 2


@dataclass
class AugmentConfig:
    jitter_ft: float = 2.0        # position sigma in feet
    dropout_max: int = 3          # up to this many players dropped
    dropout_prob: float = 0.5     # chance a frame drops anyone at all
    id_swap_prob: float = 0.15    # chance of one same-team identity swap
    vel_lag_ft: float = 1.0       # velocity noise (ft/s-ish, in normalized units)


def augment_sample(ents, pidx, mask, rng: np.random.Generator, cfg: AugmentConfig):
    """Return perturbed copies of (ents, pidx, mask). globals are left untouched."""
    ents = ents.copy()
    pidx = pidx.copy()
    mask = mask.copy()

    active = np.where(mask > 0)[0]
    is_ball = ents[:, 6] > 0.5

    # 1. Position jitter (normalized units).
    jx = rng.normal(0, cfg.jitter_ft / X_NORM, size=len(ents))
    jy = rng.normal(0, cfg.jitter_ft / Y_NORM, size=len(ents))
    ents[:, 0] += jx * mask
    ents[:, 1] += jy * mask

    # 2. Velocity lag jitter.
    ents[:, 2] += rng.normal(0, cfg.vel_lag_ft / 10.0, size=len(ents)) * mask
    ents[:, 3] += rng.normal(0, cfg.vel_lag_ft / 10.0, size=len(ents)) * mask

    # 3. Player dropout (never the ball, never the handler).
    droppable = [i for i in active if not is_ball[i] and ents[i, 7] < 0.5]
    if droppable and rng.random() < cfg.dropout_prob:
        k = int(rng.integers(1, cfg.dropout_max + 1))
        k = min(k, len(droppable))
        for i in rng.choice(droppable, size=k, replace=False):
            mask[i] = 0.0

    # 4. Same-team identity swap (re-ID error): swap two players' embeddings.
    if rng.random() < cfg.id_swap_prob:
        for team_flag in (4, 5):  # is_offense col, is_defense col
            members = [i for i in active if mask[i] > 0 and ents[i, team_flag] > 0.5]
            if len(members) >= 2:
                a, b = rng.choice(members, size=2, replace=False)
                pidx[a], pidx[b] = pidx[b], pidx[a]
                break

    return ents, pidx, mask
