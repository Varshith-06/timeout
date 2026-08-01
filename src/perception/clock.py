"""Shot-clock extraction with temporal validation (roadmap 4.8).

The scoreboard graphic gives a per-frame OCR reading of the shot clock. Any
single reading can be an OCR blunder, but the clock obeys hard physics: it
decreases at ~1.0/sec and only ever jumps *up* on a reset (to 24, or 14 after an
offensive rebound). Readings that violate monotonicity are discarded and filled
by interpolation, which makes the extracted clock near-perfect in practice.

The OCR itself (PaddleOCR/EasyOCR on a digit ROI) is upstream; this is the
validation layer, and it is what makes the signal trustworthy.
"""
from __future__ import annotations

import numpy as np

RESET_VALUES = (24.0, 14.0)
RESET_TOL = 1.5
RESET_FROM_MAX = 8.0   # the clock only resets after it has wound down low


def _is_plausible_reset(prev: float, cur: float) -> bool:
    """A reset jumps UP to 24/14, and only from a low prior value.

    Requiring ``prev <= RESET_FROM_MAX`` stops a mid-possession OCR blunder near
    24 from being mistaken for a reset — which would otherwise cascade and reject
    every real reading after it. In production the actual reset is confirmed by
    the possession boundary from Phase 1 segmentation, not the clock alone.
    """
    if prev > RESET_FROM_MAX:
        return False
    return cur > prev and any(abs(cur - r) <= RESET_TOL for r in RESET_VALUES)


def validate_clock(readings, dt: float = 0.2, max_drop_per_frame: float = 2.0):
    """Clean a sequence of raw shot-clock readings.

    readings: list of floats or None. dt: seconds between frames. Returns a list
    of validated floats (None only where it cannot be recovered). A reading is
    accepted if it continues the monotonic countdown within tolerance, or is a
    plausible reset; otherwise it is dropped and later linearly interpolated.
    """
    n = len(readings)
    accepted: list[float | None] = [None] * n
    last_val = None
    for i, r in enumerate(readings):
        if r is None:
            continue
        if last_val is None:
            accepted[i] = r
            last_val = r
            continue
        expected = last_val - dt
        if _is_plausible_reset(last_val, r):
            accepted[i] = r
            last_val = r
        elif r <= last_val + 0.5 and r >= expected - max_drop_per_frame:
            # Monotone-ish countdown within tolerance.
            accepted[i] = r
            last_val = r
        # else: OCR violation -> reject (leave None for interpolation).
    return _interpolate(accepted)


def _interpolate(vals):
    """Linear fill of None gaps between known values; edge-holds the ends."""
    n = len(vals)
    idx = [i for i, v in enumerate(vals) if v is not None]
    if not idx:
        return vals
    out = list(vals)
    # Fill interior gaps.
    for a, b in zip(idx, idx[1:]):
        if b > a + 1:
            for k in range(a + 1, b):
                frac = (k - a) / (b - a)
                out[k] = out[a] + frac * (out[b] - out[a])
    # Edge-hold.
    for k in range(0, idx[0]):
        out[k] = out[idx[0]]
    for k in range(idx[-1] + 1, n):
        out[k] = out[idx[-1]]
    return out


def clock_accuracy(validated, truth, tol: float = 0.75) -> float:
    """Fraction of frames within tol seconds of the true clock (eval only)."""
    pairs = [(v, t) for v, t in zip(validated, truth) if v is not None and t is not None]
    if not pairs:
        return float("nan")
    return float(np.mean([abs(v - t) <= tol for v, t in pairs]))
