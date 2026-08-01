"""Camera-cut detection (roadmap 4.3).

Broadcasts cut to replays, close-ups, and crowd shots constantly. Without cut
detection, track IDs persist across a replay and the recovered state becomes
nonsense. The production method is a frame-to-frame color-histogram distance
threshold; that is implemented here and is testable on image arrays.

The synthetic broadcast does not render pixels, so its per-frame ``cut`` flag is
used to drive the tracker reset in the pipeline — but the histogram detector
below is the real component a video path would call each frame.
"""
from __future__ import annotations

import numpy as np


def frame_histogram(image: np.ndarray, bins: int = 8) -> np.ndarray:
    """Normalized 3D RGB color histogram of an (H,W,3) uint8 image."""
    img = np.asarray(image)
    if img.ndim != 3 or img.shape[2] < 3:
        raise ValueError("expected an (H, W, 3) image")
    idx = (img[..., :3].astype(int) * bins // 256).reshape(-1, 3)
    flat = idx[:, 0] * bins * bins + idx[:, 1] * bins + idx[:, 2]
    hist = np.bincount(flat, minlength=bins ** 3).astype(float)
    total = hist.sum()
    return hist / total if total > 0 else hist


def histogram_distance(h1: np.ndarray, h2: np.ndarray) -> float:
    """Bhattacharyya distance in [0, 1]; ~0 same shot, near 1 across a cut."""
    bc = float(np.sum(np.sqrt(h1 * h2)))
    return float(np.sqrt(max(0.0, 1.0 - bc)))


class CutDetector:
    """Flags a cut when consecutive-frame histogram distance exceeds a threshold."""

    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold
        self._prev = None

    def update(self, image: np.ndarray) -> bool:
        h = frame_histogram(image)
        cut = self._prev is not None and histogram_distance(self._prev, h) > self.threshold
        self._prev = h
        return bool(cut)
