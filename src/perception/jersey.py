"""Jersey-number OCR for real broadcast video (roadmap 4.5).

Reads jersey numbers off player crops with EasyOCR and stamps a *tracklet-voted*
number onto each detection, so a player who is legible in only a few frames still
gets a stable identity across the possession. Back-facing / blurred players get no
number (correctly) — the vote just abstains.

Heavy dep (easyocr) is imported lazily; the pipeline runs without it. On real
video the read numbers become the player labels in the overlay and the "#N"
references in the rationale and the assistant.
"""
from __future__ import annotations

from collections import Counter

import numpy as np

from src.perception.tracking import run_tracker


class JerseyReader:
    """EasyOCR wrapper: read a single jersey number (0-99) from a player crop."""

    def __init__(self, gpu: bool = False, min_conf: float = 0.5):
        self.gpu = gpu
        self.min_conf = min_conf
        self._reader = None

    def _load(self):
        if self._reader is None:
            import easyocr  # lazy: heavy dep
            self._reader = easyocr.Reader(["en"], gpu=self.gpu, verbose=False)
        return self._reader

    def read(self, rgb: np.ndarray, bbox) -> tuple[int, float] | None:
        """Best (number, confidence) from the upper-torso band of a player box."""
        x1, y1, x2, y2 = [int(v) for v in bbox]
        h, w = y2 - y1, x2 - x1
        if h < 40 or w < 12:
            return None                      # too small to read
        cy1, cy2 = y1 + int(0.15 * h), y1 + int(0.55 * h)   # where the number sits
        crop = rgb[max(0, cy1):cy2, max(0, x1):x2]
        if crop.size == 0:
            return None
        big = np.repeat(np.repeat(crop, 3, axis=0), 3, axis=1)   # upscale for small text
        best = None
        for _, text, conf in self._load().readtext(big, allowlist="0123456789"):
            text = text.strip()
            if not text.isdigit() or not (0 <= int(text) <= 99) or conf < self.min_conf:
                continue
            if best is None or conf > best[1]:
                best = (int(text), float(conf))
        return best


def assign_jerseys(frames, reader: JerseyReader | None = None, samples: int = 5,
                   min_len: int = 3) -> dict:
    """Track over ``frames`` and stamp a voted jersey number onto each detection.

    For every tracklet, OCR its largest (nearest-camera) crops, confidence-weight
    the votes, and write the winner to ``det.jersey_read`` on *all* of that
    tracklet's detections — so any frame's detection carries the possession-stable
    number. Returns {track_id: jersey}.
    """
    if reader is None:
        reader = JerseyReader()
    tracker = run_tracker(frames)
    out: dict = {}
    for t in tracker.all_tracklets(min_len=min_len):
        # OCR the biggest crops (numbers are most legible up close).
        hist = sorted(t.history, key=lambda fd: -(fd[1].bbox[3] - fd[1].bbox[1]))[:samples]
        votes: Counter = Counter()
        for fidx, det in hist:
            if not (0 <= fidx < len(frames)) or frames[fidx].image is None:
                continue
            r = reader.read(frames[fidx].image, det.bbox)
            if r is not None:
                votes[r[0]] += r[1]          # weight by confidence
        if not votes:
            continue
        number, weight = votes.most_common(1)[0]
        # Require corroboration (multiple/confident reads) so one bad frame doesn't
        # stamp a wrong number like 64/84 across a whole tracklet.
        if weight < 1.0:
            continue
        out[t.track_id] = number
        for _, det in t.history:
            det.jersey_read = number
    return out
