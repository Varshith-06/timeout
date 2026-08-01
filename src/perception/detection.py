"""Detection schema + detector interface (roadmap 4.2).

Detection classes: player, ball, referee, rim. In production these come from a
YOLO detection model fine-tuned from COCO weights (Ultralytics), with a separate
higher-resolution model for the ball (small, fast, occluded ~half the time).
Here a real model would implement :class:`Detector`; the synthetic broadcast
(:mod:`src.perception.synthetic_broadcast`) emulates its output — including the
ball's 60-80% per-frame recall — so the whole downstream pipeline is testable
without the annotation budget.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

CLASSES = ("player", "ball", "referee", "rim")


@dataclass
class Detection:
    cls: str                      # one of CLASSES
    bbox: tuple                   # (x1, y1, x2, y2) pixels
    conf: float
    # Appearance embedding of the torso crop (SigLIP in production; synthetic
    # here) — drives team clustering (roadmap 4.4).
    embedding: np.ndarray | None = None
    # Jersey-number OCR read for this crop, or None if illegible (roadmap 4.5).
    jersey_read: int | None = None
    # Ground-truth tags, populated only by the synthetic detector for evaluation.
    # The pipeline must never read these; they exist to measure the domain gap.
    true_player_id: int | None = None
    true_team: str | None = None

    @property
    def foot(self) -> tuple:
        """Bottom-center of the box — the point that lies on the court plane."""
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2, y2)

    @property
    def center(self) -> tuple:
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)


@dataclass
class FrameDetections:
    frame_idx: int
    detections: list[Detection] = field(default_factory=list)

    def by_class(self, cls: str) -> list[Detection]:
        return [d for d in self.detections if d.cls == cls]

    @property
    def players(self) -> list[Detection]:
        return self.by_class("player")


class Detector:
    """Interface a real YOLO model implements. Not used by the synthetic path."""

    def detect(self, image: np.ndarray, frame_idx: int) -> FrameDetections:  # pragma: no cover
        raise NotImplementedError(
            "Plug a fine-tuned Ultralytics YOLO here (player/ball/referee/rim). "
            "See roadmap 4.2 — separate high-res ball model recommended."
        )


def boxes_array(dets: list[Detection]) -> np.ndarray:
    """(N,4) array of bboxes for a list of detections."""
    if not dets:
        return np.zeros((0, 4))
    return np.array([d.bbox for d in dets], dtype=float)
