"""Broadcast-video ingestion + a pretrained detector adapter (roadmap 4.1/4.2).

This is the bridge from real `.mp4` pixels to the perception spine. It reads
frames with OpenCV and runs a *pretrained* (COCO) YOLO detector zero-shot —
person -> player, sports ball -> ball — so a real clip flows through the
existing tracking / homography / state pipeline without any training. A
fine-tuned basketball detector (roadmap 4.2) drops in behind the same
:class:`~src.perception.detection.Detector` interface later.

Heavy deps (ultralytics) are imported lazily so the rest of the project runs
without them; install with `pip install ultralytics`.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np

from src.perception.detection import Detection, FrameDetections, Detector

# Map a detector's own class *names* onto our schema, so a COCO model and a
# fine-tuned basketball model both work with no code change. Matched by
# substring against lowercased model.names, most-specific first.
_NAME_TO_CLASS = [
    ("referee", "referee"), ("ref", "referee"),
    ("rim", "rim"), ("hoop", "rim"), ("basket", "rim"), ("net", "rim"),
    ("basketball", "ball"), ("ball", "ball"), ("sports ball", "ball"),
    ("player", "player"), ("person", "player"),
]
_EMBED_BINS = 6  # torso colour histogram bins per channel (team clustering)


def _class_map(names: dict) -> dict:
    """{class_id -> our-class} from a model's names dict; unmatched ids dropped.

    Whole-word matching so "baseball bat" is not a ball and "refrigerator" is not
    a referee; multi-word needles match as a substring.
    """
    out = {}
    for cid, name in names.items():
        low = str(name).lower()
        words = set(re.split(r"[^a-z]+", low))
        for needle, ours in _NAME_TO_CLASS:
            hit = (needle in low) if " " in needle else (needle in words)
            if hit:
                out[int(cid)] = ours
                break
    return out


class VideoSource:
    """Thin OpenCV VideoCapture wrapper returning RGB frames."""

    def __init__(self, path: str):
        import cv2
        self.cv2 = cv2
        self.cap = cv2.VideoCapture(str(path))
        if not self.cap.isOpened():
            raise FileNotFoundError(f"cannot open video: {path}")
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 25.0
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    def frame_at_index(self, i: int):
        self.cap.set(self.cv2.CAP_PROP_POS_FRAMES, max(0, i))
        ok, bgr = self.cap.read()
        return self.cv2.cvtColor(bgr, self.cv2.COLOR_BGR2RGB) if ok else None

    def frame_at_time(self, seconds: float):
        return self.frame_at_index(int(round(seconds * self.fps)))

    def gray(self, rgb):
        return self.cv2.cvtColor(rgb, self.cv2.COLOR_RGB2GRAY)

    def release(self):
        self.cap.release()


def torso_embedding(rgb: np.ndarray, bbox) -> np.ndarray:
    """A cheap appearance embedding: colour histogram of the torso crop.

    Enough to cluster two team jerseys without SigLIP; the SigLIP crop embedding
    (roadmap 4.4) is a drop-in upgrade if the deps are available."""
    x1, y1, x2, y2 = [int(v) for v in bbox]
    # Torso ~ top 45%, central 60% of the box (avoids floor/shorts).
    h, w = y2 - y1, x2 - x1
    ty1, ty2 = y1 + int(0.15 * h), y1 + int(0.55 * h)
    tx1, tx2 = x1 + int(0.2 * w), x1 + int(0.8 * w)
    crop = rgb[max(0, ty1):max(1, ty2), max(0, tx1):max(1, tx2)]
    if crop.size == 0:
        return np.zeros(_EMBED_BINS ** 3, dtype=np.float32)
    idx = (crop.reshape(-1, 3).astype(int) * _EMBED_BINS // 256)
    flat = idx[:, 0] * _EMBED_BINS * _EMBED_BINS + idx[:, 1] * _EMBED_BINS + idx[:, 2]
    hist = np.bincount(flat, minlength=_EMBED_BINS ** 3).astype(np.float32)
    return hist / (hist.sum() + 1e-6)


# A fine-tuned basketball model, if present, is preferred over generic COCO.
_BASKETBALL_WEIGHTS = "models/basketball.pt"


def _default_weights() -> str:
    from pathlib import Path
    return _BASKETBALL_WEIGHTS if Path(_BASKETBALL_WEIGHTS).exists() else "yolo11l.pt"


@dataclass
class PretrainedYOLODetector(Detector):
    """YOLO adapter that maps the model's class *names* onto our schema.

    Works both with generic COCO (person -> player, sports ball -> ball) and with
    a fine-tuned basketball model (player/ball/hoop) dropped in at
    ``models/basketball.pt`` — no code change, classes are matched by name. A
    basketball model, being players-only, also skips the crowd/bench that COCO's
    ``person`` class boxes.

    Defaults to CPU inference: the installed torchvision (+cpu) mismatches a
    CUDA torch build, which breaks the NMS op on GPU. A few frames per pause on
    CPU is fine; install a matching torchvision+cuXXX to run YOLO on the GPU.
    """

    weights: str = field(default_factory=_default_weights)
    conf: float = 0.2          # player confidence
    ball_conf: float = 0.05    # the basketball is small/blurred — accept it far lower
    device: str = "cpu"
    _model: object = None
    _clsmap: dict = None

    def _load(self):
        if self._model is None:
            from ultralytics import YOLO  # lazy: heavy dep
            self._model = YOLO(self.weights)
            self._clsmap = _class_map(self._model.names)
        return self._model

    def detect(self, image: np.ndarray, frame_idx: int) -> FrameDetections:
        """image is RGB (as VideoSource returns); ultralytics wants BGR."""
        import cv2
        model = self._load()
        bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        # Predict at the lower (ball) threshold, then gate each class separately:
        # players at the strict `conf`, the ball at `ball_conf`. A low ball
        # threshold recovers the small/blurred, handler-defining ball; detectors
        # rarely false-positive a ball on a wood court.
        floor = min(self.conf, self.ball_conf)
        res = model.predict(bgr, conf=floor, device=self.device, verbose=False)[0]
        dets = []
        for box in res.boxes:
            ours = self._clsmap.get(int(box.cls[0]))
            if ours is None:
                continue
            xyxy = tuple(float(v) for v in box.xyxy[0]); c = float(box.conf[0])
            if ours == "player" and c >= self.conf:
                dets.append(Detection("player", xyxy, c, embedding=torso_embedding(image, xyxy)))
            elif ours == "ball" and c >= self.ball_conf:
                dets.append(Detection("ball", xyxy, c))
            elif ours in ("referee", "rim") and c >= self.conf:
                dets.append(Detection(ours, xyxy, c))
        return FrameDetections(frame_idx, dets)
