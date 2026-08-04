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

import os
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


def _match_class(name: str) -> str | None:
    """Map one detector class *name* onto our schema, or None if unrecognised.

    Whole-word matching so "baseball bat" is not a ball and "refrigerator" is not
    a referee; multi-word needles match as a substring.
    """
    low = str(name).lower()
    words = set(re.split(r"[^a-z]+", low))
    for needle, ours in _NAME_TO_CLASS:
        if (needle in low) if " " in needle else (needle in words):
            return ours
    return None


def _class_map(names: dict) -> dict:
    """{class_id -> our-class} from a model's names dict; unmatched ids dropped."""
    out = {}
    for cid, name in names.items():
        ours = _match_class(name)
        if ours is not None:
            out[int(cid)] = ours
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


def looks_like_ref(rgb: np.ndarray, bbox) -> bool:
    """True only if a torso is a clear NBA referee's *solid gray* uniform.

    Gray = nearly colourless (very low saturation) AND uniform mid-brightness
    (not a bright white jersey, not black, and low variance so a high-contrast
    two-tone jersey isn't mistaken for a ref). Deliberately conservative: removing
    a real player is worse than missing a ref. (No 'striped' rule — NBA refs wear
    gray, and that rule mis-flagged shadowed white/gold jerseys as refs.)
    """
    import cv2
    x1, y1, x2, y2 = [int(v) for v in bbox]
    h, w = y2 - y1, x2 - x1
    if h < 30 or w < 12:
        return False
    crop = rgb[max(0, y1 + int(0.15 * h)):y1 + int(0.5 * h),
               max(0, x1 + int(0.25 * w)):x1 + int(0.75 * w)]
    if crop.size == 0:
        return False
    hsv = cv2.cvtColor(crop, cv2.COLOR_RGB2HSV)
    sat = float(hsv[..., 1].mean())
    val = hsv[..., 2].astype(float)
    return sat < 40 and 70 < float(val.mean()) < 160 and float(val.std()) < 40


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


def _auto_device() -> str:
    """Use the GPU when a CUDA build of torch+torchvision is present, else CPU."""
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


@dataclass
class PretrainedYOLODetector(Detector):
    """YOLO adapter that maps the model's class *names* onto our schema.

    Works both with generic COCO (person -> player, sports ball -> ball) and with
    a fine-tuned basketball model (player/ball/hoop) dropped in at
    ``models/basketball.pt`` — no code change, classes are matched by name. A
    basketball model, being players-only, also skips the crowd/bench that COCO's
    ``person`` class boxes.

    Runs on the GPU automatically when a matching CUDA torch+torchvision is
    installed (``_auto_device``); falls back to CPU otherwise. (A torchvision
    +cpu against a CUDA torch breaks the NMS op on GPU — install a matching
    torchvision+cuXXX to enable it.)
    """

    weights: str = field(default_factory=_default_weights)
    conf: float = 0.2          # player confidence
    ball_conf: float = 0.05    # the basketball is small/blurred — accept it far lower
    device: str = field(default_factory=_auto_device)
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


def _roboflow_predictions(resp: dict) -> list:
    """Pull the detection list out of a Roboflow workflow response, defensively."""
    for out in resp.get("outputs", []):
        preds = out.get("predictions")
        if isinstance(preds, dict) and isinstance(preds.get("predictions"), list):
            return preds["predictions"]
        if isinstance(preds, list):
            return preds
    return []


@dataclass
class RoboflowWorkflowDetector(Detector):
    """Hosted detector via a Roboflow serverless *workflow* (no local weights).

    Calls the workflow REST endpoint with ``requests`` (so no `inference-sdk`
    dependency / Python-version pin), maps each prediction's ``class`` name onto
    our schema, and returns the same :class:`Detection` objects the rest of the
    pipeline consumes. The API key is read from ``$ROBOFLOW_API_KEY`` — never
    hard-coded, so it never lands in the repo. Each frame is one network call, so
    prefer a larger ``stride`` on the pause window.
    """

    workspace: str
    workflow_id: str
    classes: str = "ball, basket, person"
    api_url: str = "https://serverless.roboflow.com"
    api_key: str = field(default_factory=lambda: os.environ.get("ROBOFLOW_API_KEY", ""))
    conf: float = 0.2
    ball_conf: float = 0.05
    timeout: float = 120.0

    def detect(self, image: np.ndarray, frame_idx: int) -> FrameDetections:
        import base64
        import cv2
        import requests
        if not self.api_key:
            raise RuntimeError("set $ROBOFLOW_API_KEY to use RoboflowWorkflowDetector")
        ok, buf = cv2.imencode(".jpg", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        b64 = base64.b64encode(buf).decode()
        url = f"{self.api_url}/infer/workflows/{self.workspace}/{self.workflow_id}"
        body = {"api_key": self.api_key, "inputs": {
            "image": {"type": "base64", "value": b64}, "classes": self.classes}}
        r = requests.post(url, json=body, timeout=self.timeout)
        r.raise_for_status()
        dets = []
        for p in _roboflow_predictions(r.json()):
            ours = _match_class(p.get("class", ""))
            if ours is None:
                continue
            c = float(p.get("confidence", 0.0))
            x, y, w, h = p["x"], p["y"], p["width"], p["height"]
            bbox = (x - w / 2, y - h / 2, x + w / 2, y + h / 2)
            if ours == "player" and c >= self.conf:
                dets.append(Detection("player", bbox, c, embedding=torso_embedding(image, bbox)))
            elif ours == "ball" and c >= self.ball_conf:
                dets.append(Detection("ball", bbox, c))
            elif ours in ("referee", "rim") and c >= self.conf:
                dets.append(Detection(ours, bbox, c))
        return FrameDetections(frame_idx, dets)
