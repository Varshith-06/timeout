"""Learned court-keypoint detector — per-frame calibration (roadmap 4.6).

Predicts where each of the 24 canonical court landmarks sits in a broadcast
frame, so the homography can be re-solved *every frame* instead of clicked once
and propagated. That is what fixes drift within a camera shot, and it is what
lets a shot the user never calibrated be used at all.

Heatmap regression rather than direct coordinate regression, for two reasons
that matter downstream. It is fully convolutional, so a landmark keeps its
spatial evidence (a line crossing) instead of being squeezed through a fully
connected layer. And the peak height is a natural per-point confidence, which is
exactly what :func:`~src.perception.homography.solve_homography` already expects
— occluded or off-frame landmarks arrive with low confidence and get dropped by
the same RANSAC + reprojection gate the synthetic path uses. No new failure
mode is introduced; a bad frame simply produces too few confident points and the
solver reports ``ok=False``.

The network is deliberately small (~2M params, a U-Net-ish encoder/decoder at
quarter resolution). The court is a rigid, high-contrast, low-variety target —
capacity is not the bottleneck, labelled data is.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from src.perception.keypoints import KEYPOINT_NAMES, KEYPOINT_XY

K = len(KEYPOINT_NAMES)
IN_W, IN_H = 512, 288          # 16:9, matches 1280x720 broadcast frames
OUT_STRIDE = 4                 # heatmaps at 128x72
HM_W, HM_H = IN_W // OUT_STRIDE, IN_H // OUT_STRIDE
TARGET_SIGMA = 1.8             # gaussian sigma in heatmap pixels

DEFAULT_WEIGHTS = "models/court_keypoints.pt"


# --- Target encoding / decoding (pure, unit-tested) --------------------------
def gaussian_targets(kp_px: np.ndarray, visible: np.ndarray, img_size,
                     sigma: float = TARGET_SIGMA,
                     supervise_absent: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """Build (K, HM_H, HM_W) gaussian heatmaps and a (K,) supervision mask.

    Keypoints are mapped from source-image pixels into heatmap coordinates. A
    point whose gaussian centre falls outside the heatmap gets an all-zero
    target rather than being clamped to the edge — clamping would teach the
    model that every off-screen landmark lives on the border.

    ``supervise_absent`` decides what happens to those off-frame landmarks, and
    it is the difference between a usable confidence and a meaningless one.
    Visibility here is *geometric*: the calibration tells us exactly whether a
    landmark is inside the frame, so "absent" is a known fact, not missing
    information. Supervising it toward an empty heatmap is what teaches the
    model to answer "not in this frame" with a low peak. Leaving it unsupervised
    (``False``) means the network is free to emit a confident peak for a
    landmark that is nowhere in the image, and those bogus correspondences go
    straight into RANSAC.

    Occlusion is deliberately *not* treated as absence: a landmark hidden behind
    a player is still geometrically in-frame, keeps its gaussian, and the model
    is expected to infer it from the surrounding lines.
    """
    w, h = float(img_size[0]), float(img_size[1])
    sx, sy = HM_W / w, HM_H / h
    hm = np.zeros((K, HM_H, HM_W), dtype=np.float32)
    mask = np.zeros(K, dtype=np.float32)

    yy, xx = np.mgrid[0:HM_H, 0:HM_W]
    for k in range(K):
        cx, cy = float(kp_px[k, 0]) * sx, float(kp_px[k, 1]) * sy
        present = bool(visible[k]) and (0 <= cx < HM_W) and (0 <= cy < HM_H)
        if present:
            hm[k] = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2.0 * sigma ** 2))
            mask[k] = 1.0
        elif supervise_absent:
            mask[k] = 1.0            # target stays all-zero: "this one is not here"
    return hm, mask


def decode_heatmaps(hm: np.ndarray, img_size) -> tuple[np.ndarray, np.ndarray]:
    """(K, h, w) heatmaps -> (K,2) pixel coords in the source image + (K,) conf.

    The integer argmax is refined to sub-pixel by fitting a parabola to the log
    of the three samples straddling the peak, independently per axis. A gaussian
    is exactly a parabola in the log domain, so against a clean target this
    recovers the true centre with zero error — the decoder contributes nothing to
    the model's reported accuracy. (An intensity-weighted centroid, the obvious
    alternative, is biased toward the peak cell and costs ~1-2 source pixels at
    this resolution, where one heatmap cell spans ten source pixels.)

    Cell ``i`` denotes coordinate ``i`` exactly, matching how
    :func:`gaussian_targets` lays the gaussians down against an integer grid.
    Adding a half-cell here would bias every point by half a cell.
    """
    hm = np.asarray(hm, dtype=np.float64)
    kk, hh, ww = hm.shape
    w, h = float(img_size[0]), float(img_size[1])
    pts = np.zeros((kk, 2), dtype=float)
    conf = np.zeros(kk, dtype=float)

    for k in range(kk):
        flat = int(np.argmax(hm[k]))
        py, px = divmod(flat, ww)
        conf[k] = float(np.clip(hm[k, py, px], 0.0, 1.0))
        pts[k] = (_parabolic(hm[k, py, max(0, px - 1):px + 2], px, px > 0, px < ww - 1),
                  _parabolic(hm[k, max(0, py - 1):py + 2, px], py, py > 0, py < hh - 1))

    pts[:, 0] *= w / ww
    pts[:, 1] *= h / hh
    return pts, conf


def _parabolic(triple: np.ndarray, peak: int, has_left: bool, has_right: bool) -> float:
    """Sub-cell peak position from three samples, via a log-domain parabola fit.

    Falls back to the integer peak at the array edge, on a non-positive sample
    (the network is unactivated, so it can emit negatives), or when the fit is
    degenerate or implausible — a true refinement never moves more than half a
    cell, so anything larger means the neighbourhood was not peak-shaped.
    """
    if not (has_left and has_right) or len(triple) < 3:
        return float(peak)
    lo, mid, hi = (float(v) for v in triple[:3])
    if min(lo, mid, hi) <= 1e-12:
        return float(peak)
    lo, mid, hi = np.log(lo), np.log(mid), np.log(hi)
    denom = lo - 2.0 * mid + hi
    if abs(denom) < 1e-9:
        return float(peak)
    delta = -0.5 * (hi - lo) / denom
    return float(peak + delta) if abs(delta) <= 0.5 else float(peak)


# --- Network -----------------------------------------------------------------
def _build_modules():
    """Import torch lazily so the rest of the package works without it."""
    import torch
    import torch.nn as nn
    return torch, nn


def _conv_block(nn, cin, cout, stride=1):
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, stride=stride, padding=1, bias=False),
        nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
        nn.Conv2d(cout, cout, 3, padding=1, bias=False),
        nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
    )


def build_net():
    """U-Net-ish encoder/decoder emitting K heatmaps at 1/4 input resolution."""
    torch, nn = _build_modules()

    class CourtKeypointNet(nn.Module):
        def __init__(self, k: int = K):
            super().__init__()
            self.enc1 = _conv_block(nn, 3, 32, stride=2)      # 1/2
            self.enc2 = _conv_block(nn, 32, 64, stride=2)     # 1/4
            self.enc3 = _conv_block(nn, 64, 128, stride=2)    # 1/8
            self.enc4 = _conv_block(nn, 128, 192, stride=2)   # 1/16
            self.up3 = nn.ConvTranspose2d(192, 128, 2, stride=2)
            self.dec3 = _conv_block(nn, 128 + 128, 128)
            self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
            self.dec2 = _conv_block(nn, 64 + 64, 64)
            self.head = nn.Conv2d(64, k, 1)

        def forward(self, x):
            e1 = self.enc1(x)
            e2 = self.enc2(e1)
            e3 = self.enc3(e2)
            e4 = self.enc4(e3)
            d3 = self.dec3(torch.cat([self.up3(e4), e3], dim=1))
            d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
            return self.head(d2)                              # (B, K, HM_H, HM_W)

    return CourtKeypointNet()


POS_WEIGHT = 120.0     # see masked_mse: without this the model predicts nothing


def masked_mse(pred, target, mask, pos_weight: float = POS_WEIGHT):
    """Foreground-weighted MSE over supervised channels only.

    ``mask`` is (B, K): a keypoint that is off-frame in this sample contributes
    nothing. Without it the model would be trained to predict an empty heatmap
    for every occluded landmark, which is a different (and wrong) task — we want
    "I cannot see it", expressed as a low peak, not "it is definitely absent".

    ``pos_weight`` is not a tuning knob, it is a correctness fix. A gaussian peak
    covers ~30 of a heatmap's 9216 cells, so unweighted MSE is minimised by
    emitting zeros everywhere: that degenerate solution scores ~0.0011 here,
    which is where training actually converged before this weighting existed.
    Scaling each cell's error by ``1 + pos_weight * target`` makes the peak worth
    more than the background it is buried in.
    """
    torch, _ = _build_modules()
    w = 1.0 + pos_weight * target                    # (B,K,h,w)
    per_px = w * (pred - target) ** 2
    per_kp = per_px.sum(dim=(2, 3)) / w.sum(dim=(2, 3)).clamp(min=1e-6)
    denom = mask.sum().clamp(min=1.0)
    return (per_kp * mask).sum() / denom


def preprocess(rgb: np.ndarray) -> np.ndarray:
    """RGB uint8 image -> (3, IN_H, IN_W) float array in [0,1], ImageNet-normed."""
    import cv2
    img = cv2.resize(rgb, (IN_W, IN_H), interpolation=cv2.INTER_LINEAR)
    x = img.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    x = (x - mean) / std
    return np.transpose(x, (2, 0, 1))


def save_checkpoint(net, path, meta: dict | None = None):
    torch, _ = _build_modules()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": net.state_dict(),
                "keypoint_names": KEYPOINT_NAMES,
                "in_size": [IN_W, IN_H], "out_stride": OUT_STRIDE,
                "meta": meta or {}}, str(path))


def load_checkpoint(path, device="cpu"):
    torch, _ = _build_modules()
    ckpt = torch.load(str(path), map_location=device, weights_only=False)
    net = build_net()
    net.load_state_dict(ckpt["state_dict"])
    net.to(device).eval()
    return net, ckpt


def _auto_device() -> str:
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


@dataclass
class CourtKeypointDetector:
    """Per-frame court keypoints, in the shape the homography solver wants.

    :meth:`predict` returns only the points that clear ``conf_thresh``, paired
    with their canonical court coordinates — the same
    ``(kp_pixels, kp_court, kp_conf)`` triple the synthetic broadcast emits, so
    it drops straight into ``BroadcastFrame`` with no other change.
    """

    weights: str = DEFAULT_WEIGHTS
    device: str = field(default_factory=_auto_device)
    # Matches solve_homography's own conf_thresh, so what this returns is what
    # actually gets used. A lower value here would report points the solver then
    # silently discards, making the returned set misleading.
    conf_thresh: float = 0.5
    _net: object = None

    @property
    def available(self) -> bool:
        return Path(self.weights).exists()

    def _load(self):
        if self._net is None:
            if not self.available:
                raise FileNotFoundError(
                    f"no court-keypoint weights at {self.weights} — train them with "
                    f"scripts/train_keypoints.py, or run with the manual calibration")
            self._net, _ = load_checkpoint(self.weights, self.device)
        return self._net

    def predict(self, rgb: np.ndarray):
        """One frame -> (pixels (M,2), court (M,2), conf (M,)) above threshold."""
        pts, conf = self.predict_all(rgb)
        keep = conf >= self.conf_thresh
        return pts[keep], KEYPOINT_XY[keep], conf[keep]

    def predict_all(self, rgb: np.ndarray):
        """All K keypoints with confidences, unfiltered (diagnostics/eval)."""
        torch, _ = _build_modules()
        net = self._load()
        h, w = rgb.shape[:2]
        x = torch.from_numpy(preprocess(rgb)).unsqueeze(0).to(self.device)
        with torch.no_grad():
            hm = net(x)[0].float().cpu().numpy()
        return decode_heatmaps(hm, (w, h))

    def predict_batch(self, frames: list[np.ndarray], batch_size: int = 16):
        """Batched inference — the GPU is idle one frame at a time."""
        torch, _ = _build_modules()
        net = self._load()
        out = []
        for i in range(0, len(frames), batch_size):
            chunk = frames[i:i + batch_size]
            x = torch.from_numpy(np.stack([preprocess(f) for f in chunk])).to(self.device)
            with torch.no_grad():
                hms = net(x).float().cpu().numpy()
            for f, hm in zip(chunk, hms):
                h, w = f.shape[:2]
                pts, conf = decode_heatmaps(hm, (w, h))
                keep = conf >= self.conf_thresh
                out.append((pts[keep], KEYPOINT_XY[keep], conf[keep]))
        return out
