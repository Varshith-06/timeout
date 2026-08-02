"""Synthetic broadcast generator — paired pixel<->court data (roadmap 4.1, 5.2).

Projects the court-space tracking we already have (a Phase 1 possession) through
the pinhole camera to produce what a broadcast CV front-end would see: noisy
pixel detections with realistic misses and false positives, partially-visible
court keypoints, a camera that pans with the ball and occasionally cuts, and a
noisy scoreboard clock. Crucially it also emits the *ground-truth* court
positions, giving the paired data the roadmap's domain-gap trick (5.2) needs to
measure position/identity error without any real footage.

Nothing here is a learned model; it is a controlled stand-in for the YOLO +
keypoint + OCR front-end so the geometry/tracking/state pipeline is testable.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from src.perception.camera import BroadcastCamera
from src.perception.detection import Detection, FrameDetections
from src.perception.keypoints import KEYPOINT_XY

# Detector characteristics (roadmap 4.2 targets).
PLAYER_RECALL = 0.97
BALL_RECALL = 0.70            # small, fast, occluded ~half the time
PLAYER_PIX_NOISE = 3.0
BALL_PIX_NOISE = 9.0          # motion blur
KP_PIX_NOISE = 2.0
FALSE_POSITIVE_RATE = 0.03    # spurious player boxes per frame
EMBED_DIM = 16


@dataclass
class BroadcastFrame:
    frame_idx: int
    camera: BroadcastCamera
    detections: FrameDetections
    kp_pixels: np.ndarray        # (M,2) detected keypoint pixels
    kp_court: np.ndarray         # (M,2) their canonical court coords
    kp_conf: np.ndarray          # (M,) confidences
    clock_read: float | None     # noisy shot-clock OCR reading
    cut: bool                    # True if a camera cut occurred at this frame
    image: np.ndarray | None = None   # real broadcast RGB frame (real video only)


@dataclass
class BroadcastClip:
    frames: list[BroadcastFrame] = field(default_factory=list)
    truth: list[dict] = field(default_factory=list)  # per-frame ground-truth court state
    team_embeddings: dict = field(default_factory=dict)  # team_id -> base vector


def _player_box(foot_px, depth):
    """A player box whose height scales inversely with depth (perspective)."""
    h = np.clip(2200.0 / max(depth, 1.0), 24, 260)
    w = h * 0.45
    fx, fy = foot_px
    return (fx - w / 2, fy - h, fx + w / 2, fy)


JERSEY_ILLEGIBLE_RATE = 0.45   # jersey unreadable in a given frame (roadmap 4.5)
JERSEY_MISREAD_RATE = 0.10     # legible but wrong digit


def generate_broadcast(possession, seed: int = 0, cut_prob: float = 0.0,
                       stride: int = 5, jersey_map: dict | None = None) -> BroadcastClip:
    """Build a broadcast clip from a Phase 1 :class:`Possession`.

    stride subsamples the 25 Hz tracking to a broadcast-like frame rate. cut_prob
    is the per-frame probability of a camera cut (replay/close-up). jersey_map
    (player_id -> jersey) lets the detector emit per-frame jersey OCR reads.
    """
    rng = np.random.default_rng(seed)
    jersey_map = jersey_map or {}
    base_cam = BroadcastCamera()

    # Distinct appearance bases: two teams + referees, well separated in embedding
    # space (emulates SigLIP torso crops clustering cleanly, roadmap 4.4).
    teams = sorted({p.team_id for fr in possession.frames for p in fr.players})
    team_base = {t: rng.normal(0, 1, EMBED_DIM) + i * 6.0 for i, t in enumerate(teams)}
    ref_base = rng.normal(0, 1, EMBED_DIM) - 8.0

    clip = BroadcastClip(team_embeddings=team_base)
    frames = possession.frames[::stride]

    # Reference rims and referees (static-ish) for extra detections.
    ref_positions = np.array([[30.0, 6.0], [60.0, 44.0]])

    for i, fr in enumerate(frames):
        cut = rng.random() < cut_prob
        # Camera pans toward the ball and breathes in zoom; a cut jumps it hard.
        ball_x = fr.ball_x
        pan = np.clip((ball_x - 47) * 0.4, -25, 25) + (rng.uniform(-40, 40) if cut else rng.normal(0, 1.5))
        zoom = rng.uniform(0.7, 1.4) if cut else 1.0 + 0.06 * np.sin(i / 6.0)
        cam = base_cam.panned(pan_deg=pan, zoom=zoom)

        dets: list[Detection] = []
        truth = {"players": {}, "ball": (fr.ball_x, fr.ball_y)}

        # Players.
        for p in fr.players:
            truth["players"][p.player_id] = {
                "x": p.x, "y": p.y, "team_id": p.team_id, "side": p.side,
            }
            foot_px = cam.project(np.array([[p.x, p.y]]))[0]
            if not cam.in_view(foot_px[None], margin=40)[0]:
                continue
            if rng.random() > PLAYER_RECALL:      # missed detection
                continue
            depth = np.linalg.norm(np.array([p.x, p.y, 0.0]) - cam.position)
            foot_n = foot_px + rng.normal(0, PLAYER_PIX_NOISE, 2)
            # Jersey OCR read: often illegible, occasionally misread.
            true_jersey = jersey_map.get(p.player_id)
            if true_jersey is None or rng.random() < JERSEY_ILLEGIBLE_RATE:
                jersey_read = None
            elif rng.random() < JERSEY_MISREAD_RATE:
                jersey_read = int(rng.integers(0, 100))
            else:
                jersey_read = int(true_jersey)
            det = Detection("player", _player_box(foot_n, depth), float(rng.uniform(0.6, 0.95)),
                            embedding=team_base[p.team_id] + rng.normal(0, 1.2, EMBED_DIM),
                            jersey_read=jersey_read,
                            true_player_id=p.player_id,
                            true_team=("home" if p.team_id == teams[0] else "away"))
            dets.append(det)

        # Ball (project its 3D point; lower recall + motion blur).
        if rng.random() < BALL_RECALL:
            ball_px = cam.project3d(np.array([[fr.ball_x, fr.ball_y, max(fr.ball_z, 0.0)]]))[0]
            if cam.in_view(ball_px[None], margin=20)[0]:
                b = ball_px + rng.normal(0, BALL_PIX_NOISE, 2)
                dets.append(Detection("ball", (b[0] - 6, b[1] - 6, b[0] + 6, b[1] + 6),
                                      float(rng.uniform(0.3, 0.7))))

        # Referees.
        for rp in ref_positions:
            rpx = cam.project(rp[None])[0]
            if cam.in_view(rpx[None], margin=20)[0] and rng.random() < 0.9:
                depth = np.linalg.norm(np.array([rp[0], rp[1], 0.0]) - cam.position)
                det = Detection("referee", _player_box(rpx + rng.normal(0, 3, 2), depth),
                                float(rng.uniform(0.5, 0.9)),
                                embedding=ref_base + rng.normal(0, 1.0, EMBED_DIM))
                dets.append(det)

        # False-positive player boxes.
        if rng.random() < FALSE_POSITIVE_RATE:
            fx, fy = rng.uniform(0, cam.img_w), rng.uniform(cam.img_h * 0.4, cam.img_h)
            dets.append(Detection("player", _player_box((fx, fy), 60), float(rng.uniform(0.3, 0.5))))

        # Court keypoints in view, with noise; some dropped (occlusion / low conf).
        kp_px = cam.project(KEYPOINT_XY)
        vis = cam.in_view(kp_px, margin=0)
        keep = vis & (rng.random(len(KEYPOINT_XY)) < 0.85)
        kp_pixels = kp_px[keep] + rng.normal(0, KP_PIX_NOISE, (int(keep.sum()), 2))
        kp_court = KEYPOINT_XY[keep]
        kp_conf = rng.uniform(0.4, 0.97, int(keep.sum()))
        if cut:  # after a cut the court model is unreliable
            kp_conf *= 0.2

        # Shot-clock OCR: mostly correct, occasional gross misread.
        clock = fr.shot_clock
        if clock is None:
            clock_read = None
        elif rng.random() < 0.05:
            clock_read = float(rng.uniform(0, 24))      # OCR blunder
        else:
            clock_read = round(clock + rng.normal(0, 0.2))

        clip.frames.append(BroadcastFrame(
            frame_idx=i, camera=cam,
            detections=FrameDetections(i, dets),
            kp_pixels=kp_pixels, kp_court=kp_court, kp_conf=kp_conf,
            clock_read=clock_read, cut=cut,
        ))
        clip.truth.append(truth)

    return clip
