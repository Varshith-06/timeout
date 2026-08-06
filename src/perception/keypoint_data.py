"""Training data for the court-keypoint model (roadmap 4.6).

The model that this data trains is the last missing piece of per-frame
calibration. Today a clip is calibrated by clicking landmarks on ONE frame and
propagating that homography across the shot with optical flow
(:class:`~src.perception.calibrate.HomographyTracker`); the propagation drifts,
and it dies at every camera cut. A single-frame keypoint model has neither
problem — but it needs labels, and hand-labelling broadcast frames is exactly
the work the click-calibration was invented to avoid.

So the labels are generated from the calibrations we already have. A solved
``Calibration`` *is* a pixel<->court correspondence for its frame: invert its H
and every one of the 24 canonical keypoints lands at a known pixel, including
the ones nobody clicked. Two harvesting strategies fall out of that:

**Flow propagation** walks outward from the anchor frame, re-solving H each step,
and labels every frame it passes. Cheap and gives real camera motion, but the
labels inherit the tracker's drift — so each frame is gated on the tracker's own
reprojection error and on how far it has travelled from the anchor.

**Homography jitter** (:func:`warp_sample`) warps the anchor frame itself by a
random perspective transform and pushes the keypoints through the same matrix.
The labels stay *exact* no matter how aggressive the warp, because the warp is
the label transform. This is the clean signal; flow propagation supplies the
real-motion variety around it.

Honest limits: the labels are only as good as the click calibrations they come
from, and they cover the handful of camera shots that were calibrated on two
games. This trains a model for *these broadcasts*, not a general NBA court
model. Widening it means calibrating more shots on more games — the harvesting
code does not change.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.perception.calibrate import Calibration, HomographyTracker, _LANDMARK_COURT
from src.perception.cuts import CutDetector
from src.perception.keypoints import KEYPOINT_NAMES, KEYPOINT_XY

# A keypoint this far outside the image is unlabelable — a heatmap has no pixel
# to put it on. Slightly negative margins are allowed so points just off the edge
# still teach the model the court continues past the frame.
VISIBLE_MARGIN_PX = 8.0

# Stop propagating once the tracker's own fit degrades past this. The clicked
# points have known court coordinates, so the tracker can be scored every frame
# against the same reprojection metric the solver uses.
MAX_TRACKER_REPROJ_FT = 2.5


@dataclass
class LabelledFrame:
    """One frame with its court keypoints in pixels.

    ``kp_px`` is (K, 2) in the canonical keypoint order; entries where
    ``visible`` is False are meaningless and must not be supervised.
    """
    kp_px: np.ndarray            # (K, 2) float
    visible: np.ndarray          # (K,) bool
    source: str                  # which calibration + frame produced it
    reproj_ft: float             # tracker fit at this frame (0.0 at the anchor)
    image: np.ndarray | None = None   # RGB, when harvested in memory


def project_canonical(H_px_to_court: np.ndarray, img_size,
                      margin: float = VISIBLE_MARGIN_PX):
    """Canonical keypoints -> pixels, plus a visibility mask.

    ``H_px_to_court`` is the calibration's H (pixel -> court feet); this inverts
    it to place every canonical point on the image, whether or not it was ever
    clicked. Points that project outside the frame, or through a degenerate
    (near-singular) inverse, come back not-visible rather than as wild numbers.
    """
    import cv2
    w, h = int(img_size[0]), int(img_size[1])
    try:
        H_court_to_px = np.linalg.inv(np.asarray(H_px_to_court, dtype=np.float64))
    except np.linalg.LinAlgError:
        return np.zeros((len(KEYPOINT_XY), 2)), np.zeros(len(KEYPOINT_XY), dtype=bool)

    pts = cv2.perspectiveTransform(
        KEYPOINT_XY.reshape(-1, 1, 2).astype(np.float64), H_court_to_px).reshape(-1, 2)
    ok = np.isfinite(pts).all(axis=1)
    inside = ((pts[:, 0] >= -margin) & (pts[:, 0] < w + margin) &
              (pts[:, 1] >= -margin) & (pts[:, 1] < h + margin))
    return pts, ok & inside


def _tracker_reproj_ft(tracker: HomographyTracker) -> float:
    """Median reprojection error, feet, of the tracker's own tracked points.

    This is the propagation's self-check: the tracked points' court coordinates
    are known, so drift shows up here before it silently corrupts labels.
    """
    import cv2
    if tracker.H is None:
        return float("inf")
    px = tracker.pts.reshape(-1, 1, 2).astype(np.float64)
    proj = cv2.perspectiveTransform(px, tracker.H).reshape(-1, 2)
    err = np.linalg.norm(proj - tracker.court, axis=1)
    return float(np.median(err)) if len(err) else float("inf")


def harvest_calibration(video, calib: Calibration, stride: int = 5,
                        max_span_sec: float = 30.0, both_directions: bool = True,
                        max_reproj_ft: float = MAX_TRACKER_REPROJ_FT,
                        keep_images: bool = True) -> list[LabelledFrame]:
    """Label every frame in the camera shot around ``calib`` by flow propagation.

    Walks forward from the calibration's anchor time and (unless
    ``both_directions`` is False) backward as well, since the shot usually starts
    before the frame that happened to get clicked. Each direction stops at a
    camera cut, at ``max_span_sec``, or as soon as the tracker's reprojection
    error exceeds ``max_reproj_ft`` — whichever comes first. Optical flow is
    direction-agnostic, so walking backward is just walking with a negative step.
    """
    if calib.time is None:
        raise ValueError("calibration has no .time — re-run calibrate.py --time")

    anchor = int(round(calib.time * video.fps))
    img_size = calib.img_size if calib.img_size[0] else (video.width, video.height)
    out: list[LabelledFrame] = []
    steps = [stride, -stride] if both_directions else [stride]

    for step in steps:
        frame = video.frame_at_index(anchor)
        if frame is None:
            continue
        tracker = HomographyTracker(video.gray(frame), calib)
        cutdet = CutDetector(threshold=0.5)
        cutdet.update(_thumb(frame))
        max_frames = int(max_span_sec * video.fps / abs(step))

        for n in range(max_frames):
            idx = anchor + step * (n + 1)
            if idx < 0 or idx >= video.frame_count:
                break
            frame = video.frame_at_index(idx)
            if frame is None:
                break
            if cutdet.update(_thumb(frame)):
                break                                  # new shot: calibration void
            if tracker.update(video.gray(frame)) is None:
                break                                  # lost too many points
            err = _tracker_reproj_ft(tracker)
            if not np.isfinite(err) or err > max_reproj_ft:
                break                                  # drifted: stop before labels rot

            kp_px, vis = project_canonical(tracker.H, img_size)
            if vis.sum() < 4:
                continue                               # too little court in view
            out.append(LabelledFrame(kp_px, vis, f"{calib_name(calib)}@{idx}", err,
                                     frame.copy() if keep_images else None))

    # The anchor itself is the highest-quality sample: its H was solved from
    # clicks and visually accepted, with no propagation between it and the label.
    frame = video.frame_at_index(anchor)
    if frame is not None:
        kp_px, vis = project_canonical(calib.H, img_size)
        if vis.sum() >= 4:
            out.append(LabelledFrame(kp_px, vis, f"{calib_name(calib)}@anchor", 0.0,
                                     frame.copy() if keep_images else None))
    return out


def _thumb(rgb: np.ndarray) -> np.ndarray:
    import cv2
    return cv2.resize(rgb, (64, 36))


def calib_name(calib: Calibration) -> str:
    """Stable short name for a calibration, used to group train/val by shot."""
    return getattr(calib, "_name", None) or f"t{calib.time}"


# --- Augmentation ------------------------------------------------------------
def warp_sample(image: np.ndarray, kp_px: np.ndarray, visible: np.ndarray,
                rng: np.random.Generator, max_shift: float = 0.06,
                scale_range: tuple = (0.85, 1.18), max_rot_deg: float = 4.0):
    """Random perspective warp of a frame, with labels carried through exactly.

    The four image corners are jittered independently (that is what makes it a
    perspective rather than affine warp — it emulates the camera panning to a new
    viewpoint), then scale and a small roll are composed on top. Because the
    keypoints are pushed through the *same* matrix, the warped labels are exact:
    this manufactures new camera geometry without manufacturing label error.

    Points warped outside the frame are marked not-visible, so the model is never
    asked to place a heatmap peak on a pixel that does not exist.
    """
    import cv2
    h, w = image.shape[:2]
    src = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)

    jitter = rng.uniform(-max_shift, max_shift, (4, 2)) * np.array([w, h], dtype=float)
    dst = src + jitter.astype(np.float32)

    # Compose scale + roll about the image centre.
    s = float(rng.uniform(*scale_range))
    ang = float(rng.uniform(-max_rot_deg, max_rot_deg))
    M = cv2.getRotationMatrix2D((w / 2, h / 2), ang, s)
    dst = cv2.transform(dst.reshape(-1, 1, 2), M).reshape(-1, 2).astype(np.float32)

    W = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(image, W, (w, h), flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_REPLICATE)
    kp_w = cv2.perspectiveTransform(
        np.asarray(kp_px, dtype=np.float64).reshape(-1, 1, 2), W.astype(np.float64)).reshape(-1, 2)

    vis = (visible & np.isfinite(kp_w).all(axis=1) &
           (kp_w[:, 0] >= -VISIBLE_MARGIN_PX) & (kp_w[:, 0] < w + VISIBLE_MARGIN_PX) &
           (kp_w[:, 1] >= -VISIBLE_MARGIN_PX) & (kp_w[:, 1] < h + VISIBLE_MARGIN_PX))
    return warped, kp_w, vis


def jitter_colour(image: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Brightness/contrast/saturation jitter — arenas differ, geometry does not."""
    import cv2
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV).astype(np.float32)
    hsv[..., 1] *= float(rng.uniform(0.7, 1.3))          # saturation
    hsv[..., 2] *= float(rng.uniform(0.7, 1.3))          # value
    hsv = np.clip(hsv, 0, 255).astype(np.uint8)
    out = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB).astype(np.float32)
    out = (out - 128.0) * float(rng.uniform(0.85, 1.15)) + 128.0    # contrast
    return np.clip(out, 0, 255).astype(np.uint8)


# --- On-disk dataset ---------------------------------------------------------
def export_images(samples: list[LabelledFrame], out_dir, offset: int = 0,
                  jpeg_quality: int = 92) -> list[dict]:
    """Write one batch of frames as JPEGs and return their label records.

    Split from the manifest write so a caller can harvest and flush **one shot at
    a time**: a full harvest holds every frame as a 2.7 MB RGB array, which runs
    into gigabytes once more than a few shots are calibrated — precisely what
    widening the training set requires. ``offset`` keeps filenames unique across
    batches.

    Frames are kept at their native size; the model resizes at load time, so the
    same dataset can train models at different input resolutions.
    """
    import cv2
    out_dir = Path(out_dir)
    (out_dir / "images").mkdir(parents=True, exist_ok=True)
    records = []
    for i, s in enumerate(samples):
        if s.image is None:
            continue
        name = f"{offset + i:06d}.jpg"
        cv2.imwrite(str(out_dir / "images" / name),
                    cv2.cvtColor(s.image, cv2.COLOR_RGB2BGR),
                    [cv2.IMWRITE_JPEG_QUALITY, int(jpeg_quality)])
        records.append({
            "image": name,
            "kp_px": [[round(float(x), 2), round(float(y), 2)] for x, y in s.kp_px],
            "visible": [bool(v) for v in s.visible],
            "source": s.source,
            "reproj_ft": round(float(s.reproj_ft), 3),
        })
    return records


def write_manifest(records: list[dict], out_dir) -> dict:
    """Write the labels manifest for already-exported images."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {"keypoint_names": KEYPOINT_NAMES, "records": records}
    (out_dir / "labels.json").write_text(json.dumps(manifest), encoding="utf-8")
    return manifest


def export_dataset(samples: list[LabelledFrame], out_dir, jpeg_quality: int = 92) -> dict:
    """Write harvested frames to ``out_dir`` as JPEGs + a labels manifest."""
    return write_manifest(export_images(samples, out_dir, 0, jpeg_quality), out_dir)


def load_dataset(out_dir) -> dict:
    """Read the manifest written by :func:`export_dataset`."""
    return json.loads((Path(out_dir) / "labels.json").read_text(encoding="utf-8"))
