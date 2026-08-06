"""Roadmap 4.6 tests: the learned court-keypoint path (per-frame calibration).

Covers the pure, deterministic parts — target encoding, sub-pixel decoding,
augmentation label algebra, and the fallback contract in the real-video builder.
Training itself is not tested here (it needs video and a GPU); what is tested is
everything that would silently corrupt a trained model's output.
"""
import numpy as np
import pytest

from src.perception.homography import (plausible_court_homography, project_points,
                                       solve_homography)
from src.perception.keypoint_data import project_canonical, warp_sample
from src.perception.keypoint_model import (K, decode_heatmaps, gaussian_targets,
                                           masked_mse)
from src.perception.keypoints import KEYPOINT_XY

IMG = (1280, 720)


def _spread_keypoints(seed=0, n=K):
    rng = np.random.default_rng(seed)
    return np.column_stack([rng.uniform(80, IMG[0] - 80, n),
                            rng.uniform(80, IMG[1] - 80, n)])


# --- Encode / decode ---------------------------------------------------------
def test_heatmap_round_trip_is_subpixel_exact():
    """decode(encode(p)) == p. A gaussian is a parabola in log space, so the
    sub-pixel fit is exact — any regression here silently costs court accuracy."""
    kp = _spread_keypoints()
    hm, mask = gaussian_targets(kp, np.ones(K, bool), IMG)
    pts, conf = decode_heatmaps(hm, IMG)
    assert mask.sum() == K
    assert np.allclose(pts, kp, atol=0.05)
    assert conf.min() > 0.9


def test_decode_survives_degenerate_heatmaps():
    """An untrained or saturated network must not produce NaNs or exceptions."""
    for hm in (np.zeros((K, 72, 128)), np.full((K, 72, 128), -1.0),
               np.full((K, 72, 128), np.nan)):
        pts, conf = decode_heatmaps(hm, IMG)
        assert pts.shape == (K, 2) and conf.shape == (K,)
        assert np.isfinite(pts).all()
        assert ((conf >= 0.0) & (conf <= 1.0)).all() or np.isnan(hm).any()


def test_absent_keypoints_are_supervised_toward_empty():
    """Off-frame landmarks get an all-zero target, not zero supervision.

    Geometric visibility is known, so absence is a fact to be taught. Leaving
    these unsupervised lets the model emit a confident peak for a landmark that
    is nowhere in the frame, and RANSAC then consumes that as a correspondence.
    """
    kp = _spread_keypoints()
    vis = np.ones(K, bool)
    vis[:5] = False
    hm, mask = gaussian_targets(kp, vis, IMG)
    assert hm[:5].max() == 0.0          # target is empty ...
    assert mask[:5].sum() == 5          # ... but still supervised
    assert mask.sum() == K


def test_absent_supervision_can_be_disabled():
    kp = _spread_keypoints()
    vis = np.ones(K, bool)
    vis[:5] = False
    _, mask = gaussian_targets(kp, vis, IMG, supervise_absent=False)
    assert mask[:5].sum() == 0
    assert mask[5:].sum() == K - 5


def test_offscreen_keypoint_is_dropped_not_clamped():
    """A point outside the frame must not be pinned to the border — that would
    teach the model every occluded landmark lives on the edge."""
    kp = _spread_keypoints()
    kp[0] = (-500.0, -500.0)
    hm, _ = gaussian_targets(kp, np.ones(K, bool), IMG)
    assert hm[0].max() == 0.0


def test_masked_mse_rejects_the_all_zeros_solution():
    """The foreground weighting exists so predicting nothing is NOT optimal.

    Without it, unweighted MSE is minimised by an empty heatmap (the peak covers
    ~30 of 9216 cells), which is exactly where training collapsed before.
    """
    torch = pytest.importorskip("torch")
    kp = _spread_keypoints()
    hm, mask = gaussian_targets(kp, np.ones(K, bool), IMG)
    target = torch.from_numpy(hm).unsqueeze(0)
    m = torch.from_numpy(mask).unsqueeze(0)

    zeros = torch.zeros_like(target)
    perfect = target.clone()
    assert masked_mse(perfect, target, m).item() < masked_mse(zeros, target, m).item() / 100


# --- Augmentation ------------------------------------------------------------
def test_warp_carries_labels_through_exactly():
    """The warp IS the label transform, so warped labels stay exact.

    Verified independently: warping a point set and re-deriving the homography
    from four warped corners must reproject the rest to the same places.
    """
    rng = np.random.default_rng(7)
    img = (rng.random((IMG[1], IMG[0], 3)) * 255).astype(np.uint8)
    kp = _spread_keypoints(seed=2)
    vis = np.ones(K, bool)

    _, kp_w, vis_w = warp_sample(img, kp, vis, np.random.default_rng(3))
    assert kp_w.shape == kp.shape
    assert np.isfinite(kp_w[vis_w]).all()

    # A perspective warp preserves collinearity and cross-ratio: fit the map from
    # 4 warped points and every other visible point must land within a pixel.
    idx = np.where(vis_w)[0]
    if len(idx) >= 8:
        import cv2
        H, _ = cv2.findHomography(kp[idx[:4]], kp_w[idx[:4]], 0)
        rest = idx[4:]
        pred = project_points(H, kp[rest])
        assert np.allclose(pred, kp_w[rest], atol=1.0)


def test_warp_marks_pushed_out_points_invisible():
    """Points warped off-frame must lose visibility, never keep a stale label."""
    rng = np.random.default_rng(11)
    img = (rng.random((IMG[1], IMG[0], 3)) * 255).astype(np.uint8)
    kp = np.column_stack([np.full(K, 5.0), np.linspace(5, IMG[1] - 5, K)])
    _, kp_w, vis_w = warp_sample(img, kp, np.ones(K, bool), np.random.default_rng(4),
                                 max_shift=0.0, scale_range=(3.0, 3.0), max_rot_deg=0.0)
    assert vis_w.sum() < K          # a 3x zoom pushes the left edge out of frame


# --- Label generation from a calibration -------------------------------------
def test_project_canonical_matches_the_homography():
    """Labels are inv(H) applied to the canonical points — verify against H."""
    # A synthetic but realistic court->pixel mapping.
    court = np.array([[0, 0], [94, 0], [94, 50], [0, 50]], dtype=np.float64)
    pix = np.array([[150, 640], [1140, 640], [980, 300], [310, 300]], dtype=np.float64)
    import cv2
    H_px2court, _ = cv2.findHomography(pix, court, 0)

    pts, vis = project_canonical(H_px2court, IMG)
    assert vis.sum() >= 8
    # Round-tripping the visible labels through H must return the canonical court
    # coordinates they were generated from.
    back = project_points(H_px2court, pts[vis])
    assert np.allclose(back, KEYPOINT_XY[vis], atol=1e-6)


def test_project_canonical_handles_singular_homography():
    pts, vis = project_canonical(np.zeros((3, 3)), IMG)
    assert vis.sum() == 0
    assert np.isfinite(pts).all()


def test_predicted_keypoints_solve_a_homography():
    """End of the contract: decoded keypoints feed solve_homography unchanged."""
    court = np.array([[0, 0], [94, 0], [94, 50], [0, 50]], dtype=np.float64)
    pix = np.array([[150, 640], [1140, 640], [980, 300], [310, 300]], dtype=np.float64)
    import cv2
    H_px2court, _ = cv2.findHomography(pix, court, 0)
    pts, vis = project_canonical(H_px2court, IMG)

    hm, _ = gaussian_targets(pts, vis, IMG)
    dec, conf = decode_heatmaps(hm, IMG)
    keep = conf >= 0.3
    res = solve_homography(dec[keep], KEYPOINT_XY[keep], conf[keep])
    assert res.ok
    assert res.median_reproj_ft < 0.5


# --- Solver robustness (regression) ------------------------------------------
def test_solve_homography_survives_empty_consensus():
    """RANSAC can return an H with no inliers; that must be a clean failure.

    Previously this raised AttributeError from cv2.perspectiveTransform returning
    None for an empty point set — a crash on real video, not just in eval.
    """
    pts = np.zeros((6, 2))            # fully degenerate: all points identical
    court = np.zeros((6, 2))
    res = solve_homography(pts, court)
    assert res.ok is False
    assert res.H is None or np.isfinite(res.H).all()


def test_project_points_accepts_empty_input():
    out = project_points(np.eye(3), np.empty((0, 2)))
    assert out.shape == (0, 2)


# --- Geometric plausibility --------------------------------------------------
def _broadcast_homography():
    """A realistic pixel->court mapping for a 1280x720 broadcast frame."""
    import cv2
    court = np.array([[0, 0], [94, 0], [94, 50], [0, 50]], dtype=np.float64)
    pix = np.array([[150, 640], [1140, 640], [980, 300], [310, 300]], dtype=np.float64)
    H, _ = cv2.findHomography(pix, court, 0)
    return H


def test_plausible_accepts_a_real_broadcast_view():
    assert plausible_court_homography(_broadcast_homography(), IMG)


def test_plausible_rejects_degenerate_homographies():
    assert not plausible_court_homography(np.eye(3), IMG)
    assert not plausible_court_homography(np.zeros((3, 3)), IMG)
    assert not plausible_court_homography(None, IMG)
    assert not plausible_court_homography(np.full((3, 3), np.nan), IMG)


def test_plausible_rejects_an_absurdly_small_court():
    """The observed failure mode: a self-consistent H that puts the whole 94 ft
    court in a few hundred pixels. Reprojection error cannot see this."""
    H = _broadcast_homography()
    # Shrink the image side by 10x -> ~10x fewer px per foot.
    S = np.diag([10.0, 10.0, 1.0])
    assert not plausible_court_homography(S @ H, IMG)


def test_plausibility_is_independent_of_reprojection_error():
    """A homography can fit its own points perfectly and still be nonsense.

    Four points always fit exactly, so solve_homography reports 0 ft while the
    implied court is absurd — which is why the pipeline checks both.
    """
    pix = np.array([[100, 100], [110, 100], [110, 110], [100, 110]], dtype=np.float64)
    court = np.array([[0, 0], [94, 0], [94, 50], [0, 50]], dtype=np.float64)
    res = solve_homography(pix, court)
    assert res.median_reproj_ft < 1e-6          # perfect fit ...
    assert not plausible_court_homography(res.H, IMG)   # ... and geometrically absurd
