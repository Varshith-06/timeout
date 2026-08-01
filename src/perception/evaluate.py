"""Measure the perception domain gap on paired data (roadmap 5.2).

The whole reason to project our own court tracking through a camera is that it
gives paired data for free: recovered court positions vs the true ones. Aligning
them yields the exact numbers Phase 2 needs to close the domain gap — position
error (feet), identity-error rate, and player-miss rate — which feed back into
the value model's noise augmentation (roadmap 3.3 / 5.2).

On real footage the same measurement is possible by running the CV pipeline over
a 2015-16 broadcast for which SportVU exists and aligning on the game clock.
"""
from __future__ import annotations

from collections import Counter

import numpy as np

from src.perception.augment_feedback import AugmentRecommendation


def _track_true_pid(track) -> int | None:
    ids = [d.true_player_id for _, d in track.history if d.true_player_id is not None]
    return Counter(ids).most_common(1)[0][0] if ids else None


def measure_domain_gap(recovery, clip) -> dict:
    """Compare recovered court positions to ground truth across the clip.

    Returns position error stats (feet), player-miss rate, and identity-error
    rate — plus a recommended augmentation sigma for retraining Phase 2's V(s).
    """
    truth_pid = {t.track_id: _track_true_pid(t) for t in recovery.tracklets}

    pos_errors = []
    n_truth_visible = 0
    n_recovered = 0
    id_correct = id_total = 0

    for i, court in enumerate(recovery.frames_court):
        truth = clip.truth[i]["players"]
        n_truth_visible += len(truth)
        for tid, (x, y) in court.items():
            pid = truth_pid.get(tid)
            if pid is not None and pid in truth:
                t = truth[pid]
                pos_errors.append(np.hypot(x - t["x"], y - t["y"]))
                n_recovered += 1
        # Identity correctness (of tracks with a claimed player_id).
        for tid in court:
            ident = recovery.identities.get(tid)
            if ident and ident.player_id is not None:
                id_total += 1
                id_correct += int(ident.player_id == truth_pid.get(tid))

    pos_errors = np.array(pos_errors) if pos_errors else np.array([np.nan])
    miss_rate = 1.0 - (n_recovered / n_truth_visible) if n_truth_visible else 1.0
    id_error = 1.0 - (id_correct / id_total) if id_total else 1.0

    median_err = float(np.nanmedian(pos_errors))
    return {
        "position_error_ft_median": median_err,
        "position_error_ft_mean": float(np.nanmean(pos_errors)),
        "position_error_ft_p90": float(np.nanpercentile(pos_errors, 90)),
        "player_miss_rate": float(miss_rate),
        "identity_error_rate": float(id_error),
        "homography_valid_rate": recovery.diagnostics["homog_valid_rate"],
        "ball_recall": recovery.diagnostics["ball_recall"],
        "augment_recommendation": AugmentRecommendation.from_gap(
            median_err, miss_rate, id_error
        ).as_dict(),
    }
