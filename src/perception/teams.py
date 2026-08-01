"""Team assignment by appearance clustering (roadmap 4.4).

Embed each player crop (SigLIP torso crop in production; synthetic embedding
here), cluster with KMeans k=2, and assign teams at the *tracklet* level by
majority vote across the tracklet's frames — team identity is stable over a
tracklet, so a per-frame label would only add noise. Referees are already
excluded by the detector's class; we additionally drop very short tracklets
(coaches, photographers, association noise).
"""
from __future__ import annotations

from collections import Counter

import numpy as np
from sklearn.cluster import KMeans

MIN_TRACKLET_LEN = 3


def _tracklet_embeddings(track):
    return [d.embedding for _, d in track.history if d.embedding is not None]


def assign_teams(tracklets, min_len: int = MIN_TRACKLET_LEN, seed: int = 0) -> dict:
    """Return {track_id: team_label in {0,1}} for qualifying player tracklets.

    Fits KMeans(k=2) over all per-frame embeddings, then majority-votes each
    tracklet. Returns an empty dict if there is not enough signal to cluster.
    """
    usable = [t for t in tracklets if len(_tracklet_embeddings(t)) >= 1 and len(t.history) >= min_len]
    all_emb, owner = [], []
    for t in usable:
        for e in _tracklet_embeddings(t):
            all_emb.append(e)
            owner.append(t.track_id)
    if len(set(owner)) < 2 or len(all_emb) < 2:
        return {t.track_id: 0 for t in usable}

    X = np.array(all_emb)
    labels = KMeans(n_clusters=2, n_init=10, random_state=seed).fit_predict(X)

    votes: dict[int, Counter] = {}
    for tid, lab in zip(owner, labels):
        votes.setdefault(tid, Counter())[int(lab)] += 1
    return {tid: c.most_common(1)[0][0] for tid, c in votes.items()}


def team_accuracy(tracklets, team_labels) -> float:
    """Fraction of tracklets whose cluster matches their true team (eval only).

    Aligns the two clusters to the two ground-truth teams by best match.
    """
    rows = []
    for t in tracklets:
        if t.track_id not in team_labels:
            continue
        true = [d.true_team for _, d in t.history if d.true_team is not None]
        if not true:
            continue
        rows.append((team_labels[t.track_id], Counter(true).most_common(1)[0][0]))
    if not rows:
        return float("nan")
    teams = sorted({r[1] for r in rows})
    if len(teams) < 2:
        return float("nan")
    # Try both cluster->team alignments; take the better.
    best = 0.0
    for flip in (False, True):
        mapping = {0: teams[0], 1: teams[1]} if not flip else {0: teams[1], 1: teams[0]}
        acc = np.mean([mapping[c] == t for c, t in rows])
        best = max(best, acc)
    return float(best)
