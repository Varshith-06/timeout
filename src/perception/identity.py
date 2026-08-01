"""Player identity from jersey numbers (roadmap 4.5).

Team assignment gives five bodies per side; to look up personnel priors we need
names. Per-frame jersey OCR is unreliable, so we vote across the whole tracklet
(60 frames of the same player give a confident answer). The two appearance
clusters are mapped to the two rosters by jersey-set overlap, then
(team, jersey) -> player_id via the game roster.

Graceful fallback (roadmap 4.5 / 4.9): when identity is uncertain, return None so
the state builder uses position-average priors and lowers ``confidence`` rather
than committing to a wrong name.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass


@dataclass
class Identity:
    track_id: int
    team_label: int
    team_id: int | None          # cluster mapped to a roster team_id
    jersey: int | None
    player_id: int | None
    confidence: float


def roster_lookup(roster_rows):
    """Build lookups from (team_id, jersey, player_id) rows.

    Returns (by_team_jersey, team_jerseys): {(team_id,jersey): player_id} and
    {team_id: set(jerseys)}.
    """
    by_tj, team_jerseys = {}, {}
    for team_id, jersey, player_id in roster_rows:
        if jersey is None:
            continue
        by_tj[(team_id, int(jersey))] = player_id
        team_jerseys.setdefault(team_id, set()).add(int(jersey))
    return by_tj, team_jerseys


def _vote_jersey(track):
    reads = [d.jersey_read for _, d in track.history if d.jersey_read is not None]
    if not reads:
        return None, 0.0
    top, n = Counter(reads).most_common(1)[0]
    conf = n / len(track.history)      # agreement across the tracklet
    return int(top), float(conf)


def _map_clusters_to_teams(cluster_jerseys, team_jerseys):
    """Assign each of the two clusters to a team_id by best jersey-set overlap."""
    teams = list(team_jerseys.keys())
    clusters = list(cluster_jerseys.keys())
    best, best_map = -1, {}
    # Only two clusters / two teams — evaluate both alignments.
    from itertools import permutations
    for perm in permutations(teams, len(clusters)):
        mapping = dict(zip(clusters, perm))
        score = sum(len(cluster_jerseys[c] & team_jerseys[mapping[c]]) for c in clusters)
        if score > best:
            best, best_map = score, mapping
    return best_map


def assign_identities(tracklets, team_labels, roster_rows) -> dict:
    """Return {track_id: Identity}. Uncertain tracklets get player_id=None."""
    by_tj, team_jerseys = roster_lookup(roster_rows)

    # Per-tracklet jersey vote.
    voted = {}
    for t in tracklets:
        if t.track_id not in team_labels:
            continue
        jersey, conf = _vote_jersey(t)
        voted[t.track_id] = (team_labels[t.track_id], jersey, conf)

    # Gather each cluster's observed jerseys and map clusters -> team_ids.
    cluster_jerseys: dict[int, set] = {}
    for _, (lab, jersey, _) in voted.items():
        if jersey is not None:
            cluster_jerseys.setdefault(lab, set()).add(jersey)
    cluster_to_team = _map_clusters_to_teams(cluster_jerseys, team_jerseys) if cluster_jerseys else {}

    out = {}
    for tid, (lab, jersey, conf) in voted.items():
        team_id = cluster_to_team.get(lab)
        player_id = by_tj.get((team_id, jersey)) if (team_id is not None and jersey is not None) else None
        # Confidence reflects both jersey legibility and a successful roster hit.
        c = conf if player_id is not None else min(conf, 0.3)
        out[tid] = Identity(tid, lab, team_id, jersey, player_id, round(c, 3))
    return out


def identity_accuracy(identities, tracklets) -> float:
    """Fraction of identified tracklets whose player_id matches truth (eval only)."""
    truth = {}
    for t in tracklets:
        ids = [d.true_player_id for _, d in t.history if d.true_player_id is not None]
        if ids:
            truth[t.track_id] = Counter(ids).most_common(1)[0][0]
    hits = [identities[tid].player_id == truth[tid]
            for tid in identities if identities[tid].player_id is not None and tid in truth]
    return float(sum(hits) / len(hits)) if hits else float("nan")
