"""Multi-object tracking with tracklets (roadmap 4.3).

Production uses ByteTrack via `supervision` with Kalman filtering and an
appearance embedding for re-ID across the severe mutual occlusion basketball
has. This is a compact stand-in with the same contract: constant-velocity
prediction, greedy nearest-foot association, tracklet bookkeeping, and — the
part that matters most — a hard reset on every camera cut, so track IDs never
persist across a replay.

Association is on foot-point pixel distance (players share similar box sizes and
move far between broadcast-rate frames, so IoU is unreliable).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from src.perception.detection import Detection


@dataclass
class Track:
    track_id: int
    foot: np.ndarray            # last foot pixel (2,)
    velocity: np.ndarray        # pixels/frame (2,)
    last_frame: int
    hits: int = 1
    missed: int = 0
    history: list = field(default_factory=list)  # (frame_idx, Detection)

    def predict(self) -> np.ndarray:
        return self.foot + self.velocity


class Tracker:
    """Greedy foot-distance tracker with camera-cut reset."""

    def __init__(self, max_dist: float = 90.0, max_missed: int = 2):
        self.max_dist = max_dist
        self.max_missed = max_missed
        self.tracks: list[Track] = []
        self.finished: list[Track] = []
        self._next_id = 0

    def _new_track(self, frame_idx, det):
        t = Track(self._next_id, np.array(det.foot), np.zeros(2), frame_idx)
        t.history.append((frame_idx, det))
        self.tracks.append(t)
        self._next_id += 1

    def reset(self):
        """Camera cut: retire all live tracks so IDs never cross the cut."""
        self.finished.extend(self.tracks)
        self.tracks = []

    def update(self, frame_idx: int, detections: list[Detection], cut: bool = False):
        if cut:
            self.reset()

        dets = list(detections)
        if not self.tracks:
            for d in dets:
                self._new_track(frame_idx, d)
            return self.tracks

        # Cost matrix: predicted track foot vs detection foot.
        preds = np.array([t.predict() for t in self.tracks])
        feet = np.array([d.foot for d in dets]) if dets else np.zeros((0, 2))
        assigned_t, assigned_d = set(), set()
        pairs = []
        if len(feet):
            cost = np.linalg.norm(preds[:, None, :] - feet[None, :, :], axis=2)
            order = np.dstack(np.unravel_index(np.argsort(cost, axis=None), cost.shape))[0]
            for ti, di in order:
                if ti in assigned_t or di in assigned_d:
                    continue
                if cost[ti, di] > self.max_dist:
                    continue
                pairs.append((ti, di))
                assigned_t.add(ti); assigned_d.add(di)

        # Update matched tracks.
        for ti, di in pairs:
            t, d = self.tracks[ti], dets[di]
            new_foot = np.array(d.foot)
            t.velocity = 0.5 * t.velocity + 0.5 * (new_foot - t.foot)
            t.foot = new_foot
            t.last_frame = frame_idx
            t.hits += 1
            t.missed = 0
            t.history.append((frame_idx, d))

        # Age unmatched tracks; retire the stale ones.
        for ti, t in enumerate(self.tracks):
            if ti not in assigned_t:
                t.missed += 1
        still, retired = [], []
        for t in self.tracks:
            (retired if t.missed > self.max_missed else still).append(t)
        self.finished.extend(retired)
        self.tracks = still

        # Spawn tracks for unmatched detections.
        for di, d in enumerate(dets):
            if di not in assigned_d:
                self._new_track(frame_idx, d)
        return self.tracks

    def all_tracklets(self, min_len: int = 1) -> list[Track]:
        """Every tracklet seen (live + retired) with at least min_len detections."""
        allt = self.finished + self.tracks
        return [t for t in allt if len(t.history) >= min_len]


def run_tracker(frames, cls: str = "player", max_dist: float = 90.0) -> Tracker:
    """Run the tracker over a list of BroadcastFrame, resetting on cuts."""
    tracker = Tracker(max_dist=max_dist)
    for f in frames:
        tracker.update(f.frame_idx, f.detections.by_class(cls), cut=f.cut)
    return tracker
