"""LightGBM sub-models: shot make, pass completion, drive success (roadmap 3.2).

Each is an independently testable binary classifier whose output is a *calibrated
probability* — checked with a reliability diagram, not just AUC (3.2a). The shot
model blends a per-player empirical-Bayes prior with the league average, because
there are never enough shots per player per zone to fit individual effects
directly — so we shrink hard (3.2a).
"""
from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path

import lightgbm as lgb
import numpy as np

from src.value import features as F

# LightGBM params tuned for small tabular data; conservative to avoid overfit.
_LGB_PARAMS = dict(
    objective="binary",
    n_estimators=200,
    learning_rate=0.05,
    num_leaves=15,
    min_child_samples=30,
    subsample=0.8,
    colsample_bytree=0.9,
    reg_lambda=1.0,
    verbose=-1,
)


class _BinaryModel:
    """Thin LightGBM wrapper with a frozen feature-name contract."""

    feature_names: list[str] = []

    def __init__(self, **params):
        self.model = lgb.LGBMClassifier(**{**_LGB_PARAMS, **params})
        self.fitted = False

    def fit(self, X, y):
        self.model.fit(np.asarray(X, dtype=float), np.asarray(y, dtype=int))
        self.fitted = True
        return self

    def predict_proba(self, X) -> np.ndarray:
        X = np.asarray(X, dtype=float).reshape(-1, len(self.feature_names))
        return self.model.predict_proba(X)[:, 1]

    def save(self, path):
        with open(path, "wb") as fh:
            pickle.dump(self, fh)

    @staticmethod
    def load(path):
        with open(path, "rb") as fh:
            return pickle.load(fh)


class PassModel(_BinaryModel):
    feature_names = F.PASS_FEATURES


class DriveModel(_BinaryModel):
    feature_names = F.DRIVE_FEATURES


@dataclass
class _EBPrior:
    """Empirical-Bayes shrinkage of per-player make rate toward league mean."""

    league: float
    strength: float               # pseudo-count K; larger = shrink harder
    table: dict                   # player_id -> shrunk rate

    def get(self, player_id: int) -> float:
        return self.table.get(int(player_id), self.league)


class ShotModel(_BinaryModel):
    """Shot make probability with an empirical-Bayes per-player prior feature."""

    feature_names = F.SHOT_FEATURES
    PRIOR_COL = F.SHOT_FEATURES.index("player_shot_prior")

    def __init__(self, eb_strength: float = 200.0, **params):
        super().__init__(**params)
        self.eb_strength = eb_strength
        self.prior: _EBPrior | None = None

    def _fit_prior(self, y, player_ids) -> _EBPrior:
        league = float(np.mean(y)) if len(y) else 0.45
        table = {}
        for pid in np.unique(player_ids):
            m = player_ids == pid
            n, made = int(m.sum()), int(y[m].sum())
            table[int(pid)] = (made + self.eb_strength * league) / (n + self.eb_strength)
        return _EBPrior(league=league, strength=self.eb_strength, table=table)

    def _inject_prior(self, X, player_ids) -> np.ndarray:
        X = np.asarray(X, dtype=float).copy()
        priors = np.array([self.prior.get(p) for p in player_ids])
        X[:, self.PRIOR_COL] = priors
        return X

    def fit(self, X, y, player_ids):
        y = np.asarray(y, dtype=int)
        player_ids = np.asarray(player_ids)
        self.prior = self._fit_prior(y, player_ids)
        X = self._inject_prior(X, player_ids)
        return super().fit(X, y)

    def predict_proba(self, X, player_ids) -> np.ndarray:
        X = np.asarray(X, dtype=float).reshape(-1, len(self.feature_names))
        player_ids = np.atleast_1d(player_ids)
        X = self._inject_prior(X, player_ids)
        return self.model.predict_proba(X)[:, 1]


@dataclass
class SubModels:
    shot: ShotModel
    pass_: PassModel
    drive: DriveModel

    def save(self, out_dir):
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        self.shot.save(out / "shot_model.pkl")
        self.pass_.save(out / "pass_model.pkl")
        self.drive.save(out / "drive_model.pkl")

    @staticmethod
    def load(out_dir):
        out = Path(out_dir)
        return SubModels(
            shot=_BinaryModel.load(out / "shot_model.pkl"),
            pass_=_BinaryModel.load(out / "pass_model.pkl"),
            drive=_BinaryModel.load(out / "drive_model.pkl"),
        )


def train_submodels(ds) -> SubModels:
    """Fit all three sub-models from a :class:`LabeledDataset`."""
    shot = ShotModel().fit(ds.shot_X, ds.shot_y, ds.shot_player)
    pass_ = PassModel().fit(ds.pass_X, ds.pass_y)
    drive = DriveModel().fit(ds.drive_X, ds.drive_y)
    return SubModels(shot=shot, pass_=pass_, drive=drive)
