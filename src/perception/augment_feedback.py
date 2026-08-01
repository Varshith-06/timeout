"""Turn measured perception error into a Phase 2 augmentation config (5.2 -> 3.3).

This is the loop the roadmap insists on closing: measure the CV pipeline's actual
position/miss/identity error (roadmap 5.2), then retrain V(s) with augmentation
*matched to it* (roadmap 3.3) instead of guessed. The output maps directly onto
:class:`src.value.augment.AugmentConfig`.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class AugmentRecommendation:
    jitter_ft: float
    dropout_prob: float
    dropout_max: int
    id_swap_prob: float
    vel_lag_ft: float

    @classmethod
    def from_gap(cls, position_err_ft: float, miss_rate: float, id_error: float):
        """Derive augmentation strength from measured perception error.

        Position jitter is set slightly above the measured median error so the
        value model trains on at least the noise it will be served; dropout and
        id-swap probabilities mirror the measured miss and identity-error rates.
        """
        jitter = round(min(max(position_err_ft * 1.2, 0.5), 4.0), 2)
        return cls(
            jitter_ft=jitter,
            dropout_prob=round(min(max(miss_rate * 2.0, 0.1), 0.9), 2),
            dropout_max=2 if miss_rate < 0.2 else 3,
            id_swap_prob=round(min(max(id_error, 0.02), 0.5), 3),
            vel_lag_ft=round(jitter * 0.5, 2),
        )

    def as_dict(self) -> dict:
        return {
            "jitter_ft": self.jitter_ft,
            "dropout_prob": self.dropout_prob,
            "dropout_max": self.dropout_max,
            "id_swap_prob": self.id_swap_prob,
            "vel_lag_ft": self.vel_lag_ft,
        }

    def to_augment_config(self):
        """Build a live :class:`src.value.augment.AugmentConfig`."""
        from src.value.augment import AugmentConfig
        return AugmentConfig(
            jitter_ft=self.jitter_ft,
            dropout_max=self.dropout_max,
            dropout_prob=self.dropout_prob,
            id_swap_prob=self.id_swap_prob,
            vel_lag_ft=self.vel_lag_ft,
        )
