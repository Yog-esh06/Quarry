"""Post-hoc probability calibration.

Calibration is not cosmetic here: label-free performance estimation (CBPE)
is only valid if predicted probabilities mean what they say, and per-row
"excess loss" (realized loss minus the loss a calibrated model expects) is the
signal used to localise failing slices.
"""

from __future__ import annotations

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

EPS = 1e-6


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


class IsotonicCalibrator:
    method = "isotonic"

    def fit(self, p: np.ndarray, y: np.ndarray, sample_weight: np.ndarray | None = None) -> "IsotonicCalibrator":
        self.iso_ = IsotonicRegression(out_of_bounds="clip", y_min=1e-4, y_max=1 - 1e-4)
        self.iso_.fit(p, y, sample_weight=sample_weight)
        return self

    def transform(self, p: np.ndarray) -> np.ndarray:
        return np.asarray(self.iso_.predict(p), dtype=float)


class PlattCalibrator:
    """Logistic regression on the logit; stable with only a few hundred labels."""

    method = "platt"

    def fit(self, p: np.ndarray, y: np.ndarray, sample_weight: np.ndarray | None = None) -> "PlattCalibrator":
        self.lr_ = LogisticRegression(C=1e4, max_iter=1000)
        self.lr_.fit(_logit(p)[:, None], y, sample_weight=sample_weight)
        return self

    def transform(self, p: np.ndarray) -> np.ndarray:
        return self.lr_.predict_proba(_logit(np.asarray(p, dtype=float))[:, None])[:, 1]


def make_calibrator(n_rows: int) -> IsotonicCalibrator | PlattCalibrator:
    """Isotonic needs data; below a few thousand rows Platt overfits less."""
    return IsotonicCalibrator() if n_rows >= 3000 else PlattCalibrator()
