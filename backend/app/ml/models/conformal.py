"""Split conformal prediction sets, standard and covariate-shift weighted.

Standard split conformal guarantees P(y in C(x)) >= 1 - alpha when test data
are exchangeable with calibration data. Under covariate shift that guarantee
breaks; weighted conformal prediction (Tibshirani et al., 2019) restores it by
re-weighting calibration scores with the density ratio w(x) = p_test(x)/p_ref(x),
which Quarry already estimates with its domain classifier. Comparing the two
coverages on labelled batches separates covariate shift (weighted coverage
holds) from concept drift (both break).

Nonconformity score: LAC, s(x, y) = 1 - p_y(x).
"""

from __future__ import annotations

import numpy as np


class SplitConformalClassifier:
    def fit(self, p_cal: np.ndarray, y_cal: np.ndarray) -> "SplitConformalClassifier":
        p_cal = np.asarray(p_cal, dtype=float)
        y_cal = np.asarray(y_cal).astype(int)
        scores = 1 - np.where(y_cal == 1, p_cal, 1 - p_cal)
        self.order_ = np.argsort(scores, kind="stable")
        self.sorted_scores_ = scores[self.order_]
        return self

    @property
    def n(self) -> int:
        return len(self.sorted_scores_)

    def qhat(self, alpha: float) -> float:
        k = int(np.ceil((self.n + 1) * (1 - alpha)))
        return float("inf") if k > self.n else float(self.sorted_scores_[k - 1])

    @staticmethod
    def _sets(p: np.ndarray, q: np.ndarray | float) -> np.ndarray:
        p = np.asarray(p, dtype=float)
        # s(x, 0) = p and s(x, 1) = 1 - p
        return np.stack([p <= q, (1 - p) <= q], axis=1)

    def predict_sets(self, p: np.ndarray, alpha: float = 0.1) -> np.ndarray:
        """Boolean (n, 2) membership matrix for classes {0, 1}."""
        return self._sets(p, self.qhat(alpha))

    def weighted_thresholds(self, w_cal: np.ndarray, w_test: np.ndarray, alpha: float) -> np.ndarray:
        """Per-test-point quantile of the weighted calibration score distribution.

        ``w_cal`` is aligned with the calibration rows in their original order.
        The test point's own mass is placed at +inf, as in Tibshirani et al.
        """
        w_sorted = np.asarray(w_cal, dtype=float)[self.order_]
        cum = np.cumsum(w_sorted)
        total = cum[-1] + np.asarray(w_test, dtype=float)
        idx = np.searchsorted(cum, (1 - alpha) * total, side="left")
        q = np.full(len(total), np.inf)
        ok = idx < len(cum)
        q[ok] = self.sorted_scores_[idx[ok]]
        return q

    def predict_sets_weighted(self, p: np.ndarray, w_cal: np.ndarray, w_test: np.ndarray, alpha: float = 0.1) -> np.ndarray:
        return self._sets(p, self.weighted_thresholds(w_cal, w_test, alpha))


def coverage(sets: np.ndarray, y: np.ndarray) -> float:
    y = np.asarray(y).astype(int)
    return float(np.mean(sets[np.arange(len(y)), y])) if len(y) else float("nan")


def mean_set_size(sets: np.ndarray) -> float:
    return float(sets.sum(axis=1).mean()) if len(sets) else float("nan")
