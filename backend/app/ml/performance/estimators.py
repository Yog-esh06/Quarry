"""Label-free performance estimation.

Ground-truth labels for credit decisions arrive weeks after the decision. In
that gap a model can be silently wrong. Quarry estimates current performance
two independent ways and reports both, because they fail differently:

* **CBPE** (confidence-based performance estimation): if probabilities are
  calibrated, the expected value of any metric can be computed from the
  predictions alone. Valid under covariate shift *if calibration holds*;
  blind to concept drift by construction.
* **Importance weighting (IW)**: re-weight the labelled reference holdout by
  the density ratio w(x) = p_cur(x)/p_ref(x). Valid under pure covariate shift
  with overlapping support; degrades as the effective sample size collapses.

Neither can see a change in P(y|x) without labels. The benchmark quantifies
that blind spot instead of hiding it.
"""

from __future__ import annotations

import numpy as np

from app.ml.drift.domain import effective_sample_size
from app.ml.evaluation import metrics as M


def _rank_auroc_draws(p: np.ndarray, draws: np.ndarray) -> np.ndarray:
    """AUROC for each row of a (n_draws, n) label matrix, scores fixed (rank formula)."""
    order = np.argsort(p, kind="mergesort")
    ranks = np.empty(len(p))
    # average ranks for ties
    sorted_p = p[order]
    _, first, counts = np.unique(sorted_p, return_index=True, return_counts=True)
    avg = first + (counts + 1) / 2.0
    ranks[order] = np.repeat(avg, counts)
    n1 = draws.sum(axis=1)
    n0 = draws.shape[1] - n1
    rank_sum = draws @ ranks
    with np.errstate(divide="ignore", invalid="ignore"):
        auc = (rank_sum - n1 * (n1 + 1) / 2.0) / (n1 * n0)
    auc[(n1 == 0) | (n0 == 0)] = np.nan
    return auc


def cbpe(p: np.ndarray, threshold: float, n_draws: int = 200, seed: int = 0) -> dict[str, float]:
    """Expected metrics if labels were drawn from the (calibrated) predictions."""
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    rng = np.random.default_rng(seed)
    draws = (rng.random((n_draws, len(p))) < p).astype(float)
    aucs = _rank_auroc_draws(p, draws)
    pred = p >= threshold
    exp_acc = float(np.mean(np.where(pred, p, 1 - p)))
    return {
        "auroc": float(np.nanmean(aucs)),
        "auroc_low": float(np.nanquantile(aucs, 0.05)),
        "auroc_high": float(np.nanquantile(aucs, 0.95)),
        "accuracy": exp_acc,
        "log_loss": float(np.mean(M.expected_row_log_loss(p))),
        "brier": float(np.mean(p * (1 - p))),
        "base_rate": float(np.mean(p)),
    }


def importance_weighted(
    y_ref: np.ndarray, p_ref: np.ndarray, w: np.ndarray, threshold: float
) -> dict[str, float]:
    """Reference-holdout metrics re-weighted toward the current distribution."""
    w = np.asarray(w, dtype=float)
    return {
        "auroc": M.auroc(y_ref, p_ref, w),
        "accuracy": M.accuracy(y_ref, p_ref, threshold, w),
        "log_loss": M.log_loss(y_ref, p_ref, w),
        "brier": M.brier(y_ref, p_ref, w),
        "base_rate": float(np.average(y_ref, weights=w)),
        "ess": effective_sample_size(w),
        "ess_fraction": effective_sample_size(w) / max(len(w), 1),
    }
