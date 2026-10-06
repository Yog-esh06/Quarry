"""Probabilistic classification metrics with uncertainty.

Accuracy alone hides almost everything that matters for a risk model, so every
report carries a proper scoring rule (log loss, Brier), a ranking metric
(AUROC / AUPRC), calibration error, and bootstrap confidence intervals.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

EPS = 1e-6


def _clip(p: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(p, dtype=float), EPS, 1 - EPS)


def auroc(y: np.ndarray, p: np.ndarray, w: np.ndarray | None = None) -> float:
    y = np.asarray(y)
    if y.min(initial=0) == y.max(initial=0) or len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, p, sample_weight=w))


def auprc(y: np.ndarray, p: np.ndarray, w: np.ndarray | None = None) -> float:
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(average_precision_score(y, p, sample_weight=w))


def log_loss(y: np.ndarray, p: np.ndarray, w: np.ndarray | None = None) -> float:
    p = _clip(p)
    losses = -(y * np.log(p) + (1 - y) * np.log(1 - p))
    return float(np.average(losses, weights=w))


def row_log_loss(y: np.ndarray, p: np.ndarray) -> np.ndarray:
    p = _clip(p)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def expected_row_log_loss(p: np.ndarray) -> np.ndarray:
    """Binary entropy: the log loss a *calibrated* model expects at each row."""
    p = _clip(p)
    return -(p * np.log(p) + (1 - p) * np.log(1 - p))


def brier(y: np.ndarray, p: np.ndarray, w: np.ndarray | None = None) -> float:
    return float(np.average((np.asarray(p, dtype=float) - y) ** 2, weights=w))


def accuracy(y: np.ndarray, p: np.ndarray, threshold: float = 0.5, w: np.ndarray | None = None) -> float:
    return float(np.average((np.asarray(p) >= threshold).astype(int) == y, weights=w))


def ece(y: np.ndarray, p: np.ndarray, n_bins: int = 15) -> float:
    """Expected calibration error with equal-mass bins (robust to skewed scores)."""
    p = np.asarray(p, dtype=float)
    if len(p) == 0:
        return float("nan")
    order = np.argsort(p, kind="stable")
    bins = np.array_split(order, min(n_bins, len(p)))
    total = 0.0
    for b in bins:
        if len(b):
            total += len(b) * abs(p[b].mean() - np.asarray(y)[b].mean())
    return float(total / len(p))


def reliability_curve(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> list[dict]:
    p = np.asarray(p, dtype=float)
    order = np.argsort(p, kind="stable")
    out = []
    for b in np.array_split(order, min(n_bins, max(len(p), 1))):
        if len(b):
            out.append({"predicted": float(p[b].mean()), "observed": float(np.asarray(y)[b].mean()), "count": int(len(b))})
    return out


def expected_cost(y: np.ndarray, p: np.ndarray, threshold: float, cost_fp: float = 1.0, cost_fn: float = 5.0) -> float:
    """Average misclassification cost per decision (a missed default costs ``cost_fn``)."""
    pred = np.asarray(p) >= threshold
    fp = np.sum(pred & (y == 0))
    fn = np.sum(~pred & (y == 1))
    return float((cost_fp * fp + cost_fn * fn) / max(len(y), 1))


def best_threshold(y: np.ndarray, p: np.ndarray, cost_fp: float = 1.0, cost_fn: float = 5.0) -> float:
    grid = np.linspace(0.02, 0.9, 89)
    costs = [expected_cost(y, p, t, cost_fp, cost_fn) for t in grid]
    return float(grid[int(np.argmin(costs))])


def classification_report(y: np.ndarray, p: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    y = np.asarray(y).astype(int)
    return {
        "auroc": auroc(y, p),
        "auprc": auprc(y, p),
        "log_loss": log_loss(y, p),
        "brier": brier(y, p),
        "ece": ece(y, p),
        "accuracy": accuracy(y, p, threshold),
        "expected_cost": expected_cost(y, p, threshold),
        "base_rate": float(np.mean(y)) if len(y) else float("nan"),
        "mean_prediction": float(np.mean(p)) if len(p) else float("nan"),
        "n": int(len(y)),
    }


def bootstrap_ci(
    fn: Callable[[np.ndarray, np.ndarray], float],
    y: np.ndarray,
    p: np.ndarray,
    n_boot: int = 200,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    n = len(y)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        v = fn(y[idx], p[idx])
        if np.isfinite(v):
            vals.append(v)
    if not vals:
        return float("nan"), float("nan")
    return float(np.quantile(vals, alpha / 2)), float(np.quantile(vals, 1 - alpha / 2))


def report_with_ci(y: np.ndarray, p: np.ndarray, threshold: float = 0.5, n_boot: int = 200, seed: int = 0) -> dict:
    rep = classification_report(y, p, threshold)
    for name, fn in (("auroc", auroc), ("log_loss", log_loss), ("brier", brier), ("ece", ece)):
        lo, hi = bootstrap_ci(fn, np.asarray(y), np.asarray(p), n_boot=n_boot, seed=seed)
        rep[f"{name}_ci"] = [lo, hi]
    return rep


def paired_bootstrap_delta(
    fn: Callable[[np.ndarray, np.ndarray], float],
    y: np.ndarray,
    p_base: np.ndarray,
    p_new: np.ndarray,
    n_boot: int = 500,
    seed: int = 0,
) -> dict[str, float]:
    """Distribution of fn(new) - fn(base) under paired resampling of rows."""
    rng = np.random.default_rng(seed)
    n = len(y)
    deltas = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        deltas[b] = fn(y[idx], p_new[idx]) - fn(y[idx], p_base[idx])
    deltas = deltas[np.isfinite(deltas)]
    point = fn(y, p_new) - fn(y, p_base)
    return {
        "delta": float(point),
        "ci_low": float(np.quantile(deltas, 0.025)) if len(deltas) else float("nan"),
        "ci_high": float(np.quantile(deltas, 0.975)) if len(deltas) else float("nan"),
        "p_le_zero": float(np.mean(deltas <= 0)) if len(deltas) else float("nan"),
    }


def benjamini_hochberg(pvals: np.ndarray) -> np.ndarray:
    """BH-adjusted q-values (false discovery rate control across many tests)."""
    p = np.asarray(pvals, dtype=float)
    n = len(p)
    if n == 0:
        return p
    order = np.argsort(p)
    ranked = p[order] * n / np.arange(1, n + 1)
    q = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.clip(q, 0, 1)
    return out
