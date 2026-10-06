"""Domain classifier: multivariate two-sample test and density-ratio estimator.

One cross-fitted classifier that learns to tell reference rows from current
rows gives Quarry three things at once:

1. A **C2ST** (classifier two-sample test, Lopez-Paz & Oquab 2017): if the
   out-of-fold AUROC is above 0.5 the joint distribution has changed, including
   changes in feature *interactions* that no per-column test can see.
2. **Density ratios** w(x) = p_cur(x) / p_ref(x) = odds(x) * n_ref / n_cur,
   used for importance-weighted performance estimation, weighted conformal
   prediction and importance-weighted retraining.
3. **Log density ratios over column subsets**, the primitive behind causal
   mechanism-shift attribution (see app.ml.causal.attribution).

Histogram GBMs are used because they handle mixed types and missing values
natively (missingness itself is a distribution change worth detecting).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold

from app.ml.data.schema import DatasetSchema, encode_frame
from app.ml.evaluation.metrics import auroc

P_CLIP = 1e-4


def _gbm(cat_mask: np.ndarray, max_iter: int, seed: int) -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        max_iter=max_iter,
        learning_rate=0.1,
        max_leaf_nodes=15,
        min_samples_leaf=30,
        l2_regularization=1.0,
        early_stopping=False,
        categorical_features=cat_mask if cat_mask.any() else None,
        random_state=seed,
    )


def auc_pvalue(auc: float, n0: int, n1: int) -> float:
    """One-sided Mann-Whitney normal approximation for H0: AUROC = 0.5."""
    if n0 == 0 or n1 == 0 or not np.isfinite(auc):
        return 1.0
    se = np.sqrt((n0 + n1 + 1) / (12.0 * n0 * n1))
    return float(stats.norm.sf((auc - 0.5) / se))


@dataclass
class C2STResult:
    auc: float
    p_value: float
    n_ref: int
    n_cur: int
    log_ratio_cur: np.ndarray  # cross-fitted log density ratio at current rows
    log_ratio_ref: np.ndarray  # ... and at reference rows


def cross_fitted_log_ratio(
    X_ref: np.ndarray,
    X_cur: np.ndarray,
    cat_mask: np.ndarray,
    folds: int = 2,
    max_iter: int = 60,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Out-of-fold log p_cur(x)/p_ref(x) for every row, plus the out-of-fold AUROC."""
    n_ref, n_cur = len(X_ref), len(X_cur)
    if X_ref.shape[1] == 0:
        return np.zeros(n_ref), np.zeros(n_cur), 0.5
    X = np.vstack([X_ref, X_cur])
    z = np.r_[np.zeros(n_ref), np.ones(n_cur)].astype(int)
    oof = np.empty(len(z))
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    for k, (tr, te) in enumerate(skf.split(X, z)):
        clf = _gbm(cat_mask, max_iter, seed + k)
        clf.fit(X[tr], z[tr])
        oof[te] = clf.predict_proba(X[te])[:, 1]
    p = np.clip(oof, P_CLIP, 1 - P_CLIP)
    log_ratio = np.log(p / (1 - p)) - np.log(n_cur / n_ref)
    return log_ratio[:n_ref], log_ratio[n_ref:], auroc(z, oof)


class DomainClassifier:
    def __init__(self, schema: DatasetSchema, columns: list[str] | None = None, folds: int = 2, max_iter: int = 60, seed: int = 0):
        self.schema = schema
        self.columns = columns or schema.feature_names
        self.folds = folds
        self.max_iter = max_iter
        self.seed = seed

    def encode(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        return encode_frame(df, self.schema, self.columns)

    def test(self, ref: pd.DataFrame, cur: pd.DataFrame) -> C2STResult:
        Xr, mask = self.encode(ref)
        Xc, _ = self.encode(cur)
        lr_ref, lr_cur, auc = cross_fitted_log_ratio(Xr, Xc, mask, self.folds, self.max_iter, self.seed)
        return C2STResult(auc, auc_pvalue(auc, len(Xr), len(Xc)), len(Xr), len(Xc), lr_cur, lr_ref)

    def fit(self, ref: pd.DataFrame, cur: pd.DataFrame) -> "DomainClassifier":
        Xr, mask = self.encode(ref)
        Xc, _ = self.encode(cur)
        self.n_ref_, self.n_cur_ = len(Xr), len(Xc)
        self.clf_ = _gbm(mask, self.max_iter, self.seed).fit(
            np.vstack([Xr, Xc]), np.r_[np.zeros(len(Xr)), np.ones(len(Xc))].astype(int)
        )
        return self

    def density_ratio(self, df: pd.DataFrame, clip: float = 50.0) -> np.ndarray:
        X, _ = self.encode(df)
        p = np.clip(self.clf_.predict_proba(X)[:, 1], P_CLIP, 1 - P_CLIP)
        w = p / (1 - p) * (self.n_ref_ / self.n_cur_)
        return np.clip(w, 1.0 / clip, clip)


def ratio_from_log(log_ratio: np.ndarray, clip: float = 50.0) -> np.ndarray:
    return np.clip(np.exp(log_ratio), 1.0 / clip, clip)


def effective_sample_size(w: np.ndarray) -> float:
    w = np.asarray(w, dtype=float)
    return float(w.sum() ** 2 / max((w**2).sum(), 1e-12))
