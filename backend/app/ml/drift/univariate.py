"""Per-feature distribution tests with false-discovery-rate control.

These are the industry-standard drift checks (KS, chi-square, PSI). Quarry
keeps them because they are cheap and interpretable, but treats them as
*symptom* detectors: a column can drift purely because an upstream cause
moved. The causal attribution layer decides which drifts are causes.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial.distance import jensenshannon

from app.ml.data.schema import DatasetSchema
from app.ml.data.validation import null_rate_pvalue
from app.ml.evaluation.metrics import benjamini_hochberg

PSI_EPS = 1e-4


@dataclass
class FeatureDrift:
    feature: str
    kind: str
    test: str
    statistic: float
    p_value: float
    q_value: float
    psi: float
    distance: float  # JS distance (categorical) or IQR-normalised Wasserstein (numeric)
    null_rate_ref: float
    null_rate_cur: float
    null_p_value: float
    drifted: bool

    def to_dict(self) -> dict:
        return asdict(self)


def _psi(ref_counts: np.ndarray, cur_counts: np.ndarray) -> float:
    r = ref_counts / max(ref_counts.sum(), 1) + PSI_EPS
    c = cur_counts / max(cur_counts.sum(), 1) + PSI_EPS
    return float(np.sum((c - r) * np.log(c / r)))


def numeric_psi(ref: np.ndarray, cur: np.ndarray, n_bins: int = 10) -> float:
    if len(ref) == 0 or len(cur) == 0:
        return 0.0
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, n_bins + 1)))
    if len(edges) < 2:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    return _psi(np.histogram(ref, edges)[0].astype(float), np.histogram(cur, edges)[0].astype(float))


def _categorical_counts(ref: pd.Series, cur: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    cats = sorted(set(ref.dropna().astype(str)) | set(cur.dropna().astype(str)))
    rc = ref.dropna().astype(str).value_counts()
    cc = cur.dropna().astype(str).value_counts()
    return (
        np.array([rc.get(c, 0) for c in cats], dtype=float),
        np.array([cc.get(c, 0) for c in cats], dtype=float),
    )


def univariate_drift(
    ref: pd.DataFrame, cur: pd.DataFrame, schema: DatasetSchema, alpha: float = 0.01
) -> list[FeatureDrift]:
    rows: list[dict] = []
    for f in schema.features:
        r, c = ref[f.name], cur[f.name]
        nr_ref, nr_cur = float(r.isna().mean()), float(c.isna().mean())
        null_p = null_rate_pvalue(int(r.isna().sum()), len(r), int(c.isna().sum()), len(c))
        if f.is_categorical:
            rc, cc = _categorical_counts(r, c)
            if len(rc) < 2 or cc.sum() == 0 or rc.sum() == 0:
                stat, p = 0.0, 1.0
            else:
                stat, p = stats.chi2_contingency(np.stack([rc, cc]) + 0.5)[:2]
            rows.append(dict(feature=f.name, kind=f.kind, test="chi2", statistic=float(stat), p_value=float(p),
                             psi=_psi(rc, cc), distance=float(jensenshannon(rc / max(rc.sum(), 1), cc / max(cc.sum(), 1), base=2)) if rc.sum() and cc.sum() else 0.0,
                             null_rate_ref=nr_ref, null_rate_cur=nr_cur, null_p_value=null_p))
        else:
            rv = pd.to_numeric(r, errors="coerce").dropna().to_numpy(float)
            cv = pd.to_numeric(c, errors="coerce").dropna().to_numpy(float)
            if len(rv) < 5 or len(cv) < 5:
                stat, p, dist = 0.0, 1.0, 0.0
            else:
                ks = stats.ks_2samp(rv, cv)
                stat, p = float(ks.statistic), float(ks.pvalue)
                iqr = float(np.subtract(*np.quantile(rv, [0.75, 0.25]))) or float(np.std(rv)) or 1.0
                dist = float(stats.wasserstein_distance(rv, cv) / iqr)
            rows.append(dict(feature=f.name, kind=f.kind, test="ks", statistic=stat, p_value=p,
                             psi=numeric_psi(rv, cv), distance=dist,
                             null_rate_ref=nr_ref, null_rate_cur=nr_cur, null_p_value=null_p))
    # One family of tests per batch: distribution tests and missingness tests.
    pvals = np.array([row["p_value"] for row in rows] + [row["null_p_value"] for row in rows])
    q = benjamini_hochberg(pvals)
    k = len(rows)
    out = []
    for i, row in enumerate(rows):
        qv = float(min(q[i], q[k + i]))
        out.append(FeatureDrift(q_value=qv, drifted=qv < alpha, **row))
    return out
