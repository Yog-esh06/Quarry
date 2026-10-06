"""Failure-slice discovery: *where* is the model failing?

Searches conjunctions of up to three predicates ("channel = partner AND
age < 27") for subgroups whose mean per-row error is significantly above the
rest of the data. Scoring follows SliceLine (Sagadeeva & Boehm, SIGMOD 2021):

    score(S) = alpha * (mean_err(S) / mean_err - 1) - (1 - alpha) * (n / |S| - 1)

which trades effect size against support, with SliceLine's upper bound used
to prune the lattice. Every reported slice must also pass a Welch t-test
against its complement with Benjamini-Hochberg FDR control over *all* slices
evaluated, so the search cannot "discover" noise by trying many slices.

The error signal is chosen by the caller:
* with labels: **excess loss** = realized log loss - binary entropy of the
  calibrated prediction. Its expectation is 0 at every x for a calibrated
  model, so positive slices are exactly where the model's beliefs are now
  wrong (rather than merely where the task is intrinsically hard).
* without labels: the per-row mechanism-shift contribution of the root cause,
  which localises where the distribution changed.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import combinations

import numpy as np
import pandas as pd
from scipy import stats

from app.ml.data.schema import DatasetSchema
from app.ml.evaluation.metrics import benjamini_hochberg


@dataclass(frozen=True)
class Predicate:
    feature: str
    op: str  # "==" | "in_range" | "is_missing"
    value: str | None = None
    low: float | None = None
    high: float | None = None

    @property
    def text(self) -> str:
        if self.op == "==":
            return f"{self.feature} = {self.value}"
        if self.op == "is_missing":
            return f"{self.feature} is missing"
        if self.low is None:
            return f"{self.feature} < {_fmt(self.high)}"
        if self.high is None:
            return f"{self.feature} >= {_fmt(self.low)}"
        return f"{_fmt(self.low)} <= {self.feature} < {_fmt(self.high)}"


def _fmt(x: float | None) -> str:
    if x is None:
        return ""
    ax = abs(x)
    if ax >= 1000:
        return f"{x:,.0f}"
    if ax >= 10:
        return f"{x:.1f}".rstrip("0").rstrip(".")
    return f"{x:.3g}"


@dataclass
class Slice:
    predicates: list[Predicate]
    size: int
    support: float
    mean_error: float
    complement_error: float
    effect: float
    lift: float
    t_stat: float
    p_value: float
    q_value: float
    score: float
    mask: np.ndarray | None = None

    @property
    def text(self) -> str:
        return " AND ".join(p.text for p in self.predicates)

    def to_dict(self) -> dict:
        d = asdict(self)
        d.pop("mask", None)
        d["text"] = self.text
        d["predicates"] = [dict(asdict(p), text=p.text) for p in self.predicates]
        return d


def build_predicates(df: pd.DataFrame, schema: DatasetSchema, n_bins: int = 4, max_cats: int = 12) -> list[tuple[Predicate, np.ndarray]]:
    preds: list[tuple[Predicate, np.ndarray]] = []
    for f in schema.features:
        col = df[f.name]
        missing = col.isna().to_numpy()
        if missing.any():
            preds.append((Predicate(f.name, "is_missing"), missing))
        if f.is_categorical:
            values = col.astype(object)
            for v in values.dropna().value_counts().index[:max_cats]:
                preds.append((Predicate(f.name, "==", value=str(v)), (values == v).to_numpy()))
            continue
        x = pd.to_numeric(col, errors="coerce").to_numpy(dtype=float)
        finite = x[np.isfinite(x)]
        if len(np.unique(finite)) < 2:
            continue
        edges = np.unique(np.quantile(finite, np.linspace(0, 1, n_bins + 1))[1:-1])
        bounds = [None, *edges.tolist(), None]
        for lo, hi in zip(bounds[:-1], bounds[1:]):
            m = np.isfinite(x)
            if lo is not None:
                m &= x >= lo
            if hi is not None:
                m &= x < hi
            preds.append((Predicate(f.name, "in_range", low=lo, high=hi), m))
        # open-ended tails capture "x >= threshold" failures that span two bins
        for e in edges:
            preds.append((Predicate(f.name, "in_range", low=float(e)), np.isfinite(x) & (x >= e)))
    return preds


def _welch(err: np.ndarray, m: np.ndarray) -> tuple[float, float]:
    a, b = err[m], err[~m]
    if len(a) < 2 or len(b) < 2:
        return 0.0, 1.0
    t, p = stats.ttest_ind(a, b, equal_var=False)
    if not np.isfinite(t):
        return 0.0, 1.0
    return float(t), float(p / 2 if t > 0 else 1 - p / 2)  # one-sided: slice worse


def find_slices(
    df: pd.DataFrame,
    error: np.ndarray,
    schema: DatasetSchema,
    max_level: int = 3,
    k: int = 8,
    alpha: float = 0.95,
    min_support: float = 0.02,
    min_rows: int = 30,
    fdr: float = 0.05,
) -> list[Slice]:
    err = np.asarray(error, dtype=float)
    n = len(err)
    # Shift so the mean is positive: SliceLine's ratio score needs positive errors.
    shift = max(0.0, -err.min()) + 1e-6
    e = err + shift
    ebar = e.mean()
    sigma = max(min_rows, int(np.ceil(min_support * n)))
    preds = [(p, m) for p, m in build_predicates(df.reset_index(drop=True), schema) if m.sum() >= sigma]

    def score(m: np.ndarray) -> float:
        s = m.sum()
        return alpha * (e[m].mean() / ebar - 1) - (1 - alpha) * (n / s - 1)

    def upper_bound(m: np.ndarray) -> float:
        # Best score any sub-slice with >= sigma rows could reach (SliceLine bound).
        s = int(m.sum())
        se = e[m]
        top = np.sort(se)[::-1]
        best = -np.inf
        for size in {sigma, max(sigma, s // 2), s}:
            if size > s:
                continue
            best = max(best, alpha * (top[:size].mean() / ebar - 1) - (1 - alpha) * (n / size - 1))
        return best

    evaluated: list[tuple[tuple[int, ...], np.ndarray, float]] = []
    level = [((i,), m) for i, (_, m) in enumerate(preds)]
    for lvl in range(1, max_level + 1):
        scored = [(idx, m, score(m)) for idx, m in level]
        evaluated.extend(scored)
        if lvl == max_level:
            break
        kth = sorted((s for _, _, s in evaluated), reverse=True)[min(k, len(evaluated)) - 1] if evaluated else -np.inf
        promising = [(idx, m) for idx, m, _ in scored if upper_bound(m) > max(kth, 0.0)]
        nxt: dict[tuple[int, ...], np.ndarray] = {}
        for (ia, ma), (ib, mb) in combinations(promising, 2):
            merged = tuple(sorted(set(ia) | set(ib)))
            if len(merged) != lvl + 1 or merged in nxt:
                continue
            feats = [preds[i][0].feature for i in merged]
            if len(set(feats)) != len(feats):
                continue  # one predicate per feature
            m = ma & mb
            if m.sum() >= sigma:
                nxt[merged] = m
        level = list(nxt.items())
        if not level:
            break

    if not evaluated:
        return []
    tests = [_welch(err, m) for _, m, _ in evaluated]
    q = benjamini_hochberg(np.array([p for _, p in tests]))
    cands = []
    for (idx, m, sc), (t, p), qv in zip(evaluated, tests, q):
        if sc <= 0 or qv > fdr:
            continue
        me, ce = float(err[m].mean()), float(err[~m].mean()) if (~m).any() else 0.0
        cands.append(Slice([preds[i][0] for i in idx], int(m.sum()), float(m.mean()), me, ce, me - ce,
                           (me + shift) / (ce + shift), t, p, float(qv), float(sc), m))
    cands.sort(key=lambda s: -s.score)

    # Drop slices that are mostly contained in an already-selected, stronger slice.
    selected: list[Slice] = []
    for s in cands:
        redundant = False
        for t in selected:
            overlap = (s.mask & t.mask).sum() / s.size
            if overlap > 0.8 and s.effect <= 1.1 * t.effect:
                redundant = True
                break
        if not redundant:
            selected.append(s)
        if len(selected) >= k:
            break
    return selected


def tree_slices(df: pd.DataFrame, error: np.ndarray, schema: DatasetSchema, max_depth: int = 3, min_support: float = 0.02, k: int = 8) -> list[Slice]:
    """Baseline: leaves of a regression tree fit to the per-row error."""
    from sklearn.tree import DecisionTreeRegressor

    from app.ml.data.schema import encode_frame

    X, _ = encode_frame(df, schema, schema.feature_names)
    X = np.where(np.isnan(X), -1e9, X)
    err = np.asarray(error, dtype=float)
    n = len(err)
    tree = DecisionTreeRegressor(max_depth=max_depth, min_samples_leaf=max(30, int(min_support * n)), random_state=0).fit(X, err)
    leaves = tree.apply(X)
    out = []
    for leaf in np.unique(leaves):
        m = leaves == leaf
        t, p = _welch(err, m)
        me, ce = float(err[m].mean()), float(err[~m].mean()) if (~m).any() else 0.0
        out.append(Slice([Predicate("tree_leaf", "==", value=str(int(leaf)))], int(m.sum()), float(m.mean()), me, ce, me - ce,
                         1.0, t, p, p, me - ce, m))
    out.sort(key=lambda s: -s.effect)
    return out[:k]


def jaccard(a: np.ndarray, b: np.ndarray) -> float:
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(inter / union) if union else 0.0
