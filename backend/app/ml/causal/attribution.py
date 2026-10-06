"""Causal mechanism-shift attribution: *which* mechanism broke?

Under the causal Markov factorisation  p(x) = prod_v p(x_v | pa(x_v)),  the KL
divergence between the current and reference joint distributions decomposes
exactly over mechanisms (chain rule of KL; cf. Budhathoki et al., AISTATS 2021):

    KL(P_cur || P_ref) = sum_v  E_cur[ log p_cur(x_v | pa_v) - log p_ref(x_v | pa_v) ]
                       = sum_v  KL_v

A column whose *own* mechanism is unchanged contributes KL_v ~ 0 even if its
marginal distribution moved a lot because a parent moved. That is what
separates a root cause from a symptom, and it is what per-column drift tests
cannot do.

Estimation without density models: for any column set S, a probabilistic
classifier that discriminates current from reference rows on S gives
log p_cur(x_S)/p_ref(x_S) = logit(c(x_S)) - log(n_cur/n_ref). Then

    log p_cur(x_v | pa)/p_ref(x_v | pa) = log r_{pa+v}(x) - log r_{pa}(x)

Both ratios are cross-fitted on identical folds (no in-sample optimism) with
histogram GBMs, which handle categoricals and missingness natively, so a
missingness spike is a mechanism shift like any other.

Significance: the same estimator applied to two disjoint halves of the
reference data gives each node's empirical null (estimator bias + noise for
that node's dimensionality and the current window size). Nodes are scored by
excess KL over that null and a z-score.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd

from app.ml.causal.graph import CausalGraph
from app.ml.data.schema import DatasetSchema, encode_frame
from app.ml.drift.domain import cross_fitted_log_ratio

Z_THRESHOLD = 3.0
MIN_EXCESS_KL = 0.004  # nats; below this a shift is not practically meaningful


@dataclass
class NodeAttribution:
    node: str
    kind: str  # "root" | "conditional" | "target"
    parents: list[str]
    kl: float
    null_mean: float
    null_std: float
    excess_kl: float
    z: float
    share: float
    significant: bool
    marginal_shift: float = 0.0  # supplied by caller (univariate drift distance)
    marginal_drifted: bool = False
    role: str = "stable"  # "root_cause" | "symptom" | "stable"
    rank: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class AttributionResult:
    graph_kind: str
    nodes: list[NodeAttribution]
    total_kl: float
    joint_kl: float  # direct estimate on all columns, a consistency check
    n_ref: int
    n_cur: int
    row_contributions: dict[str, np.ndarray] = field(default_factory=dict, repr=False)

    def ranking(self) -> list[str]:
        return [n.node for n in sorted(self.nodes, key=lambda a: a.rank)]

    @property
    def root_causes(self) -> list[NodeAttribution]:
        return [n for n in sorted(self.nodes, key=lambda a: a.rank) if n.role == "root_cause"]

    def to_dict(self) -> dict:
        return {
            "graph_kind": self.graph_kind,
            "total_kl": self.total_kl,
            "joint_kl": self.joint_kl,
            "n_ref": self.n_ref,
            "n_cur": self.n_cur,
            "nodes": [n.to_dict() for n in sorted(self.nodes, key=lambda a: a.rank)],
        }


class MechanismShiftAttributor:
    def __init__(
        self,
        graph: CausalGraph,
        schema: DatasetSchema,
        folds: int = 2,
        max_iter: int = 60,
        null_reps: int = 3,
        max_ref: int = 6000,
        seed: int = 0,
    ):
        self.graph = graph
        self.schema = schema
        self.folds = folds
        self.max_iter = max_iter
        self.null_reps = null_reps
        self.max_ref = max_ref
        self.seed = seed

    # -- primitives ------------------------------------------------------------
    def _log_ratios(self, ref: pd.DataFrame, cur: pd.DataFrame, cols: list[str], cache: dict) -> np.ndarray:
        key = tuple(sorted(cols))
        if key not in cache:
            if not cols:
                cache[key] = np.zeros(len(cur))
            else:
                Xr, mask = encode_frame(ref, self.schema, list(key))
                Xc, _ = encode_frame(cur, self.schema, list(key))
                _, lr_cur, _ = cross_fitted_log_ratio(Xr, Xc, mask, self.folds, self.max_iter, self.seed)
                cache[key] = lr_cur
        return cache[key]

    def _node_kls(self, ref: pd.DataFrame, cur: pd.DataFrame, nodes: list[str]) -> tuple[dict, dict, float]:
        cache: dict = {}
        kls, rows = {}, {}
        for v in nodes:
            pa = [p for p in self.graph.parents(v) if p in nodes]
            contrib = self._log_ratios(ref, cur, pa + [v], cache) - self._log_ratios(ref, cur, pa, cache)
            rows[v] = contrib
            kls[v] = float(np.mean(contrib))
        joint = float(np.mean(self._log_ratios(ref, cur, list(nodes), cache)))
        return kls, rows, joint

    def _subsample(self, df: pd.DataFrame, n: int, rng: np.random.Generator) -> pd.DataFrame:
        if len(df) <= n:
            return df.reset_index(drop=True)
        return df.iloc[rng.choice(len(df), n, replace=False)].reset_index(drop=True)

    # -- public API ------------------------------------------------------------
    def null_distribution(self, ref: pd.DataFrame, n_cur: int, nodes: list[str]) -> dict[str, tuple[float, float]]:
        """Per-node KL estimates on reference-vs-reference splits (no shift by construction)."""
        rng = np.random.default_rng(self.seed + 101)
        n_ref = min(self.max_ref, max(len(ref) - n_cur, 50), 3 * n_cur)
        samples: dict[str, list[float]] = {v: [] for v in nodes}
        for _ in range(self.null_reps):
            perm = rng.permutation(len(ref))
            pseudo_cur = ref.iloc[perm[:n_cur]].reset_index(drop=True)
            pseudo_ref = ref.iloc[perm[n_cur : n_cur + n_ref]].reset_index(drop=True)
            kls, _, _ = self._node_kls(pseudo_ref, pseudo_cur, nodes)
            for v in nodes:
                samples[v].append(kls[v])
        return {v: (float(np.mean(s)), float(np.std(s, ddof=1)) if len(s) > 1 else 0.0) for v, s in samples.items()}

    def attribute(
        self,
        ref: pd.DataFrame,
        cur: pd.DataFrame,
        include_target: bool = False,
        null: dict[str, tuple[float, float]] | None = None,
        marginal: dict[str, tuple[float, bool]] | None = None,
    ) -> AttributionResult:
        target = self.schema.target
        nodes = [v for v in self.graph.topological_order() if v != target or include_target]
        nodes = [v for v in nodes if v in cur.columns and v in ref.columns]
        rng = np.random.default_rng(self.seed)
        ref_s = self._subsample(ref, min(self.max_ref, 3 * len(cur)), rng)
        cur_s = cur.reset_index(drop=True)
        if null is None:
            null = self.null_distribution(ref, len(cur_s), nodes)
        kls, rows, joint = self._node_kls(ref_s, cur_s, nodes)

        out: list[NodeAttribution] = []
        for v in nodes:
            mu, sd = null.get(v, (0.0, 0.0))
            sd_eff = max(sd, 0.25 * abs(mu), 1e-3)
            excess = kls[v] - mu
            z = excess / sd_eff
            ms, md = (marginal or {}).get(v, (0.0, False))
            out.append(
                NodeAttribution(
                    node=v,
                    kind="target" if v == target else ("root" if not self.graph.parents(v) else "conditional"),
                    parents=[p for p in self.graph.parents(v) if p in nodes],
                    kl=kls[v],
                    null_mean=mu,
                    null_std=sd,
                    excess_kl=excess,
                    z=z,
                    share=0.0,
                    significant=bool(z >= Z_THRESHOLD and excess >= MIN_EXCESS_KL),
                    marginal_shift=ms,
                    marginal_drifted=md,
                )
            )
        pos = sum(max(n.excess_kl, 0.0) for n in out if n.significant) or sum(max(n.excess_kl, 0.0) for n in out) or 1.0
        for n in out:
            n.share = max(n.excess_kl, 0.0) / pos if (n.significant or not any(m.significant for m in out)) else 0.0
            n.role = "root_cause" if n.significant else ("symptom" if n.marginal_drifted else "stable")
        for i, n in enumerate(sorted(out, key=lambda a: (not a.significant, -a.excess_kl))):
            n.rank = i + 1
        total = float(sum(kls.values()))
        return AttributionResult(self.graph.kind, out, total, joint, len(ref_s), len(cur_s), rows)


# -- baselines used by the benchmark -------------------------------------------------


def rank_by_marginal_drift(drifts: list) -> list[str]:
    """Industry baseline: blame the columns whose marginal distribution moved most."""
    return [d.feature for d in sorted(drifts, key=lambda d: (d.q_value, -d.distance))]


def rank_by_domain_importance(ref: pd.DataFrame, cur: pd.DataFrame, schema: DatasetSchema, columns: list[str], seed: int = 0) -> list[str]:
    """Baseline: permutation importance of a drift classifier (what drift tools usually show)."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.inspection import permutation_importance
    from sklearn.model_selection import train_test_split

    Xr, mask = encode_frame(ref, schema, columns)
    Xc, _ = encode_frame(cur, schema, columns)
    X = np.vstack([Xr, Xc])
    z = np.r_[np.zeros(len(Xr)), np.ones(len(Xc))].astype(int)
    Xtr, Xte, ztr, zte = train_test_split(X, z, test_size=0.4, stratify=z, random_state=seed)
    clf = HistGradientBoostingClassifier(max_iter=60, max_leaf_nodes=15, categorical_features=mask if mask.any() else None,
                                         early_stopping=False, random_state=seed).fit(Xtr, ztr)
    imp = permutation_importance(clf, Xte, zte, scoring="roc_auc", n_repeats=3, random_state=seed)
    order = np.argsort(-imp.importances_mean)
    return [columns[i] for i in order]
