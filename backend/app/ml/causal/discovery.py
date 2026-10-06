"""Causal discovery with the PC-stable algorithm and tiered background knowledge.

Why PC: it is constraint-based, so every edge decision is an explicit
conditional-independence test that can be inspected, and it scales to a
dozen mixed-type variables in seconds on a CPU.

CI test: the variables are mixed (nominal categoricals, skewed continuous,
counts), so each variable is mapped to a *block* of columns - normal scores
(Gaussian copula / nonparanormal transform) for ordered variables and one-hot
columns for nominal ones. X _||_ Y | Z is tested by residualising both blocks
on Z and testing that every canonical correlation between the residual
blocks is zero (Bartlett's chi-square approximation to Wilks' lambda). For two
scalar variables this is exactly the Fisher-z partial-correlation test.

Background knowledge: variables are assigned to tiers (exogenous applicant
attributes < everything else < outcome); no edge may point from a later tier
into an earlier one. Edges PC cannot orient are oriented by tier, then by
name, and are reported as unoriented so they are never presented as findings.
"""

from __future__ import annotations

from itertools import combinations

import networkx as nx
import numpy as np
import pandas as pd
from scipy import stats

from app.ml.causal.graph import CausalGraph


def _normal_scores(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    x = np.where(np.isnan(x), np.nanmedian(x), x)
    r = stats.rankdata(x)
    return stats.norm.ppf(r / (len(x) + 1))


def encode_blocks(df: pd.DataFrame, kinds: dict[str, str]) -> dict[str, np.ndarray]:
    blocks: dict[str, np.ndarray] = {}
    for name, kind in kinds.items():
        col = df[name]
        if kind == "categorical":
            dummies = pd.get_dummies(col.astype(str), drop_first=True).to_numpy(dtype=float)
            if dummies.shape[1] == 0:
                dummies = np.zeros((len(df), 1))
            blocks[name] = dummies - dummies.mean(axis=0)
        else:
            z = _normal_scores(pd.to_numeric(col, errors="coerce").to_numpy(dtype=float))
            blocks[name] = (z - z.mean())[:, None]
    return blocks


def _residualize(A: np.ndarray, Z: np.ndarray | None) -> np.ndarray:
    if Z is None or Z.shape[1] == 0:
        return A - A.mean(axis=0)
    Zc = np.column_stack([np.ones(len(Z)), Z])
    coef, *_ = np.linalg.lstsq(Zc, A, rcond=None)
    return A - Zc @ coef


def block_ci_test(X: np.ndarray, Y: np.ndarray, Z: np.ndarray | None) -> float:
    """p-value for H0: X _||_ Y | Z under a Gaussian-copula model."""
    n = len(X)
    rx, ry = _residualize(X, Z), _residualize(Y, Z)

    def _whiten(R: np.ndarray) -> np.ndarray:
        cov = R.T @ R / n
        vals, vecs = np.linalg.eigh(cov)
        keep = vals > 1e-10 * max(vals.max(), 1e-12)
        return R @ (vecs[:, keep] / np.sqrt(vals[keep]))

    wx, wy = _whiten(rx), _whiten(ry)
    if wx.shape[1] == 0 or wy.shape[1] == 0:
        return 1.0
    rho = np.clip(np.linalg.svd(wx.T @ wy / n, compute_uv=False), 0, 1 - 1e-12)
    p, q = wx.shape[1], wy.shape[1]
    r = 0 if Z is None else Z.shape[1]
    stat = -(n - 1 - r - (p + q + 1) / 2.0) * np.sum(np.log(1 - rho**2))
    return float(stats.chi2.sf(max(stat, 0.0), p * q))


def pc_discover(
    df: pd.DataFrame,
    kinds: dict[str, str],
    tiers: dict[str, int],
    alpha: float = 0.01,
    max_cond: int = 3,
    max_rows: int = 8000,
    seed: int = 0,
) -> CausalGraph:
    nodes = list(kinds)
    if len(df) > max_rows:
        df = df.sample(max_rows, random_state=seed)
    blocks = encode_blocks(df.reset_index(drop=True), kinds)
    adj = {v: set(nodes) - {v} for v in nodes}
    sepset: dict[frozenset, set] = {}
    n_tests = 0

    # --- skeleton (PC-stable: adjacency frozen per level) -------------------
    for level in range(max_cond + 1):
        frozen = {v: set(a) for v, a in adj.items()}
        if all(len(a) - 1 < level for a in frozen.values()):
            break
        # Each unordered pair is visited from both endpoints: conditioning sets
        # are drawn from x's (frozen) neighbourhood, then from y's.
        for x in nodes:
            for y in sorted(frozen[x]):
                if y not in adj[x]:
                    continue
                cands = sorted(frozen[x] - {y})
                if len(cands) < level:
                    continue
                for S in combinations(cands, level):
                    Z = np.column_stack([blocks[s] for s in S]) if S else None
                    n_tests += 1
                    if block_ci_test(blocks[x], blocks[y], Z) > alpha:
                        adj[x].discard(y)
                        adj[y].discard(x)
                        sepset[frozenset((x, y))] = set(S)
                        break

    # --- orientation ----------------------------------------------------------
    directed: set[tuple[str, str]] = set()
    undirected: set[frozenset] = {frozenset((x, y)) for x in nodes for y in adj[x]}

    def orient(a: str, b: str) -> bool:
        """Orient a -> b if allowed by tiers and not already oriented the other way."""
        e = frozenset((a, b))
        if tiers.get(a, 1) > tiers.get(b, 1) or (b, a) in directed:
            return False
        if e in undirected:
            undirected.discard(e)
            directed.add((a, b))
            return True
        return False

    # v-structures x -> z <- y
    for z in nodes:
        for x, y in combinations(sorted(adj[z]), 2):
            if y in adj[x]:
                continue
            if z not in sepset.get(frozenset((x, y)), set()):
                if tiers.get(z, 1) >= max(tiers.get(x, 1), tiers.get(y, 1)):
                    orient(x, z)
                    orient(y, z)

    # tier knowledge
    for e in list(undirected):
        a, b = tuple(e)
        if tiers.get(a, 1) < tiers.get(b, 1):
            orient(a, b)
        elif tiers.get(b, 1) < tiers.get(a, 1):
            orient(b, a)

    # Meek rules R1-R3
    def is_adj(a: str, b: str) -> bool:
        return b in adj[a]

    changed = True
    while changed:
        changed = False
        for e in list(undirected):
            a, b = tuple(e)
            for u, v in ((a, b), (b, a)):
                if frozenset((u, v)) not in undirected:
                    break
                # R1: w -> u - v, w not adj v  =>  u -> v
                if any((w, u) in directed and not is_adj(w, v) for w in nodes if w not in (u, v)):
                    changed |= orient(u, v)
                    continue
                # R2: u -> w -> v and u - v  =>  u -> v
                if any((u, w) in directed and (w, v) in directed for w in nodes):
                    changed |= orient(u, v)
                    continue
                # R3: u - w1 -> v, u - w2 -> v, w1 not adj w2, u - v  =>  u -> v
                ws = [w for w in nodes if frozenset((u, w)) in undirected and (w, v) in directed]
                if any(not is_adj(w1, w2) for w1, w2 in combinations(ws, 2)):
                    changed |= orient(u, v)

    # Fallback orientation for the rest: tier, then name; never create a cycle.
    unresolved = []
    g = nx.DiGraph(list(directed))
    g.add_nodes_from(nodes)
    for e in sorted(undirected, key=lambda s: tuple(sorted(s))):
        a, b = sorted(e, key=lambda v: (tiers.get(v, 1), v))
        g.add_edge(a, b)
        if not nx.is_directed_acyclic_graph(g):
            g.remove_edge(a, b)
            a, b = b, a
            g.add_edge(a, b)
        unresolved.append((a, b))

    # Any cycle from conflicting v-structures: drop the edge closing the cycle.
    while not nx.is_directed_acyclic_graph(g):
        cyc = nx.find_cycle(g)
        g.remove_edge(*cyc[-1][:2])

    edges = sorted(g.edges())
    return CausalGraph(
        nodes,
        edges,
        kind="discovered",
        undirected=[e for e in unresolved if e in set(edges)],
        meta={"algorithm": "pc-stable", "alpha": alpha, "max_cond": max_cond, "n_ci_tests": n_tests, "n_rows": len(df)},
    )
