"""Causal DAG over the monitored variables (features + target)."""

from __future__ import annotations

from dataclasses import dataclass, field

import networkx as nx


@dataclass
class CausalGraph:
    nodes: list[str]
    edges: list[tuple[str, str]]
    kind: str = "declared"  # "declared" (domain knowledge) | "discovered" (PC) | "empty" (ablation)
    undirected: list[tuple[str, str]] = field(default_factory=list)  # edges PC could not orient
    meta: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.edges = [tuple(e) for e in self.edges]
        self.undirected = [tuple(e) for e in self.undirected]
        unknown = {v for e in self.edges for v in e} - set(self.nodes)
        if unknown:
            raise ValueError(f"edges reference unknown nodes: {sorted(unknown)}")
        g = nx.DiGraph()
        g.add_nodes_from(self.nodes)
        g.add_edges_from(self.edges)
        if not nx.is_directed_acyclic_graph(g):
            raise ValueError("causal graph must be acyclic")
        self._g = g

    def parents(self, v: str) -> list[str]:
        return sorted(self._g.predecessors(v))

    def children(self, v: str) -> list[str]:
        return sorted(self._g.successors(v))

    def ancestors(self, v: str) -> set[str]:
        return nx.ancestors(self._g, v)

    def descendants(self, v: str) -> set[str]:
        return nx.descendants(self._g, v)

    def topological_order(self) -> list[str]:
        return list(nx.lexicographical_topological_sort(self._g))

    def layers(self) -> dict[str, int]:
        """Longest-path depth of each node (for layered drawing)."""
        depth: dict[str, int] = {}
        for v in self.topological_order():
            ps = self.parents(v)
            depth[v] = 0 if not ps else 1 + max(depth[p] for p in ps)
        return depth

    def restricted(self, nodes: list[str]) -> "CausalGraph":
        keep = set(nodes)
        return CausalGraph(
            [n for n in self.nodes if n in keep],
            [e for e in self.edges if e[0] in keep and e[1] in keep],
            self.kind,
            [e for e in self.undirected if e[0] in keep and e[1] in keep],
            dict(self.meta),
        )

    def to_dict(self) -> dict:
        return {
            "kind": self.kind,
            "nodes": list(self.nodes),
            "edges": [list(e) for e in self.edges],
            "undirected": [list(e) for e in self.undirected],
            "meta": self.meta,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "CausalGraph":
        return cls(d["nodes"], [tuple(e) for e in d["edges"]], d.get("kind", "declared"),
                   [tuple(e) for e in d.get("undirected", [])], d.get("meta", {}))

    @classmethod
    def empty(cls, nodes: list[str]) -> "CausalGraph":
        return cls(list(nodes), [], "empty")


def compare_graphs(estimated: CausalGraph, truth: CausalGraph) -> dict[str, float]:
    """Structural Hamming distance and edge precision/recall (skeleton and orientation)."""
    est = set(estimated.edges)
    tru = set(truth.edges)
    skel = lambda es: {frozenset(e) for e in es}  # noqa: E731
    est_skel = skel(estimated.edges) | skel(estimated.undirected)
    tru_skel = skel(truth.edges)
    tp_skel = len(est_skel & tru_skel)
    tp_dir = len(est & tru)
    # SHD: missing + extra skeleton edges + wrongly oriented / unoriented true edges
    missing = len(tru_skel - est_skel)
    extra = len(est_skel - tru_skel)
    misoriented = sum(1 for e in tru if frozenset(e) in est_skel and e not in est)
    return {
        "shd": float(missing + extra + misoriented),
        "skeleton_precision": tp_skel / max(len(est_skel), 1),
        "skeleton_recall": tp_skel / max(len(tru_skel), 1),
        "orientation_precision": tp_dir / max(len(est), 1),
        "orientation_recall": tp_dir / max(len(tru), 1),
        "n_edges_true": float(len(tru_skel)),
        "n_edges_estimated": float(len(est_skel)),
        "n_unoriented": float(len(estimated.undirected)),
    }
