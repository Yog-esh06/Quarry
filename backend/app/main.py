"""HTTP boundary for the local QUARRY demonstration."""

from __future__ import annotations

from functools import lru_cache

import numpy as np
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from app.ml.causal.graph import CausalGraph
from app.ml.data.schema import lending_schema
from app.ml.simulation.scm import FEATURES, TARGET, TRUE_EDGES, sample_reference

app = FastAPI(
    title="QUARRY API",
    version="1.0.0",
    description="Assumption-aware causal decision intelligence for intervention analysis.",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


class SampleRequest(BaseModel):
    rows: int = Field(default=1000, ge=100, le=20_000)
    seed: int = Field(default=42, ge=0)


def _graph() -> CausalGraph:
    return CausalGraph(
        nodes=[*FEATURES, TARGET],
        edges=list(TRUE_EDGES),
        kind="configured",
        meta={
            "label": "Configured causal structure",
            "warning": "Causal conclusions depend on this domain assumption.",
            "treatment": "channel",
            "outcome": TARGET,
        },
    )


@lru_cache(maxsize=8)
def _reference(rows: int, seed: int) -> tuple:
    return sample_reference(rows, seed)


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "service": "quarry-api", "version": app.version}


@app.get("/api/v1/causal/graph")
def causal_graph() -> dict:
    return _graph().to_dict()


@app.get("/api/v1/schema")
def schema() -> dict:
    return lending_schema().to_dict()


@app.get("/api/v1/datasets/demo/summary")
def dataset_summary(rows: int = Query(default=1000, ge=100, le=20_000), seed: int = Query(default=42, ge=0)) -> dict:
    _, y = _reference(rows, seed)
    return {
        "dataset_id": "demo-lending-scm",
        "rows": rows,
        "features": len(FEATURES),
        "target": TARGET,
        "default_rate": float(np.mean(y)),
        "synthetic": True,
        "seed": seed,
        "note": "Synthetic structural causal model; hidden ground truth is reserved for evaluation.",
    }


@app.post("/api/v1/datasets/demo/sample")
def dataset_sample(request: SampleRequest) -> dict:
    frame, y = _reference(request.rows, request.seed)
    preview = frame.head(20).replace({np.nan: None}).to_dict(orient="records")
    return {"dataset_id": "demo-lending-scm", "rows": request.rows, "target": TARGET, "preview": preview, "outcome_preview": y[:20].tolist()}


@app.get("/api/v1/assumptions")
def assumptions() -> dict:
    return {
        "treatment": "channel",
        "outcome": TARGET,
        "adjustment_variables": ["region", "age", "income", "credit_score", "debt_to_income"],
        "required": [
            "Consistency: observed outcomes correspond to the defined intervention.",
            "Positivity: comparable applicants have support across treatment channels.",
            "Conditional exchangeability: adjustment variables block relevant confounding.",
        ],
        "warning": "These are modeling assumptions, not empirically proven facts.",
    }


@app.get("/api/v1/causal/edges/{node}")
def node_edges(node: str) -> dict:
    graph = _graph()
    if node not in graph.nodes:
        raise HTTPException(status_code=404, detail=f"Unknown graph node: {node}")
    return {"node": node, "parents": graph.parents(node), "children": graph.children(node)}
