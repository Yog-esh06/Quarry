from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def test_health() -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_graph_is_explicit_and_acyclic() -> None:
    response = client.get("/api/v1/causal/graph")
    body = response.json()
    assert body["kind"] == "configured"
    assert ["channel", "default"] in body["edges"]
    assert body["meta"]["treatment"] == "channel"


def test_summary_is_reproducible() -> None:
    first = client.get("/api/v1/datasets/demo/summary?rows=200&seed=7").json()
    second = client.get("/api/v1/datasets/demo/summary?rows=200&seed=7").json()
    assert first == second
    assert first["synthetic"] is True
