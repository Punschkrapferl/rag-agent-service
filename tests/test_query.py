from fastapi.testclient import TestClient

from app.main import app
from app.api.routes_query import get_vector_store


class DummyVectorStore:
    """Minimal stand-in for VectorStore used during tests."""

    def search(self, query_embedding, top_k: int = 5, qfilter=None):
        # Simulate an empty index: no documents found
        return []


def override_get_vector_store():
    # Return dummy instead of the real Qdrant-backed VectorStore
    return DummyVectorStore()


# IMPORTANT: register override before creating TestClient
app.dependency_overrides[get_vector_store] = override_get_vector_store

client = TestClient(app)


def test_query_empty_index() -> None:
    payload = {"query": "test", "top_k": 3}
    resp = client.post("/api/query", json=payload)

    assert resp.status_code == 200
    data = resp.json()

    assert data["query"] == "test"
    assert "No relevant documents" in data["answer"]
    assert data["documents"] == []
