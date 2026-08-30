import pytest
from fastapi.testclient import TestClient
from backend.app.main import app
from backend.app.services.document_store import document_store
from backend.app.models.schemas import DocumentInfo


@pytest.fixture
def client():
    return TestClient(app)


def test_health_endpoint(client):
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["app"] == "Research Mind AI"
    assert "total_chunks" in data


def test_root_endpoint(client):
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "operational"


def test_query_endpoint(client):
    response = client.post(
        "/api/query",
        json={"query": "What are physics informed neural networks?"},
    )
    assert response.status_code == 200
    data = response.json()
    assert "answer" in data
    assert "citations" in data
    assert "latency_ms" in data


def test_documents_and_bibtex_endpoints(client):
    # Seed a document into registry
    doc = DocumentInfo(
        doc_id="api-test-doc",
        filename="test_paper.pdf",
        title="Automated Multimodal Scientific Discovery",
        authors=["Alice Smith", "Bob Chen"],
        total_pages=10,
        chunk_count=5,
        upload_time="2026-08-30 12:00:00",
        file_size_bytes=5120,
        abstract="A framework for autonomous scientific discovery.",
    )
    document_store.add(doc)

    # 1. List documents
    list_res = client.get("/api/documents")
    assert list_res.status_code == 200
    docs = list_res.json()
    assert any(d["doc_id"] == "api-test-doc" for d in docs)

    # 2. Extract BibTeX
    bib_res = client.post("/api/extract-bibtex", json={"doc_id": "api-test-doc"})
    assert bib_res.status_code == 200
    bib_data = bib_res.json()
    assert "@article{" in bib_data["bibtex"]
    assert "Alice Smith and Bob Chen" in bib_data["bibtex"]

    # 3. Clean up
    del_res = client.delete("/api/documents/api-test-doc")
    assert del_res.status_code == 200
