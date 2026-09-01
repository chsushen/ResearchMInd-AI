"""Unit and API tests for Deep Research cross-document reasoning agent."""

import pytest
from fastapi.testclient import TestClient
from backend.app.main import app
from backend.app.services.deep_researcher import deep_researcher
from backend.app.services.document_store import document_store
from backend.app.models.schemas import DocumentInfo

client = TestClient(app)


def test_deep_research_service_synthesis():
    # Seed a document into the store to ensure synthesis has target manuscripts
    doc = DocumentInfo(
        doc_id="deep-test-doc",
        filename="test_paper.pdf",
        title="Physics-Informed Neural Networks for Fluid Dynamics",
        authors=["Alice Smith"],
        total_pages=5,
        chunk_count=2,
        upload_time="2026-08-30 12:00:00",
        file_size_bytes=4096,
        abstract="Physics-informed neural networks for fluid dynamics.",
    )
    document_store.add(doc)

    response = deep_researcher.deep_research(topic="Physics-Informed Neural Networks and Flow Fields")

    assert response.topic == "Physics-Informed Neural Networks and Flow Fields"
    assert len(response.executive_synthesis) > 50
    assert response.latency_ms >= 0
    assert "Manuscript" in response.comparative_matrix_markdown or "Document" in response.comparative_matrix_markdown
    # Verify BibTeX exports exist if documents are indexed
    if document_store.list_all():
        assert len(response.bibtex_citations) >= 1
        assert "@article" in response.bibtex_citations[0] or "@inproceedings" in response.bibtex_citations[0]


def test_deep_research_endpoint():
    payload = {
        "topic": "Uncertainty Quantification in Fluid Mechanics",
        "extract_contradictions": True,
    }
    res = client.post("/api/deep-research", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["topic"] == "Uncertainty Quantification in Fluid Mechanics"
    assert "executive_synthesis" in data
    assert "comparative_matrix_markdown" in data
    assert isinstance(data["key_claims"], list)
    assert isinstance(data["contradictions"], list)
