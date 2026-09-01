"""Unit and API tests for Deep Research cross-document reasoning agent."""

import pytest
from fastapi.testclient import TestClient
from backend.app.main import app
from backend.app.services.deep_researcher import deep_researcher
from backend.app.services.document_store import document_store
from backend.app.models.schemas import DocumentInfo

client = TestClient(app)


def test_deep_research_service_synthesis():
    response = deep_researcher.deep_research(topic="Physics-Informed Neural Networks and Flow Fields")

    assert response.topic == "Physics-Informed Neural Networks and Flow Fields"
    assert len(response.executive_synthesis) > 50
    assert response.latency_ms > 0
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
