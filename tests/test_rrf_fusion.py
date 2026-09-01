"""Unit tests for Reciprocal Rank Fusion (RRF) and parallel hybrid retrieval."""

import pytest
from backend.app.services.hybrid_retriever import HybridRetrieverService
from backend.app.models.schemas import DocumentChunk


def test_rrf_fusion_formula_calculation():
    dense_ranks = {"chunk_1": 1, "chunk_2": 2, "chunk_3": 3}
    sparse_ranks = {"chunk_2": 1, "chunk_1": 3, "chunk_4": 2}

    # k = 60
    scores = HybridRetrieverService.compute_rrf_fusion(dense_ranks, sparse_ranks, k=60)

    # chunk_1: 1/(60+1) + 1/(60+3) = 1/61 + 1/63 = 0.0163934 + 0.0158730 = 0.0322664
    # chunk_2: 1/(60+2) + 1/(60+1) = 1/62 + 1/61 = 0.0161290 + 0.0163934 = 0.0325224
    # chunk_3: 1/(60+3) = 1/63 = 0.0158730
    # chunk_4: 1/(60+2) = 1/62 = 0.0161290
    assert scores["chunk_2"] > scores["chunk_1"]
    assert scores["chunk_1"] > scores["chunk_4"]
    assert scores["chunk_4"] > scores["chunk_3"]
    assert abs(scores["chunk_2"] - (1 / 62 + 1 / 61)) < 1e-6


def test_rrf_fusion_edge_cases():
    # Empty inputs
    assert HybridRetrieverService.compute_rrf_fusion({}, {}, k=60) == {}

    # Disjoint inputs
    scores = HybridRetrieverService.compute_rrf_fusion({"c1": 1}, {"c2": 1}, k=60)
    assert scores["c1"] == scores["c2"]
    assert scores["c1"] == 1 / 61

    # Varying k
    scores_k30 = HybridRetrieverService.compute_rrf_fusion({"c1": 1}, {}, k=30)
    assert scores_k30["c1"] == 1 / 31


def test_parallel_hybrid_search_execution():
    retriever = HybridRetrieverService()
    if not retriever.all_chunks:
        # Seed test chunks
        chunk_a = DocumentChunk(
            chunk_id="test_c1",
            doc_id="d1",
            doc_title="Paper Alpha",
            page_number=1,
            text="Deep neural networks and transformer self-attention mechanisms.",
        )
        chunk_b = DocumentChunk(
            chunk_id="test_c2",
            doc_id="d2",
            doc_title="Paper Beta",
            page_number=2,
            text="Physics-informed neural networks and partial differential equations.",
        )
        retriever.all_chunks["test_c1"] = chunk_a
        retriever.all_chunks["test_c2"] = chunk_b
        retriever.bm25_chunk_ids = ["test_c1", "test_c2"]
        retriever.bm25_corpus = [["deep", "neural", "networks"], ["physics", "informed", "neural", "networks"]]
        from rank_bm25 import BM25Okapi
        retriever.bm25_model = BM25Okapi(retriever.bm25_corpus)

    results = retriever.hybrid_search(query="neural networks", top_k=2)
    assert len(results) >= 1
    chunk, score = results[0]
    assert score > 0.0
    assert isinstance(chunk, DocumentChunk)
