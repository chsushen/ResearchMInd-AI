import pytest
from backend.app.models.schemas import DocumentChunk, DocumentInfo
from backend.app.services.hybrid_retriever import HybridRetrieverService
from backend.app.services.rag_engine import RAGEngineService
from backend.app.services.pdf_parser import PDFParserService


@pytest.fixture
def sample_chunks():
    return [
        DocumentChunk(
            chunk_id="paper1-p1-c1",
            doc_id="paper1",
            doc_title="Physics-Informed Neural Networks for Fluid Dynamics",
            page_number=1,
            text="Physics-Informed Neural Networks (PINNs) integrate Navier-Stokes equations into the neural loss function.",
            metadata={"page": 1, "doc_id": "paper1"},
        ),
        DocumentChunk(
            chunk_id="paper1-p3-c2",
            doc_id="paper1",
            doc_title="Physics-Informed Neural Networks for Fluid Dynamics",
            page_number=3,
            text="Our empirical benchmark achieves a 42% reduction in L2 relative error compared to traditional finite element solvers.",
            metadata={"page": 3, "doc_id": "paper1"},
        ),
        DocumentChunk(
            chunk_id="paper2-p2-c1",
            doc_id="paper2",
            doc_title="Fourier Neural Operators for Turbulent Flow Simulation",
            page_number=2,
            text="Fourier Neural Operators (FNO) parameterize the integral kernel in Fourier space, achieving zero-shot super-resolution.",
            metadata={"page": 2, "doc_id": "paper2"},
        ),
    ]


def test_pdf_parser_cleaning_and_chunking():
    parser = PDFParserService(chunk_size_chars=100, chunk_overlap_chars=20)
    # Test hyphenation and spacing cleaning
    dirty = "Physics-in-\nformed neu-\nral networks with    irregular   spaces."
    cleaned = parser._clean_text(dirty)
    assert "Physics-informed neural networks with irregular spaces." == cleaned

    # Test chunking with page retention
    pages = [
        (1, "First page content discussing spatiotemporal turbulence models."),
        (2, "Second page empirical benchmarks with 99.4% accuracy."),
    ]
    chunks = parser._create_page_chunks(pages, "doc1", "Sample Study")
    assert len(chunks) >= 2
    assert chunks[0].page_number == 1
    assert chunks[-1].page_number == 2


def test_reciprocal_rank_fusion_logic(tmp_path, sample_chunks):
    # Use temporary directory for ChromaDB
    retriever = HybridRetrieverService(persist_dir=str(tmp_path / "chroma_test"))
    retriever.add_chunks(sample_chunks)

    # Query matching fluid dynamics
    results = retriever.hybrid_search(query="Navier-Stokes loss function", top_k=2)
    assert len(results) > 0
    top_chunk, score = results[0]
    assert score > 0.0
    assert "paper1" in top_chunk.doc_id


def test_filtered_retrieval_by_doc_id(tmp_path, sample_chunks):
    retriever = HybridRetrieverService(persist_dir=str(tmp_path / "chroma_test2"))
    retriever.add_chunks(sample_chunks)

    # Search query that would otherwise match paper1, but restrict to paper2
    results = retriever.hybrid_search(
        query="neural networks simulation",
        doc_ids=["paper2"],
        top_k=2,
    )
    assert len(results) > 0
    for chunk, _ in results:
        assert chunk.doc_id == "paper2"


def test_rag_engine_strict_grounding(tmp_path, sample_chunks):
    retriever = HybridRetrieverService(persist_dir=str(tmp_path / "chroma_test3"))
    retriever.add_chunks(sample_chunks)

    engine = RAGEngineService(retriever=retriever)
    # Execute query
    response = engine.query("What is the L2 relative error reduction?")
    assert response.query == "What is the L2 relative error reduction?"
    assert len(response.citations) > 0
    assert response.retrieved_chunks_count > 0

    # Ensure citation has exact page number
    top_cit = response.citations[0]
    assert top_cit.page_number in [1, 2, 3]


def test_multi_document_comparison(tmp_path, sample_chunks):
    retriever = HybridRetrieverService(persist_dir=str(tmp_path / "chroma_test4"))
    retriever.add_chunks(sample_chunks)

    engine = RAGEngineService(retriever=retriever)
    docs = [
        DocumentInfo(
            doc_id="paper1",
            filename="pinn.pdf",
            title="Physics-Informed Neural Networks for Fluid Dynamics",
            authors=["Raissi et al."],
            total_pages=5,
            chunk_count=2,
            upload_time="2026-08-30",
            file_size_bytes=1024,
            abstract="Study of PINNs for PDE solving.",
        ),
        DocumentInfo(
            doc_id="paper2",
            filename="fno.pdf",
            title="Fourier Neural Operators for Turbulent Flow Simulation",
            authors=["Li et al."],
            total_pages=6,
            chunk_count=1,
            upload_time="2026-08-30",
            file_size_bytes=2048,
            abstract="Study of FNOs for turbulence.",
        ),
    ]

    comp_res = engine.compare_documents(docs)
    assert len(comp_res.dimensions) > 0
    assert "| Research Dimension |" in comp_res.markdown_table
    assert "Physics-Informed Neural Networks" in comp_res.markdown_table
    assert "Fourier Neural Operators" in comp_res.markdown_table


def test_bibtex_generation():
    engine = RAGEngineService()
    doc = DocumentInfo(
        doc_id="test-doc-1",
        filename="raissi2026.pdf",
        title="Deep Physics Learning for Disaster Resilience",
        authors=["Maziar Raissi", "George Karniadakis"],
        total_pages=14,
        chunk_count=8,
        upload_time="2026-08-30",
        file_size_bytes=4096,
        abstract="Deep physics learning review.",
    )
    bib = engine.extract_bibtex(doc)
    assert "@article{" in bib.bibtex
    assert "title     = {{Deep Physics Learning for Disaster Resilience}}" in bib.bibtex
    assert "Maziar Raissi and George Karniadakis" in bib.bibtex


def test_large_document_100_plus_pages():
    """Edge Case: Verifies 100+ pages are chunked with accurate page boundaries."""
    parser = PDFParserService(chunk_size_chars=400, chunk_overlap_chars=50)
    # Generate 120 synthetic pages
    synthetic_pages = [
        (i, f"Page {i} academic dissertation text on PDE residual analysis and domain decomposition.")
        for i in range(1, 121)
    ]
    chunks = parser._create_page_chunks(synthetic_pages, "dissertation-1", "Ph.D. Thesis on PINNs")
    assert len(chunks) >= 120
    assert chunks[0].page_number == 1
    assert chunks[-1].page_number == 120
    # Check page numbers are strictly monotonic
    for idx in range(len(chunks) - 1):
        assert chunks[idx].page_number <= chunks[idx + 1].page_number


def test_scanned_document_empty_page_handling():
    """Edge Case: Verifies scanned/empty pages do not crash parser or produce invalid chunks."""
    parser = PDFParserService()
    # Mixed real and scanned/blank pages
    pages = [
        (1, "Title: Scanned Paper Test\n\nAbstract: Testing scanned handling."),
        (2, "   \n  \n"),  # Blank scanned page
        (3, "Actual methodology text on page 3."),
    ]
    chunks = parser._create_page_chunks(pages, "scanned-doc", "Scanned Document Study")
    assert len(chunks) == 2
    assert chunks[0].page_number == 1
    assert chunks[1].page_number == 3


def test_concurrent_queries_execution(tmp_path, sample_chunks):
    """Edge Case: Verifies thread-safe concurrent queries execute cleanly."""
    import concurrent.futures

    retriever = HybridRetrieverService(persist_dir=str(tmp_path / "chroma_concurrent"))
    retriever.add_chunks(sample_chunks)
    engine = RAGEngineService(retriever=retriever)

    queries = [
        "What is Navier-Stokes error?",
        "Explain Fourier Neural Operators",
        "What is the empirical reduction in error?",
        "How do PINNs formulate loss?",
        "Compare fluid dynamics solvers",
    ] * 4  # 20 concurrent queries

    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(engine.query, q) for q in queries]
        results = [f.result() for f in concurrent.futures.as_completed(futures)]

    assert len(results) == 20
    for r in results:
        assert r.query in queries
        assert r.latency_ms >= 0.0

