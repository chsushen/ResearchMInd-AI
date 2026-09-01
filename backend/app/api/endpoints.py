import os
import uuid
import shutil
from pathlib import Path
from fastapi import APIRouter, UploadFile, File, HTTPException, Body, status, Request
from ..models.schemas import (
    DocumentInfo,
    UploadResponse,
    QueryRequest,
    RAGResponse,
    ComparisonRequest,
    ComparisonResponse,
    BibTeXResponse,
    HealthResponse,
    UsageResponse,
    DeepResearchRequest,
    DeepResearchResponse,
)
from ..core.config import settings
from ..core.rate_limiter import rate_limiter
from ..services.pdf_parser import pdf_parser
from ..services.hybrid_retriever import hybrid_retriever
from ..services.rag_engine import rag_engine
from ..services.deep_researcher import deep_researcher
from ..services.document_store import document_store
from ..services.embeddings import embedding_service

router = APIRouter()


@router.post("/upload", response_model=UploadResponse, status_code=status.HTTP_201_CREATED)
async def upload_pdf(file: UploadFile = File(...)):
    """
    Uploads, parses, and indexes an academic research paper PDF.
    Preserves exact page numbers and metadata for hybrid RAG retrieval.
    """
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Unsupported file format. Please upload a standard PDF (.pdf) document.",
        )

    doc_id = str(uuid.uuid4())[:8]
    save_path = Path(settings.UPLOAD_DIR) / f"{doc_id}_{file.filename}"

    # Persist raw PDF file to disk
    try:
        with open(save_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to store file on disk: {e}")

    # Parse and chunk with strict page boundary preservation
    try:
        doc_info, chunks = pdf_parser.parse_pdf(
            file_obj=save_path,
            filename=file.filename,
            doc_id=doc_id,
        )
    except Exception as e:
        if os.path.exists(save_path):
            os.remove(save_path)
        raise HTTPException(status_code=422, detail=f"Failed to parse PDF document: {e}")

    # Index chunks into Hybrid Retriever (ChromaDB + BM25)
    try:
        hybrid_retriever.add_chunks(chunks)
        document_store.add(doc_info)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to index document in vector store: {e}")

    return UploadResponse(
        doc_id=doc_info.doc_id,
        filename=doc_info.filename,
        title=doc_info.title,
        pages_processed=doc_info.total_pages,
        chunks_created=doc_info.chunk_count,
        message=f"Successfully indexed '{doc_info.title}' across {doc_info.total_pages} pages.",
    )


@router.post("/query", response_model=RAGResponse)
async def query_rag(request: QueryRequest, http_req: Request):
    """
    Executes hybrid retrieval (ChromaDB dense + BM25 with RRF) and synthesizes
    a strictly grounded answer with page-level citations.
    Enforces SaaS token-bucket rate limiting (Free vs. Pro).
    """
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    # Enforce tier-based rate limits
    rate_limiter.check_rate_limit(http_req, estimated_tokens=len(request.query) // 4 + 100)

    try:
        response = rag_engine.query(
            query=request.query,
            doc_ids=request.doc_ids,
            gemini_api_key=request.gemini_api_key,
            conversation_history=request.conversation_history,
            top_k=request.top_k,
        )
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"RAG query execution failed: {e}")


@router.post("/compare", response_model=ComparisonResponse)
async def compare_papers(request: ComparisonRequest, http_req: Request):
    """
    Generates a structured, multi-dimensional comparative analysis matrix
    across multiple indexed academic papers.
    """
    if not request.doc_ids or len(request.doc_ids) < 2:
        raise HTTPException(
            status_code=400,
            detail="Please provide at least two valid document IDs to perform comparative analysis.",
        )

    rate_limiter.check_rate_limit(http_req, estimated_tokens=400)

    docs: list[DocumentInfo] = []
    for did in request.doc_ids:
        doc = document_store.get(did)
        if not doc:
            raise HTTPException(
                status_code=404,
                detail=f"Document with ID '{did}' not found in registry.",
            )
        docs.append(doc)

    try:
        comparison = rag_engine.compare_documents(
            docs=docs,
            dimensions=request.dimensions,
            gemini_api_key=request.gemini_api_key,
        )
        return comparison
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Document comparison failed: {e}")


@router.post("/deep-research", response_model=DeepResearchResponse)
async def execute_deep_research(request: DeepResearchRequest, http_req: Request):
    """
    Executes autonomous cross-document synthesis, empirical claim verification,
    contradiction detection, and BibTeX citation generation.
    """
    if not request.topic.strip():
        raise HTTPException(status_code=400, detail="Topic cannot be empty.")

    rate_limiter.check_rate_limit(http_req, estimated_tokens=800)

    try:
        response = deep_researcher.deep_research(
            topic=request.topic,
            doc_ids=request.doc_ids,
            gemini_api_key=request.gemini_api_key,
            extract_contradictions=request.extract_contradictions,
        )
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Deep research execution failed: {e}")


@router.post("/extract-bibtex", response_model=BibTeXResponse)
async def extract_bibtex(payload: dict = Body(...)):
    """
    Synthesizes and exports standard formatted BibTeX citation for an indexed paper.
    """
    doc_id = payload.get("doc_id")
    if not doc_id:
        raise HTTPException(status_code=400, detail="Field 'doc_id' is required.")

    doc_info = document_store.get(doc_id)
    if not doc_info:
        raise HTTPException(status_code=404, detail=f"Document '{doc_id}' not found.")

    try:
        bibtex_res = rag_engine.extract_bibtex(doc_info)
        return bibtex_res
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"BibTeX extraction failed: {e}")


@router.get("/documents", response_model=list[DocumentInfo])
async def list_documents():
    """Returns all currently indexed academic documents and metadata."""
    return document_store.list_all()


@router.delete("/documents/{doc_id}", status_code=status.HTTP_200_OK)
async def delete_document(doc_id: str):
    """Removes a document and its chunks from vector indices and disk."""
    doc = document_store.get(doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail=f"Document '{doc_id}' not found.")

    hybrid_retriever.delete_document(doc_id)
    document_store.delete(doc_id)

    # Attempt to remove file from uploads folder
    for f in Path(settings.UPLOAD_DIR).glob(f"{doc_id}_*"):
        try:
            os.remove(f)
        except Exception:
            pass

    return {"message": f"Document '{doc.title}' ({doc_id}) successfully deleted."}


@router.get("/usage", response_model=UsageResponse)
async def get_usage(http_req: Request):
    """Returns SaaS usage metrics, query quota, and active subscription tier."""
    usage = rate_limiter.get_usage(http_req)
    return UsageResponse(
        tier=usage.tier,
        queries_used=usage.queries_used,
        queries_limit=usage.queries_limit,
        queries_remaining=usage.queries_remaining,
        total_tokens_consumed=usage.total_tokens_consumed,
        reset_in_seconds=usage.reset_in_seconds,
        is_rate_limited=usage.queries_remaining <= 0 and usage.tier == "free",
    )


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """System health check and diagnostic statistics."""
    has_key = bool(settings.GEMINI_API_KEY or embedding_service.client_configured)
    return HealthResponse(
        status="healthy",
        app=settings.APP_NAME,
        version=settings.APP_VERSION,
        indexed_documents=len(document_store.list_all()),
        total_chunks=hybrid_retriever.get_total_chunks(),
        gemini_configured=has_key,
        saas_rate_limiter_active=True,
    )
