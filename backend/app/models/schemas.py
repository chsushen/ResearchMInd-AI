from typing import Any
from pydantic import BaseModel, Field


class DocumentChunk(BaseModel):
    chunk_id: str
    doc_id: str
    doc_title: str
    page_number: int
    text: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class DocumentInfo(BaseModel):
    doc_id: str
    filename: str
    title: str
    authors: list[str] = Field(default_factory=list)
    total_pages: int
    chunk_count: int
    upload_time: str
    file_size_bytes: int
    abstract: str = ""


class UploadResponse(BaseModel):
    doc_id: str
    filename: str
    title: str
    pages_processed: int
    chunks_created: int
    message: str


class Citation(BaseModel):
    doc_id: str
    doc_title: str
    page_number: int
    snippet: str
    score: float = 0.0


class QueryRequest(BaseModel):
    query: str
    doc_ids: list[str] | None = None
    gemini_api_key: str | None = None
    conversation_history: list[dict[str, str]] | None = None
    top_k: int = 6


class RAGResponse(BaseModel):
    query: str
    answer: str
    citations: list[Citation]
    retrieved_chunks_count: int
    model_used: str
    latency_ms: float


class ComparisonRequest(BaseModel):
    doc_ids: list[str]
    dimensions: list[str] | None = None
    gemini_api_key: str | None = None


class ComparisonDimension(BaseModel):
    dimension: str
    evaluations: dict[str, str]  # doc_title -> evaluation text


class ComparisonResponse(BaseModel):
    doc_ids: list[str]
    doc_titles: list[str]
    dimensions: list[ComparisonDimension]
    executive_summary: str
    markdown_table: str


class BibTeXResponse(BaseModel):
    doc_id: str
    title: str
    bibtex: str
    citation_key: str


class HealthResponse(BaseModel):
    status: str
    app: str
    version: str
    indexed_documents: int
    total_chunks: int
    gemini_configured: bool
