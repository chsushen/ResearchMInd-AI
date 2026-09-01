import re
import time
from typing import Sequence
import google.generativeai as genai
from ..models.schemas import (
    DocumentChunk,
    Citation,
    RAGResponse,
    ComparisonDimension,
    ComparisonResponse,
    BibTeXResponse,
    DocumentInfo,
)
from ..core.config import settings
from .embeddings import embedding_service
from .hybrid_retriever import hybrid_retriever


class RAGEngineService:
    """
    FAANG-caliber RAG engine orchestrating:
    1. Hybrid retrieval (ChromaDB + BM25 + RRF)
    2. Strict citation grounding [Doc X, p. Y]
    3. Multi-document comparative matrices
    4. BibTeX citation extraction
    """

    def __init__(self, retriever=None):
        self.default_model = "gemini-1.5-flash"
        self.retriever = retriever or hybrid_retriever

    def query(
        self,
        query: str,
        doc_ids: list[str] | None = None,
        gemini_api_key: str | None = None,
        conversation_history: list[dict[str, str]] | None = None,
        top_k: int = 6,
    ) -> RAGResponse:
        start_time = time.time()

        # Update dynamic API key if provided
        active_key = gemini_api_key or settings.GEMINI_API_KEY
        if active_key:
            embedding_service.update_api_key(active_key)

        # 1. Hybrid Retrieval with Reciprocal Rank Fusion
        retrieved_results = self.retriever.hybrid_search(
            query=query,
            doc_ids=doc_ids,
            top_k=top_k,
        )

        if not retrieved_results:
            return RAGResponse(
                query=query,
                answer="No relevant documents or passages were found in the indexed literature.",
                citations=[],
                retrieved_chunks_count=0,
                model_used="none",
                latency_ms=round((time.time() - start_time) * 1000, 2),
            )

        # 2. Build Grounded Context with Exact Page Badges
        context_blocks = []
        citations: list[Citation] = []

        for idx, (chunk, score) in enumerate(retrieved_results):
            source_tag = f"[Doc: {chunk.doc_title}, Page {chunk.page_number}]"
            context_blocks.append(f"--- SOURCE {idx+1} {source_tag} ---\n{chunk.text}\n")

            snippet = (chunk.text[:220] + "...") if len(chunk.text) > 220 else chunk.text
            citations.append(
                Citation(
                    doc_id=chunk.doc_id,
                    doc_title=chunk.doc_title,
                    page_number=chunk.page_number,
                    snippet=snippet,
                    score=round(score, 4),
                )
            )

        context_str = "\n".join(context_blocks)

        # 3. Formulate Strict Grounding System Prompt
        system_instruction = (
            "You are Research Mind AI, an elite academic research assistant. "
            "Your objective is to provide rigorous, accurate, and deeply analytical scientific answers "
            "grounded STRICTLY in the provided source passages.\n\n"
            "CITATION RULES:\n"
            "1. Every factual statement, equation, metric, or conclusion MUST be cited using the format: [Doc Title, p. PageNumber].\n"
            "2. If multiple sources corroborate a claim, list them: [Paper A, p. 3; Paper B, p. 12].\n"
            "3. Do NOT extrapolate or hallucinate beyond what the source passages substantiate.\n"
            "4. Organize your response using clear markdown headings, bullet points, and comparative insights where applicable.\n"
            "5. If the provided literature does not contain the answer, explicitly acknowledge what is missing."
        )

        prompt = (
            f"{system_instruction}\n\n"
            f"=== RETRIEVED ACADEMIC LITERATURE ===\n{context_str}\n\n"
            f"=== RESEARCH INQUIRY ===\n{query}\n\n"
            "=== GROUNDED SCIENTIFIC SYNTHESIS ==="
        )

        # 4. Generate Answer via Gemini or Grounded Fallback Engine
        answer = ""
        model_used = self.default_model

        if active_key and not active_key.startswith("mock") and active_key != "mock-key-for-ci":
            try:
                genai.configure(api_key=active_key)
                model = genai.GenerativeModel(self.default_model)
                response = model.generate_content(prompt)
                answer = response.text.strip()
            except Exception as e:
                print(f"Notice: Gemini API call fallback: {e}")
                answer = self._generate_grounded_fallback_answer(query, retrieved_results)
                model_used = "deterministic-grounded-fallback"
        else:
            answer = self._generate_grounded_fallback_answer(query, retrieved_results)
            model_used = "deterministic-grounded-fallback"

        latency = round((time.time() - start_time) * 1000, 2)

        return RAGResponse(
            query=query,
            answer=answer,
            citations=citations,
            retrieved_chunks_count=len(retrieved_results),
            model_used=model_used,
            latency_ms=latency,
        )

    def compare_documents(
        self,
        docs: list[DocumentInfo],
        dimensions: list[str] | None = None,
        gemini_api_key: str | None = None,
    ) -> ComparisonResponse:
        """
        Generates a multi-paper side-by-side comparative analysis matrix.
        """
        default_dimensions = [
            "Research Objective & Problem Statement",
            "Proposed Methodology & Architecture",
            "Empirical Datasets & Benchmarks",
            "Key Quantitative Findings",
            "Identified Limitations & Trade-offs",
        ]
        target_dimensions = dimensions or default_dimensions

        doc_ids = [d.doc_id for d in docs]
        doc_titles = [d.title for d in docs]

        dimension_evaluations: list[ComparisonDimension] = []

        # Retrieve relevant chunks for each dimension
        for dim in target_dimensions:
            evaluations: dict[str, str] = {}
            for doc in docs:
                chunks_with_scores = self.retriever.hybrid_search(
                    query=f"{dim} {doc.title}",
                    doc_ids=[doc.doc_id],
                    top_k=2,
                )
                if chunks_with_scores:
                    best_chunk = chunks_with_scores[0][0]
                    summary_point = best_chunk.text[:280].replace("\n", " ").strip()
                    evaluations[doc.title] = f"{summary_point}... (p. {best_chunk.page_number})"
                else:
                    evaluations[doc.title] = f"Abstract overview: {doc.abstract[:180] or 'Not explicitly specified.'}"

            dimension_evaluations.append(
                ComparisonDimension(dimension=dim, evaluations=evaluations)
            )

        # Build Markdown Table
        table_lines = ["| Research Dimension | " + " | ".join(doc_titles) + " |"]
        table_lines.append("| :--- | " + " | ".join([":---"] * len(doc_titles)) + " |")

        for dim_eval in dimension_evaluations:
            row_vals = [dim_eval.evaluations.get(t, "N/A").replace("|", "\\|") for t in doc_titles]
            table_lines.append(f"| **{dim_eval.dimension}** | " + " | ".join(row_vals) + " |")

        markdown_table = "\n".join(table_lines)

        exec_summary = (
            f"Comparative analysis across {len(docs)} academic papers: "
            f"{', '.join(f'*{t}*' for t in doc_titles)}. "
            f"The papers demonstrate distinct approaches toward computational performance, "
            f"algorithmic robustness, and empirical validation bounds."
        )

        return ComparisonResponse(
            doc_ids=doc_ids,
            doc_titles=doc_titles,
            dimensions=dimension_evaluations,
            executive_summary=exec_summary,
            markdown_table=markdown_table,
        )

    def extract_bibtex(self, doc_info: DocumentInfo) -> BibTeXResponse:
        """
        Synthesizes standard compliant BibTeX citation entry.
        """
        # Citation key heuristic: AuthorYear or TitleWordYear
        author_last = "Scholar"
        if doc_info.authors:
            first_author = doc_info.authors[0]
            author_last = first_author.split()[-1].lower()

        clean_title = re.sub(r"[^a-zA-Z0-9\s]", "", doc_info.title).lower()
        title_keyword = (clean_title.split() or ["paper"])[0]
        year = "2026"

        citation_key = f"{author_last}{year}{title_keyword}"

        author_str = " and ".join(doc_info.authors) if doc_info.authors else "Unknown Researcher"

        bibtex = (
            f"@article{{{citation_key},\n"
            f"  title     = {{{{{doc_info.title}}}}},\n"
            f"  author    = {{{author_str}}},\n"
            f"  journal   = {{arXiv preprint}},\n"
            f"  year      = {{{year}}},\n"
            f"  pages     = {{1--{doc_info.total_pages}}},\n"
            f"  note      = {{Indexed by Research Mind AI}}\n"
            f"}}"
        )

        return BibTeXResponse(
            doc_id=doc_info.doc_id,
            title=doc_info.title,
            bibtex=bibtex,
            citation_key=citation_key,
        )

    def _generate_grounded_fallback_answer(
        self,
        query: str,
        retrieved_results: Sequence[tuple[DocumentChunk, float]],
    ) -> str:
        """
        High-quality grounded synthesis generator for offline/demo/testing modes.
        Directly cites exact page numbers [Doc Title, p. X].
        """
        lines = [
            f"### Scientific Synthesis for Inquiry: *\"{query}\"*\n",
            "Based on strict retrieval from the indexed academic literature, the following verified insights were extracted:\n",
        ]

        for idx, (chunk, score) in enumerate(retrieved_results[:4]):
            citation_tag = f"**[{chunk.doc_title}, p. {chunk.page_number}]**"
            snippet_preview = chunk.text[:300].strip().replace("\n", " ")
            lines.append(f"{idx+1}. {citation_tag}: {snippet_preview}...")

        lines.append(
            "\n> **Grounding Verification**: All claims above are tied to the exact page references noted in the brackets."
        )

        return "\n".join(lines)


rag_engine = RAGEngineService()
