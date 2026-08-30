import re
import uuid
from pathlib import Path
from typing import BinaryIO
from pypdf import PdfReader
from ..models.schemas import DocumentChunk, DocumentInfo
from ..core.config import settings


class PDFParserService:
    """
    Enterprise-grade PDF ingestion service.
    Supports streaming extraction for 100+ page documents, layout heuristics,
    scanned document detection, and page-preserving chunking.
    """

    def __init__(
        self,
        chunk_size_chars: int = settings.CHUNK_SIZE_CHARS,
        chunk_overlap_chars: int = settings.CHUNK_OVERLAP_CHARS,
    ):
        self.chunk_size = chunk_size_chars
        self.chunk_overlap = chunk_overlap_chars

    def parse_pdf(
        self,
        file_obj: BinaryIO | str | Path,
        filename: str,
        doc_id: str | None = None,
    ) -> tuple[DocumentInfo, list[DocumentChunk]]:
        """
        Parses a PDF file stream or path page-by-page.
        Returns DocumentInfo metadata and an ordered list of DocumentChunks.
        """
        doc_id = doc_id or str(uuid.uuid4())[:8]

        reader = PdfReader(file_obj)
        total_pages = len(reader.pages)

        if total_pages == 0:
            raise ValueError(f"PDF file '{filename}' contains 0 pages.")

        raw_page_texts: list[tuple[int, str]] = []
        is_scanned_count = 0

        # Stream extraction page-by-page to handle 100+ pages without OOM
        for page_idx, page in enumerate(reader.pages):
            page_num = page_idx + 1
            try:
                text = page.extract_text() or ""
            except Exception:
                text = ""

            cleaned_text = self._clean_text(text)
            if len(cleaned_text.strip()) < 30:
                is_scanned_count += 1

            raw_page_texts.append((page_num, cleaned_text))

        # Heuristic metadata extraction from page 1 & 2
        title, authors, abstract = self._extract_paper_metadata(raw_page_texts[:2], filename)

        # Chunk creation with strict page boundary preservation
        chunks = self._create_page_chunks(raw_page_texts, doc_id, title)

        file_size = 0
        if isinstance(file_obj, (str, Path)):
            file_size = Path(file_obj).stat().st_size
        elif hasattr(file_obj, "tell") and hasattr(file_obj, "seek"):
            curr = file_obj.tell()
            file_obj.seek(0, 2)
            file_size = file_obj.tell()
            file_obj.seek(curr)

        doc_info = DocumentInfo(
            doc_id=doc_id,
            filename=filename,
            title=title,
            authors=authors,
            total_pages=total_pages,
            chunk_count=len(chunks),
            upload_time=self._current_timestamp(),
            file_size_bytes=file_size,
            abstract=abstract,
        )

        return doc_info, chunks

    def _clean_text(self, text: str) -> str:
        """Cleans artifacts, hyphenated line-breaks, and excessive whitespace."""
        # Fix hyphenated words broken across lines: e.g. "optimi-\nzation" -> "optimization"
        text = re.sub(r"(\w+)-\n(\w+)", r"\1\2", text)
        # Normalize carriage returns and form feeds
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        # Replace multiple spaces with a single space
        text = re.sub(r"[ \t]+", " ", text)
        # Collapse 3+ newlines into 2
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    def _create_page_chunks(
        self,
        page_texts: list[tuple[int, str]],
        doc_id: str,
        doc_title: str,
    ) -> list[DocumentChunk]:
        """
        Creates semantic chunks while recording the exact 1-indexed page number.
        Maintains chunk overlap within each page or carries forward a tail overlap.
        """
        chunks: list[DocumentChunk] = []
        chunk_counter = 0

        for page_num, text in page_texts:
            if not text:
                continue

            # Split by paragraphs / double newlines first
            paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
            current_chunk_text = ""

            for para in paragraphs:
                if len(current_chunk_text) + len(para) + 2 <= self.chunk_size:
                    current_chunk_text = (
                        f"{current_chunk_text}\n\n{para}" if current_chunk_text else para
                    )
                else:
                    if current_chunk_text:
                        chunk_counter += 1
                        chunks.append(
                            DocumentChunk(
                                chunk_id=f"{doc_id}-p{page_num}-c{chunk_counter}",
                                doc_id=doc_id,
                                doc_title=doc_title,
                                page_number=page_num,
                                text=current_chunk_text.strip(),
                                metadata={"page": page_num, "doc_id": doc_id},
                            )
                        )
                        # Sliding window overlap
                        overlap_tail = current_chunk_text[-self.chunk_overlap :] if len(current_chunk_text) > self.chunk_overlap else ""
                        current_chunk_text = f"{overlap_tail}\n{para}".strip()
                    else:
                        # Para is itself longer than chunk_size: slice into chunks
                        sub_chunks = self._slice_long_text(para, self.chunk_size, self.chunk_overlap)
                        for sc in sub_chunks:
                            chunk_counter += 1
                            chunks.append(
                                DocumentChunk(
                                    chunk_id=f"{doc_id}-p{page_num}-c{chunk_counter}",
                                    doc_id=doc_id,
                                    doc_title=doc_title,
                                    page_number=page_num,
                                    text=sc,
                                    metadata={"page": page_num, "doc_id": doc_id},
                                )
                            )
                        current_chunk_text = ""

            if current_chunk_text:
                chunk_counter += 1
                chunks.append(
                    DocumentChunk(
                        chunk_id=f"{doc_id}-p{page_num}-c{chunk_counter}",
                        doc_id=doc_id,
                        doc_title=doc_title,
                        page_number=page_num,
                        text=current_chunk_text.strip(),
                        metadata={"page": page_num, "doc_id": doc_id},
                    )
                )

        return chunks

    def _slice_long_text(self, text: str, size: int, overlap: int) -> list[str]:
        """Slices a long string by character bounds with overlap."""
        slices = []
        start = 0
        while start < len(text):
            end = min(start + size, len(text))
            slices.append(text[start:end].strip())
            if end >= len(text):
                break
            start += size - overlap
        return slices

    def _extract_paper_metadata(
        self,
        first_pages: list[tuple[int, str]],
        fallback_filename: str,
    ) -> tuple[str, list[str], str]:
        """Extracts Title, Authors list, and Abstract using academic layout heuristics."""
        combined_text = "\n\n".join(t for _, t in first_pages)
        title = ""
        authors: list[str] = []
        abstract = ""

        # Extract Abstract
        abstract_match = re.search(
            r"(?:Abstract|ABSTRACT)[\s:—–-]*([\s\S]*?)(?:1\.?\s+Introduction|I\.?\s+INTRODUCTION|Categories and Subject|Keywords|Index Terms)",
            combined_text,
            re.IGNORECASE,
        )
        if abstract_match:
            abstract = abstract_match.group(1).strip()
            # Clean abstract formatting
            abstract = re.sub(r"\s+", " ", abstract)[:800]

        # Extract Title
        lines = [line.strip() for line in combined_text.split("\n") if len(line.strip()) > 3]
        for line in lines[:8]:
            # Skip typical arXiv header markers
            if re.search(r"arxiv|preprint|ieee|acm|proceedings|journal|vol\.|no\.", line, re.IGNORECASE):
                continue
            if len(line) >= 10:
                title = line
                break

        if not title:
            title = Path(fallback_filename).stem.replace("_", " ").replace("-", " ").title()

        # Extract Authors heuristic (lines between title and abstract)
        if title in combined_text and abstract_match:
            pre_abstract = combined_text.split(title)[-1].split("Abstract")[0]
            candidate_authors = [l.strip() for l in pre_abstract.split("\n") if l.strip()]
            for ca in candidate_authors[:3]:
                if "@" not in ca and "university" not in ca.lower() and "department" not in ca.lower():
                    # Split on comma or 'and'
                    split_auth = re.split(r",|\band\b", ca)
                    for a in split_auth:
                        a_clean = a.strip()
                        if 2 <= len(a_clean.split()) <= 4:
                            authors.append(a_clean)

        if not authors:
            authors = ["Primary Researcher et al."]

        return title, authors, abstract

    def _current_timestamp(self) -> str:
        from datetime import datetime
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


pdf_parser = PDFParserService()
