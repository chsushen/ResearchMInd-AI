import re
import chromadb
from chromadb.config import Settings as ChromaSettings
from rank_bm25 import BM25Okapi
from ..models.schemas import DocumentChunk
from ..core.config import settings
from .embeddings import embedding_service


class HybridRetrieverService:
    """
    Production Hybrid Retriever combining:
    1. ChromaDB (Dense Vector Search)
    2. BM25Okapi (Sparse Lexical Search)
    3. Reciprocal Rank Fusion (RRF, k=60)
    """

    def __init__(self, persist_dir: str = settings.CHROMA_PERSIST_DIR):
        self.persist_dir = persist_dir
        self.chroma_client = chromadb.PersistentClient(
            path=persist_dir,
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self.collection = self.chroma_client.get_or_create_collection(
            name="research_mind_chunks",
            metadata={"hnsw:space": "cosine"},
        )

        # In-memory BM25 index and chunk cache
        self.all_chunks: dict[str, DocumentChunk] = {}
        self.bm25_corpus: list[list[str]] = []
        self.bm25_chunk_ids: list[str] = []
        self.bm25_model: BM25Okapi | None = None

        # Rehydrate in-memory sparse index from Chroma on startup
        self._rehydrate_from_chroma()

    def _tokenize(self, text: str) -> list[str]:
        return re.findall(r"\w+", text.lower())

    def _rehydrate_from_chroma(self) -> None:
        """Loads existing chunks from Chroma into BM25 index upon startup."""
        try:
            results = self.collection.get()
            ids = results.get("ids", [])
            documents = results.get("documents", [])
            metadatas = results.get("metadatas", [])

            if ids and documents:
                for chunk_id, doc_text, meta in zip(ids, documents, metadatas):
                    chunk = DocumentChunk(
                        chunk_id=chunk_id,
                        doc_id=meta.get("doc_id", "unknown"),
                        doc_title=meta.get("doc_title", "Untitled Document"),
                        page_number=int(meta.get("page_number", 1)),
                        text=doc_text,
                        metadata=meta,
                    )
                    self.all_chunks[chunk_id] = chunk

                self._rebuild_bm25_index()
        except Exception as e:
            print(f"Notice: Initial rehydration of BM25 from Chroma: {e}")

    def _rebuild_bm25_index(self) -> None:
        """Rebuilds the BM25Okapi model from current in-memory corpus."""
        if not self.all_chunks:
            self.bm25_model = None
            self.bm25_corpus = []
            self.bm25_chunk_ids = []
            return

        self.bm25_chunk_ids = list(self.all_chunks.keys())
        self.bm25_corpus = [self._tokenize(self.all_chunks[cid].text) for cid in self.bm25_chunk_ids]
        self.bm25_model = BM25Okapi(self.bm25_corpus)

    def add_chunks(self, chunks: list[DocumentChunk]) -> None:
        """Indexes a list of document chunks into both ChromaDB and BM25."""
        if not chunks:
            return

        ids = [c.chunk_id for c in chunks]
        texts = [c.text for c in chunks]
        metadatas = [
            {
                "doc_id": c.doc_id,
                "doc_title": c.doc_title,
                "page_number": c.page_number,
                **c.metadata,
            }
            for c in chunks
        ]

        # Generate dense embeddings
        embeddings = embedding_service.get_embeddings(texts)

        # Upsert into ChromaDB
        self.collection.upsert(
            ids=ids,
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas,
        )

        # Update in-memory state for BM25
        for chunk in chunks:
            self.all_chunks[chunk.chunk_id] = chunk

        self._rebuild_bm25_index()

    def delete_document(self, doc_id: str) -> None:
        """Deletes all chunks belonging to a document from ChromaDB and BM25."""
        # Delete from ChromaDB
        self.collection.delete(where={"doc_id": doc_id})

        # Remove from local chunk cache
        to_delete = [cid for cid, chunk in self.all_chunks.items() if chunk.doc_id == doc_id]
        for cid in to_delete:
            del self.all_chunks[cid]

        self._rebuild_bm25_index()

    def hybrid_search(
        self,
        query: str,
        doc_ids: list[str] | None = None,
        top_k: int = 6,
        rrf_k: int = settings.RRF_K,
    ) -> list[tuple[DocumentChunk, float]]:
        """
        Executes Reciprocal Rank Fusion over Dense (ChromaDB) and Sparse (BM25) searches.
        RRF(d) = sum(1 / (k + rank_m(d)))
        """
        if not self.all_chunks:
            return []

        # 1. Dense Retrieval (ChromaDB)
        dense_ranks: dict[str, int] = {}
        try:
            query_embedding = embedding_service.get_query_embedding(query)
            where_filter = None
            if doc_ids and len(doc_ids) == 1:
                where_filter = {"doc_id": doc_ids[0]}
            elif doc_ids and len(doc_ids) > 1:
                where_filter = {"doc_id": {"$in": doc_ids}}

            dense_results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=min(top_k * 2, len(self.all_chunks)),
                where=where_filter,
            )

            retrieved_ids = dense_results.get("ids", [[]])[0]
            for rank_idx, chunk_id in enumerate(retrieved_ids):
                dense_ranks[chunk_id] = rank_idx + 1  # 1-indexed rank
        except Exception as e:
            print(f"Warning: Dense retrieval issue: {e}")

        # 2. Sparse Retrieval (BM25)
        sparse_ranks: dict[str, int] = {}
        if self.bm25_model and self.bm25_chunk_ids:
            query_tokens = self._tokenize(query)
            scores = self.bm25_model.get_scores(query_tokens)

            # Pair chunk IDs with their BM25 scores
            scored_candidates = []
            for cid, score in zip(self.bm25_chunk_ids, scores):
                chunk = self.all_chunks.get(cid)
                if chunk and (not doc_ids or chunk.doc_id in doc_ids):
                    scored_candidates.append((cid, float(score)))

            # Sort descending by BM25 score
            scored_candidates.sort(key=lambda x: x[1], reverse=True)
            for rank_idx, (cid, score) in enumerate(scored_candidates[: top_k * 2]):
                if score > 0.0:
                    sparse_ranks[cid] = rank_idx + 1

        # 3. Reciprocal Rank Fusion (RRF)
        all_candidate_ids = set(dense_ranks.keys()).union(set(sparse_ranks.keys()))
        rrf_scores: dict[str, float] = {}

        for cid in all_candidate_ids:
            score = 0.0
            if cid in dense_ranks:
                score += 1.0 / (rrf_k + dense_ranks[cid])
            if cid in sparse_ranks:
                score += 1.0 / (rrf_k + sparse_ranks[cid])
            rrf_scores[cid] = score

        # Sort candidate IDs by final RRF score descending
        sorted_cids = sorted(rrf_scores.items(), key=lambda item: item[1], reverse=True)

        results: list[tuple[DocumentChunk, float]] = []
        for cid, score in sorted_cids[:top_k]:
            if cid in self.all_chunks:
                results.append((self.all_chunks[cid], score))

        return results

    def get_total_chunks(self) -> int:
        return len(self.all_chunks)


hybrid_retriever = HybridRetrieverService()
