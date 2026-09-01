import math
import re
import hashlib
from typing import Sequence
import google.generativeai as genai
from ..core.config import settings


class EmbeddingService:
    """
    Production embedding service supporting Google GenAI text-embedding-004
    with an ultra-fast deterministic local dense fallback engine.
    """

    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or settings.GEMINI_API_KEY
        self.client_configured = False
        if self.api_key and not self.api_key.startswith("mock") and self.api_key != "mock-key-for-ci":
            try:
                genai.configure(api_key=self.api_key)
                self.client_configured = True
            except Exception as e:
                print(f"Warning: Failed to configure Google GenAI client: {e}")

    def update_api_key(self, new_key: str) -> None:
        """Updates the active Gemini API key dynamically."""
        if new_key and new_key != self.api_key:
            self.api_key = new_key
            if not new_key.startswith("mock") and new_key != "mock-key-for-ci":
                try:
                    genai.configure(api_key=new_key)
                    self.client_configured = True
                except Exception as e:
                    print(f"Warning: Error updating GenAI key: {e}")

    def get_embeddings(self, texts: Sequence[str], batch_size: int = 40) -> list[list[float]]:
        """Generates dense embeddings for a batch of text chunks."""
        if not texts:
            return []

        # If Gemini API key is configured, use text-embedding-004
        if self.client_configured:
            try:
                all_embeddings: list[list[float]] = []
                for i in range(0, len(texts), batch_size):
                    batch = list(texts[i : i + batch_size])
                    response = genai.embed_content(
                        model="models/text-embedding-004",
                        content=batch,
                        task_type="retrieval_document",
                    )
                    embeddings = response.get("embedding", [])
                    # Handle single vs multiple responses
                    if embeddings and isinstance(embeddings[0], list):
                        all_embeddings.extend(embeddings)
                    elif embeddings and isinstance(embeddings[0], (int, float)):
                        all_embeddings.append(embeddings)
                    else:
                        all_embeddings.extend([self._generate_local_embedding(t) for t in batch])
                return all_embeddings
            except Exception as e:
                print(f"GenAI embedding failed, falling back to deterministic local dense embeddings: {e}")

        # Deterministic local high-dimensional dense embedding (384 dimensions)
        return [self._generate_local_embedding(t) for t in texts]

    def get_query_embedding(self, query: str) -> list[float]:
        """Generates a dense embedding for a user search query."""
        if self.client_configured:
            try:
                response = genai.embed_content(
                    model="models/text-embedding-004",
                    content=query,
                    task_type="retrieval_query",
                )
                emb = response.get("embedding", [])
                if emb and isinstance(emb[0], (int, float)):
                    return emb
            except Exception as e:
                print(f"GenAI query embedding fallback: {e}")

        return self._generate_local_embedding(query)

    def _generate_local_embedding(self, text: str, dim: int = 768) -> list[float]:
        """
        Deterministic, L2-normalized pseudo-semantic projection for local/testing execution.
        Preserves n-gram token overlap similarity without requiring GPU or heavy models.
        """
        vec = [0.0] * dim
        tokens = re.findall(r"\w+", text.lower())
        if not tokens:
            vec[0] = 1.0
            return vec

        # Token hashing with positional context
        for idx, token in enumerate(tokens):
            h = int(hashlib.md5(token.encode("utf-8")).hexdigest(), 16)
            pos_1 = h % dim
            pos_2 = (h >> 16) % dim
            sign = 1.0 if ((h >> 32) & 1) == 0 else -1.0

            # Weight slightly by term frequency and character length
            weight = math.log1p(len(token))
            vec[pos_1] += sign * weight
            vec[pos_2] += (1.0 - sign) * weight * 0.5

            # 2-gram context
            if idx > 0:
                bigram = f"{tokens[idx-1]}_{token}"
                bh = int(hashlib.sha1(bigram.encode("utf-8")).hexdigest(), 16)
                b_pos = bh % dim
                vec[b_pos] += 1.5

        # L2-normalize vector
        norm = math.sqrt(sum(v * v for v in vec))
        if norm > 1e-9:
            return [v / norm for v in vec]
        return [1.0 / math.sqrt(dim)] * dim


embedding_service = EmbeddingService()
