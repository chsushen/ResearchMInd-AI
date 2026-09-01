# 🔬 ResearchMind AI: Monetizable Deep Research & Hybrid RAG SaaS Platform

> **FAANG-Caliber Deep Research & Multimodal Hybrid RAG Copilot for Scientific Literature**  
> *Concurrent Dense/Sparse Hybrid Retrieval • Reciprocal Rank Fusion ($k=60$) • Claim-Level Citations `[Doc, p. X]` • Cross-Document Contradiction Engine • SaaS Rate Limiting • 1-Click CLI & Docker Compose*

[![Python Version](https://img.shields.io/badge/Python-3.10%2B-blue.svg?logo=python)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-009688.svg?style=flat&logo=FastAPI&logoColor=white)](https://fastapi.tiangolo.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.32+-FF4B4B.svg?style=flat&logo=Streamlit&logoColor=white)](https://streamlit.io)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-0.4.24-orange.svg?style=flat)](https://www.trychroma.com/)
[![Tests](https://img.shields.io/badge/Tests-21%20Passed-brightgreen.svg?logo=pytest)](tests/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

---

## 🏛️ System Architecture

```text
+-------------------------------------------------------------------------+
|                  1. STREAMING INGESTION & PARSING                       |
|   Academic PDF (100+ pages / Scanned) ---> pdf_parser.py (Page Slicing) |
+------------------------------------+------------------------------------+
                                     |
                 +-------------------+-------------------+
                 |                                       |
                 v                                       v
+--------------------------------+      +--------------------------------+
|  Dense Embeddings (Concurrent) |      |  Sparse Lexical (Concurrent)   |
|  Google text-embedding-004     |      |  rank-bm25 (BM25Okapi)         |
|  ChromaDB HNSW Vector Store    |      |  Inverted Token Index          |
+----------------+---------------+      +----------------+---------------+
                 |                                       |
                 +-------------------+-------------------+
                                     |
                                     v
+-------------------------------------------------------------------------+
|               2. PARALLEL HYBRID RETRIEVAL & RANK FUSION                |
|   User Query ---> ThreadPoolExecutor(max_workers=2)                     |
|   Formula: RRF(d) = SUM[ 1 / (60 + rank_m(d)) ]  ===> Reranked Passages |
+------------------------------------+------------------------------------+
                                     |
                                     v
+-------------------------------------------------------------------------+
|         3. DEEP RESEARCH & CROSS-DOCUMENT REASONING AGENT               |
|   - Claim Extraction with Exact Badges: [Doc Title, p. Y]               |
|   - Cross-Validation & Contradiction Detection between Manuscripts      |
|   - Comparative Matrix Synthesis & LaTeX BibTeX Citation Export         |
+------------------------------------+------------------------------------+
                                     |
                                     v
+-------------------------------------------------------------------------+
|               4. SAAS MONETIZATION & USAGE RATE LIMITER                 |
|   Token-Bucket Middleware: Free Tier (5 req/hr) vs Pro Tier (Unlimited) |
|   Endpoints: POST /api/query, POST /api/deep-research, GET /api/usage   |
+-------------------------------------------------------------------------+
```

---

## ⚡ SaaS Monetization & Pricing Tiers

ResearchMind AI incorporates an enterprise token-bucket rate limiter that identifies users by API Key, Session ID, or IP:

| Feature | Free Tier | Pro Tier (`X-Subscription-Tier: pro`) |
| :--- | :---: | :---: |
| **Query Quota** | 5 queries / session window | **Unlimited** |
| **Retrieval Engine** | Concurrent Dense + Sparse (RRF, $k=60$) | Concurrent Dense + Sparse (RRF, $k=60$) |
| **Claim-Level Badges** | `[Doc, p. X]` page-level badges | `[Doc, p. X]` page-level badges |
| **Deep Research Agent** | Cross-document claim validation | Full cross-document reasoning + contradiction engine |
| **LaTeX BibTeX Export** | Included | Included |
| **Rate-Limit Enforcement** | HTTP 429 with retry header | None (unrestricted throughput) |

---

## 💻 1-Click CLI Query Tool (`researchmind`)

ResearchMind AI is packaged with standard `pyproject.toml` exposing the `researchmind` console script.

### Installation
```bash
pip install -e .
```

### 1. Hybrid RAG Search
```bash
# Query literature with claim citations
researchmind search --query "attention mechanism and sparse transformers" --top-k 6

# JSON output for automated pipelines
researchmind search --query "Navier-Stokes physics neural networks" --format json
```

### 2. Deep Research Cross-Document Synthesis
```bash
researchmind deep-research --topic "Physics-Informed Neural Networks and Flow Fields"
```

### 3. Launch Local Server
```bash
researchmind server --host 0.0.0.0 --port 8000 --reload
```

---

## 🐳 1-Click Multi-Container Deployment (Docker Compose)

```bash
docker compose up --build -d
```

- **Operations & Search UI**: [http://localhost:8501](http://localhost:8501)
- **FastAPI OpenAPI Swagger**: [http://localhost:8000/docs](http://localhost:8000/docs)
- **ChromaDB Standalone Vector Database**: [http://localhost:8001](http://localhost:8001)
- **Usage Telemetry**: [http://localhost:8000/api/usage](http://localhost:8000/api/usage)

---

## 📐 Mathematical Formulation: Reciprocal Rank Fusion (RRF)

Standard reciprocal rank fusion fuses disjoint or overlapping ranking lists without requiring score normalization:

$$\text{RRF}(d \in D) = \sum_{m \in M} \frac{1}{k + \text{rank}_m(d)}$$

Where:
- $M = \{\text{dense\_chroma}, \text{sparse\_bm25}\}$
- $k = 60$ (standard smoothing factor)
- $\text{rank}_m(d)$ is the 1-indexed position of chunk $d$ in system $m$.

---

## 🧪 Comprehensive Test Suite (21 Tests Passing)

Execute the full automated test suite:

```bash
PYTHONPATH=. pytest tests/ -v
```

### Passing Test Suites:
- **`test_rrf_fusion.py`**: RRF mathematical formula ($k=60$), parallel dense/sparse search, ties, and edge cases.
- **`test_rate_limiter.py`**: Free Tier quota exhaustion (429 on 6th request), Pro Tier bypass, and `/api/usage` telemetry.
- **`test_deep_research.py`**: Autonomous claim extraction, contradiction detection, and comparative matrix synthesis.
- **`test_rag.py`**: PDF cleaning, page-boundary preservation, 100+ page documents, and strict grounding.
- **`test_api.py`**: End-to-end FastAPI endpoint contracts (`/api/health`, `/api/query`, `/api/compare`, `/api/documents`).

---

## 🔌 API Reference

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `POST` | `/api/query` | Hybrid RAG query with parallel Chroma/BM25, RRF, and page citations |
| `POST` | `/api/deep-research` | Cross-document reasoning, claim verification, and contradiction detection |
| `POST` | `/api/compare` | Structured multi-dimensional comparison matrix across manuscripts |
| `GET` | `/api/usage` | SaaS query quota, active tier, and token consumption metrics |
| `POST` | `/api/upload` | Ingests PDF research papers with page-level chunking |
| `POST` | `/api/extract-bibtex` | Generates standardized LaTeX BibTeX citation entry |
| `GET` | `/api/health` | Diagnostic health status and vector store chunk count |

---

## 📄 License

Apache License 2.0. Authored by Chunduri Sushen.
