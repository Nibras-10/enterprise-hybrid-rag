# Enterprise Hybrid RAG

[![Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115%2B-009688.svg)](https://fastapi.tiangolo.com/)
[![Next.js](https://img.shields.io/badge/Next.js-15-black.svg)](https://nextjs.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An enterprise-grade Retrieval-Augmented Generation (RAG) system engineered for complex legal, financial, and regulatory documents. Combines **structure-aware parsing**, **semantic chunking**, **Gemini dense embeddings**, **persistent tenant-isolated BM25 keyword search**, **reciprocal rank fusion**, and **Cohere reranking** to deliver factually grounded answers with verified citations.

---

## Why Hybrid RAG?

Standard vector-only RAG often fails on enterprise business documents:
- **Identifier blindness:** Dense embeddings struggle with exact contract numbers (e.g., `MSA-2026-X94`), statutory references, financial line items, and specific dates.
- **Context fragmentation:** Naive character-count splitters break tables, legal clauses, and section hierarchies.
- **Hallucinated citations:** Ungrounded generation quotes text that does not appear in retrieved chunks.

**Enterprise Hybrid RAG solves this:**
1. **Lexical + Dense Fusion:** BM25 retrieves exact clause numbers and terms; Gemini embeddings retrieve semantic meaning.
2. **Reciprocal Rank Fusion & Reranking:** Merges both candidate sets and scores them with Cohere Rerank before sending to Gemini.
3. **Atomic Structures:** Tables and section hierarchies remain intact throughout chunking.
4. **Strict Grounding:** Gemini answers are constrained strictly to retrieved context, with source citations linked to chunk IDs and page numbers.

---

## Architecture

```text
Next.js 15 UI (TypeScript)
       │
       ▼
FastAPI Gateway ─── Auth, Rate Limiter (Redis), CORS, Request IDs
       │
       ▼
Upload & Validation (PDF, DOCX, TXT)
       │
       ▼
Unstructured Parsing + LlamaIndex TextNodes
       │
       ▼
Structure-Aware Semantic Chunking (Preserves tables & headings)
       │
       ├─────────────────────────────────┐
       ▼                                 ▼
Gemini Embedding 2               Persistent BM25 Index
(Pinecone Dense Search)          (Tenant-Scoped Keyword Search)
       │                                 │
       └───────────────┬─────────────────┘
                       ▼
          Reciprocal Rank Fusion (RRF)
                       ▼
              Cohere Rerank v4
                       ▼
          Grounded Gemini Generation
                       ▼
           Verified Source Citations
```

- **Metadata & Logs:** Supabase PostgreSQL (Documents, Chunks, Queries, Evaluations)
- **Caching & Rate Limiting:** Redis
- **Evaluation:** Ragas & Deterministic Retrieval Metrics (Precision@K, Recall@K, MRR)
- **Observability:** Optional metadata-only tracing via Langfuse

---

## Tech Stack

| Component | Technology |
|---|---|
| **Frontend** | Next.js 15, React 19, TypeScript |
| **API & Pipeline** | FastAPI, SQLAlchemy 2.0, Alembic, Uvicorn |
| **Document Parsing** | Unstructured, `unstructured-inference`, `python-docx` |
| **Document Framework** | LlamaIndex Core |
| **Embeddings & LLM** | Google Gemini (`gemini-embedding-2`, `gemini-3.5-flash-lite` / `gemini-3.8-flash`) |
| **Vector Database** | Pinecone |
| **Lexical Search** | `rank-bm25` (tenant-isolated persistent indexes) |
| **Reranker** | Cohere Rerank (`rerank-v4.0-pro`) |
| **Database** | Supabase PostgreSQL |
| **Cache & Throttling** | Redis |
| **Evaluation** | Ragas, HuggingFace Datasets |
| **Deployment** | Vercel (Frontend), Render (Backend with persistent disk) |

---

## Getting Started

### Prerequisites

- **Python 3.11** or **3.12**
- **Node.js 18+** and **npm**
- Accounts and API keys for:
  - [Google AI Studio](https://aistudio.google.com/) (Gemini)
  - [Pinecone](https://www.pinecone.io/)
  - [Cohere](https://cohere.com/)
  - [Supabase](https://supabase.com/) (PostgreSQL)
  - [Redis](https://upstash.com/ or Render Redis) *(optional in local dev, local memory fallback supported)*

---

### 1. Clone the Repository

```bash
git clone https://github.com/<your-username>/enterprise-hybrid-rag.git
cd enterprise-hybrid-rag
```

---

### 2. Configure Environment Variables

Create `.env` inside `backend/` (or copy from `.env.example`):

```bash
cp .env.example backend/.env
```

Fill in your service credentials in `backend/.env`:

```env
# Database (Supabase PostgreSQL)
SUPABASE_DATABASE_URL=postgresql://user:password@host:5432/postgres

# Google Gemini
GEMINI_API_KEY=your_gemini_api_key
GEMINI_MODEL=gemini-3.5-flash-lite
GEMINI_EMBEDDING_MODEL=gemini-embedding-2
EMBEDDING_DIMENSION=1536

# Pinecone
PINECONE_API_KEY=your_pinecone_api_key
PINECONE_INDEX_NAME=index-rag
PINECONE_METRIC=cosine

# Cohere Rerank
COHERE_API_KEY=your_cohere_api_key
RERANK_MODEL=rerank-v4.0-pro

# Redis (Caching & Rate Limiting)
REDIS_URL=redis://localhost:6379

# Storage & Security
APP_ENV=development
CORS_ALLOWED_ORIGINS=http://localhost:3000
DOCUMENT_STORAGE_PATH=./data/uploads
BM25_INDEX_PATH=./data/bm25
```

---

### 3. Backend Setup

Open a terminal and set up the Python environment:

```bash
cd backend

# Create virtual environment
python -m venv .venv

# Activate virtual environment
# Windows (PowerShell):
.\.venv\Scripts\Activate.ps1
# macOS / Linux:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run database migrations
alembic upgrade head

# Start the API server
uvicorn app.main:app --reload
```

Backend will be available at: **`http://localhost:8000`**  
Interactive API Docs (Swagger): **`http://localhost:8000/docs`**

---

### 4. Frontend Setup

Open a second terminal:

```bash
cd frontend

# Install dependencies
npm install

# Start Next.js development server
npm run dev
```

Frontend will be available at: **`http://localhost:3000`**

---

## Testing with Sample Documents

Sample enterprise agreements are provided under [`sample_documents/`](./sample_documents/):
- [`sample_documents/enterprise_service_level_agreement.docx`](./sample_documents/enterprise_service_level_agreement.docx)
- [`sample_documents/enterprise_service_level_agreement.txt`](./sample_documents/enterprise_service_level_agreement.txt)

### Steps to test:
1. Open the UI at `http://localhost:3000`.
2. Upload either `enterprise_service_level_agreement.docx` or `.txt`.
3. Wait for the status indicator to progress: `UPLOADED` $\rightarrow$ `PARSED` $\rightarrow$ `CHUNKED` $\rightarrow$ `EMBEDDED` $\rightarrow$ `INDEXED`.
4. Switch to the **Chat** tab and ask grounded questions:
   - *"What is the aggregate liability cap under the agreement?"*
   - *"What is the required uptime SLA percentage and the associated service credit schedule?"*
   - *"Are customer data or query logs allowed to be used to train AI models?"*
   - *"What are the invoice payment terms and late interest rates?"*

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Health check endpoint |
| `POST` | `/api/documents` | Upload and trigger asynchronous document ingestion pipeline |
| `GET` | `/api/documents` | List uploaded documents with status and metadata |
| `GET` | `/api/documents/{id}` | Retrieve document details and chunk breakdown |
| `DELETE` | `/api/documents/{id}` | Delete document, vectors, and BM25 index entries |
| `POST` | `/api/query` | Execute hybrid search, rerank, and generate grounded answer |
| `GET` | `/api/queries/{id}` | Retrieve past query result with verified citations |
| `POST` | `/api/evaluations/run` | Run retrieval and Ragas evaluations against benchmark dataset |

---

## Deployment Guide

### Backend: Render

1. Connect your repository to [Render](https://render.com/).
2. Select **Blueprint** to deploy using [`render.yaml`](./render.yaml), or create a **Web Service**:
   - **Root Directory:** `backend`
   - **Environment:** `Python 3`
   - **Build Command:** `pip install -r requirements.txt && alembic upgrade head`
   - **Start Command:** `uvicorn app.main:app --host 0.0.0.0 --port $PORT`
   - **Disk:** Mount a 10 GB persistent disk at `/var/data`
3. Add Environment Variables in Render:
   - `PYTHON_VERSION`: `3.11.11`
   - `APP_ENV`: `production`
   - `SUPABASE_DATABASE_URL`, `GEMINI_API_KEY`, `PINECONE_API_KEY`, `COHERE_API_KEY`, `REDIS_URL`
   - `DOCUMENT_STORAGE_PATH`: `/var/data/uploads`
   - `BM25_INDEX_PATH`: `/var/data/bm25`
   - `API_BEARER_TOKEN`: `<your-secure-secret-token>`
   - `CORS_ALLOWED_ORIGINS`: `https://your-frontend.vercel.app`

### Frontend: Vercel

1. Import the repository into [Vercel](https://vercel.com/).
2. Set **Root Directory** to `frontend`.
3. Add Environment Variable:
   - `NEXT_PUBLIC_API_URL`: `https://your-backend-api.onrender.com`
4. Deploy. Copy your Vercel domain and set it in Render's `CORS_ALLOWED_ORIGINS`.

> **Secured Deployments:** In production, the UI requires your `API_BEARER_TOKEN`. Enter it in the **"Access token"** field at the bottom of the left sidebar to unlock access.

---

## Evaluation & Metrics

The repository includes a benchmark evaluation framework comparing:
1. **Dense Retrieval Only** (Gemini embeddings + Pinecone)
2. **Lexical Retrieval Only** (BM25)
3. **Hybrid Search** (Reciprocal Rank Fusion)
4. **Hybrid Search + Cohere Reranking**

Calculates deterministic metrics (**Precision@K**, **Recall@K**, **MRR**) and optional **Ragas** metrics (**Faithfulness**, **Answer Relevance**, **Context Precision**).

Sample benchmark dataset format: [`evaluation/datasets/contract_retrieval.example.json`](./evaluation/datasets/contract_retrieval.example.json).

---

## License

This project is licensed under the [MIT License](LICENSE).
