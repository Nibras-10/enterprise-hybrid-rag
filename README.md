# Enterprise Hybrid RAG

An enterprise-oriented retrieval-augmented generation application for legal, financial, and other business documents. It combines structure-aware parsing, semantic chunking, dense and lexical search, rank fusion, reranking, grounded Gemini answers, and source citations.

Basic vector RAG can miss exact identifiers, clause numbers, dates, and figures. This implementation combines Gemini embeddings and Pinecone with persistent BM25 keyword search, then fuses both result sets and optionally reranks them with Cohere.

## Architecture

```text
Next.js + TypeScript
        ↓
FastAPI ─── request IDs, CORS allowlist, auth, rate limit, tenant-scoped cache
        ↓
Upload → Unstructured → LlamaIndex nodes → structure-aware semantic chunks
        ↓
Gemini Embedding 2 ──→ Pinecone dense search
        │
        └──────────────→ persistent tenant BM25
                              ↓
                      weighted rank fusion
                              ↓
                       Cohere reranking
                              ↓
                  grounded Gemini response
                              ↓
                   verified citations

Supabase PostgreSQL: documents, versions, chunks, queries, and evaluation records
Redis: query-response cache and distributed rate-limit counters
Langfuse: optional metadata-only RAG traces
Ragas: Gemini-backed faithfulness, relevance, context precision, and recall
```

LlamaIndex `TextNode` values preserve element metadata at the semantic chunking boundary. The chunker retains control over headings, section boundaries, tables, and legal/financial structure. Pinecone stores embeddings; PostgreSQL stores application metadata only.

## Technology

- Frontend: Next.js 15, React, TypeScript
- API and orchestration: FastAPI, SQLAlchemy, Alembic
- Parsing: Unstructured
- RAG document nodes: LlamaIndex Core
- Embeddings and generation: Gemini Embedding 2 and Gemini
- Vector search: Pinecone
- Lexical search: rank-bm25
- Reranking: Cohere Rerank
- Evaluation: Ragas and retrieval metrics
- Metadata database: Supabase PostgreSQL
- Cache and rate limiting: Redis, with a process-local development fallback
- Tracing: Langfuse, opt-in
- Deployment targets: Vercel and Render, without Docker

## Implemented capabilities

- Document upload validation checks extension and file structure, size, filename, UTF-8 for text files, and duplicate content. Upload schedules parsing, chunking, embedding, Pinecone and BM25 indexing, with lifecycle state and failure recording.
- Parsing retains headings, paragraphs, lists, tables, page metadata, and section context.
- Semantic chunking uses structure, semantic similarity, and configurable token limits; tables remain atomic.
- Gemini embeddings preserve chunk IDs and enforce configured vector dimensions.
- Pinecone namespaces and BM25 indexes are separated by configured tenant ID. Hybrid retrieval uses weighted reciprocal-rank fusion.
- Cohere failures are explicit. A configured fallback retains original ranking and never fabricates rerank scores.
- Gemini generation treats retrieved documents as untrusted data, grounds answers in retrieved evidence, and only returns citations from retrieved chunks.
- Evaluation runs compare dense, BM25, hybrid, and hybrid-plus-reranking strategies. Precision@K, Recall@K, and MRR are deterministic. Optional Ragas metrics use Gemini and can be expensive.
- Query results are cached with tenant, question, document scope, and model/config version in the key. Cache entries are invalidated after indexing or deletion.
- Langfuse tracing records stage names, counts, model, token usage, and timings without sending question or document text.
- API provides health, upload, list/detail/delete, query, query history, and evaluation run/list/detail routes.
- The frontend provides document management, chat with source cards, and evaluation upload/comparison/history views.

## Configuration

Copy `.env.example` to `.env` for local work and provide credentials only in the environment. Do not commit `.env` or service keys. Required integration keys are only needed for the corresponding workflows.

Important settings include:

```env
SUPABASE_DATABASE_URL=
GEMINI_API_KEY=
GEMINI_MODEL=gemini-3.8-flash
GEMINI_EMBEDDING_MODEL=gemini-embedding-2
EMBEDDING_DIMENSION=1536
PINECONE_API_KEY=
PINECONE_INDEX_NAME=
COHERE_API_KEY=
RERANK_MODEL=rerank-v4.0-pro
REDIS_URL=
LANGFUSE_ENABLED=false
LANGFUSE_PUBLIC_KEY=
LANGFUSE_SECRET_KEY=
LANGFUSE_HOST=
NEXT_PUBLIC_API_URL=http://localhost:8000
```

Chunk, retrieval, timeout, retry, cache, rate-limit, storage, and evaluation settings are listed in `.env.example`. In production, set `APP_ENV=production`, configure `API_BEARER_TOKEN`, and set `CORS_ALLOWED_ORIGINS` to the deployed frontend origin. The frontend accepts a bearer token at runtime for secured deployments; it keeps that value in memory and does not embed a service secret in the build.

## Local development

### Backend

From `backend/`, create a Python environment, install `requirements-dev.txt` for local checks (or `requirements.txt` for runtime), then run:

```text
uvicorn app.main:app --reload
```

Before using document or query features, set the needed managed-service credentials. Configure Supabase PostgreSQL and apply Alembic migrations from `backend/`:

```text
alembic upgrade head
```

### Frontend

From `frontend/`, install the locked dependencies and start the development server:

```text
npm ci
npm run dev
```

The frontend defaults to `http://localhost:8000`; set `NEXT_PUBLIC_API_URL` to change the backend origin.

### Evaluation datasets

The file `evaluation/datasets/contract_retrieval.example.json` illustrates the dataset format. Replace placeholder ground truths and add relevant chunk IDs from your actual corpus before running it. Empty relevance labels produce no retrieval precision/recall values. Evaluation results are written under `evaluation/runs/` and stored in PostgreSQL. Ragas metrics are optional and require `GEMINI_API_KEY`.

Do not report evaluation values until a real labeled dataset and the actual corpus have been used. No retrieval quality or production performance result is claimed by this repository.

## Deployment

`render.yaml` describes a non-containerized FastAPI service. It uses a Render persistent disk for uploaded files, BM25 data, and evaluation reports. Set its Supabase, Gemini, Pinecone, Cohere, Redis, Langfuse, and frontend-origin environment variables in Render. Enable Langfuse after setting its keys. Apply the database migrations before sending traffic.

`frontend/vercel.json` selects Vercel's Next.js build. Set `NEXT_PUBLIC_API_URL` in the Vercel project to the Render API origin. Configure the matching Vercel origin in Render's CORS allowlist.

Production rollout still requires managed credentials, database migrations, managed service setup, and deployment verification. The Render and Vercel manifests are deployment configuration, not evidence of a completed deployment.

## Build and verification status

Phases 0 and 1 previously passed local startup/build and SQLite/Alembic checks. The remaining implementation has been added in phases through API, frontend, security, evaluation, observability, caching, and deployment configuration. Endpoint, full pipeline, security, and retrieval-quality tests are intentionally deferred until the implementation pass is complete.

The following checks still require the final verification pass and/or configured credentials:

- Full backend and frontend builds/lint, unit and API tests, including security cases.
- Live Supabase migration and metadata persistence.
- Live document parsing, Gemini embeddings/generation, Pinecone search, and Cohere reranking.
- Redis shared-cache/rate-limit behavior and Langfuse trace delivery.
- Ragas evaluation and measured Dense/BM25/Hybrid/Reranked retrieval comparisons.
- Vercel and Render deployment.
