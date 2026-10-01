"""Endpoint checks that avoid live external integrations."""

from types import SimpleNamespace
from uuid import uuid4

from app.api.routes import documents as document_routes
from app.api.routes import evaluations as evaluation_routes
from app.api.routes import queries as query_routes
from app.generation.citations import Citation
from app.models import Chunk, Document, DocumentVersion
from app.reranking.models import RerankOutcome, RerankedCandidate
from app.retrieval.models import RetrievalCandidate
from app.services.rag_pipeline import RAGPipelineMetadata, RAGQueryResult


def test_health_and_document_list(api_client):
    client, _ = api_client

    health = client.get("/health")
    listing = client.get("/api/documents")

    assert health.status_code == 200
    assert health.json() == {"status": "ok"}
    assert health.headers.get("x-request-id")
    assert listing.status_code == 200
    assert listing.json() == []


def test_query_endpoint_caches_response_and_keeps_history(api_client, monkeypatch):
    client, factory = api_client
    calls = 0
    chunk_id = uuid4()
    document_id = uuid4()
    version_id = uuid4()
    with factory() as session:
        document = Document(id=document_id, filename="contract.pdf", status="READY")
        version = DocumentVersion(
            id=version_id,
            document_id=document_id,
            version=1,
            content_hash="a" * 64,
            status="READY",
        )
        chunk = Chunk(
            id=chunk_id,
            document_id=document_id,
            document_version_id=version_id,
            chunk_index=0,
            text="Termination requires notice.",
            page_number=4,
            section_title="Termination",
            chunk_metadata={},
        )
        session.add_all([document, version, chunk])
        session.commit()

    class FakePipeline:
        def query(self, question, **_scope):
            nonlocal calls
            calls += 1
            return RAGQueryResult(
                query_id=uuid4(),
                answer="The contract allows termination after notice.",
                sources=(Citation(document_id, "contract.pdf", 4, "Termination", chunk_id),),
                metadata=RAGPipelineMetadata(
                    retrieval_count=1,
                    dense_count=1,
                    bm25_count=1,
                    hybrid_count=1,
                    reranked_count=1,
                    rerank_fallback_used=False,
                    rerank_latency_ms=3,
                    generation_latency_ms=8,
                    total_latency_ms=15,
                    model="test-model",
                    prompt_tokens=10,
                    output_tokens=8,
                ),
            )

    monkeypatch.setattr(query_routes, "create_rag_pipeline", lambda *_args: FakePipeline())
    request = {"question": "What are the termination terms?"}
    first = client.post("/api/query", json=request)
    second = client.post("/api/query", json=request)

    assert first.status_code == 200
    assert second.status_code == 200
    assert first.json()["answer"] == second.json()["answer"]
    assert first.json()["sources"][0]["page_number"] == 4
    assert first.json()["query_id"] != second.json()["query_id"]
    assert calls == 1
    history = client.get(f"/api/queries/{second.json()['query_id']}")
    assert history.status_code == 200
    assert history.json()["answer"] == first.json()["answer"]
    assert history.json()["sources"] == first.json()["sources"]


def test_upload_records_failure_when_integration_configuration_is_missing(
    api_client,
    monkeypatch,
    tmp_path,
):
    client, factory = api_client
    monkeypatch.setenv("DOCUMENT_STORAGE_PATH", str(tmp_path / "uploads"))
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setattr(document_routes, "_session_factory", lambda: factory)

    response = client.post(
        "/api/documents",
        files={"file": ("policy.txt", b"Termination requires thirty days' notice.", "text/plain")},
    )

    assert response.status_code == 201
    document_id = response.json()["id"]
    detail = client.get(f"/api/documents/{document_id}")
    assert detail.status_code == 200
    assert detail.json()["status"] == "FAILED"
    assert detail.json()["ingestion_jobs"][0]["status"] == "FAILED"


def test_production_requires_bearer_auth(api_client, monkeypatch):
    client, _ = api_client
    monkeypatch.setenv("APP_ENV", "production")
    monkeypatch.setenv("API_BEARER_TOKEN", "configured-secret")

    unauthorized = client.get("/api/documents")
    authorized = client.get(
        "/api/documents",
        headers={"Authorization": "Bearer configured-secret"},
    )

    assert unauthorized.status_code == 401
    assert authorized.status_code == 200


def test_document_delete_removes_metadata_and_local_source(api_client, monkeypatch, tmp_path):
    client, factory = api_client
    document_id = uuid4()
    version_id = uuid4()
    storage_root = tmp_path / "uploads"
    stored_path = storage_root / "documents" / str(document_id) / "source.txt"
    stored_path.parent.mkdir(parents=True)
    stored_path.write_text("document source", encoding="utf-8")
    monkeypatch.setenv("DOCUMENT_STORAGE_PATH", str(storage_root))
    monkeypatch.setenv("BM25_INDEX_PATH", str(tmp_path / "bm25"))
    monkeypatch.delenv("PINECONE_API_KEY", raising=False)
    monkeypatch.delenv("PINECONE_INDEX_NAME", raising=False)
    with factory() as session:
        document = Document(
            id=document_id,
            filename="contract.txt",
            status="READY",
            file_reference=f"documents/{document_id}/source.txt",
        )
        version = DocumentVersion(
            id=version_id,
            document_id=document_id,
            version=1,
            content_hash="b" * 64,
            status="READY",
        )
        session.add_all([document, version])
        session.commit()

    response = client.delete(f"/api/documents/{document_id}")

    assert response.status_code == 204
    assert not stored_path.exists()
    assert client.get(f"/api/documents/{document_id}").status_code == 404


def test_evaluation_endpoint_runs_and_persists_strategy_comparison(
    api_client,
    monkeypatch,
    tmp_path,
):
    client, _factory = api_client
    document_id = uuid4()
    chunk_id = uuid4()
    candidate = RetrievalCandidate(
        chunk_id=chunk_id,
        document_id=document_id,
        text="Section 7 requires thirty days' notice.",
        dense_score=0.91,
        bm25_score=2.0,
        fusion_score=0.5,
        retrieval_methods=("dense", "bm25"),
        metadata={"source_filename": "agreement.pdf", "page_number": 7},
    )

    class FakeDense:
        embedding_service = SimpleNamespace(settings=SimpleNamespace(model="test-embedding"))

        def retrieve(self, _question, **_scope):
            return [candidate]

    class FakeHybrid:
        settings = SimpleNamespace(dense_weight=0.5, bm25_weight=0.5, rank_constant=60)

        def fuse(self, dense, bm25, **_options):
            return dense or bm25

    class FakeBM25:
        def __init__(self, _tenant):
            pass

        def retrieve(self, _question, **_options):
            return [candidate]

    class FakeReranker:
        settings = SimpleNamespace(candidate_k=30, top_k=8, model="test-rerank")

        def rerank(self, _question, candidates):
            return RerankOutcome((RerankedCandidate(candidates[0], 1, 0.5, 0.95, 1),), False, 2)

    class FakeGenerator:
        settings = SimpleNamespace(model="test-generation")

        def generate(self, _question, _candidates):
            return SimpleNamespace(answer="Thirty days' notice is required.", model="test-generation")

    monkeypatch.setattr(
        evaluation_routes,
        "get_rag_components",
        lambda _tenant: (FakeDense(), FakeHybrid(), FakeReranker(), FakeGenerator()),
    )
    monkeypatch.setattr(evaluation_routes, "BM25Retriever", FakeBM25)
    monkeypatch.setenv("EVALUATION_RUNS_PATH", str(tmp_path / "evaluation-runs"))
    response = client.post("/api/evaluations/run", json={
        "name": "notice-period-check",
        "description": "Checks the labeled notice clause.",
        "questions": [{
            "question": "What notice period is required?",
            "ground_truth": "Thirty days' notice is required.",
            "relevant_chunk_ids": [str(chunk_id)],
            "expected_document": "agreement.pdf",
            "expected_page": 7,
            "expected_chunk": str(chunk_id),
        }],
        "retrieval_k": 1,
    })

    assert response.status_code == 201
    assert set(response.json()["strategies"]) == {"dense", "bm25", "hybrid", "hybrid_reranked"}
    assert response.json()["strategies"]["hybrid"]["metrics"]["recall_at_k"] == 1.0
    runs = client.get("/api/evaluations")
    assert runs.status_code == 200
    assert len(runs.json()) == 4
    detail = client.get(f"/api/evaluations/{runs.json()[0]['id']}")
    assert detail.status_code == 200
    assert detail.json()["results"][0]["metrics"]["expected_page_hit"] == 1.0
