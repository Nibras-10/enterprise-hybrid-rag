"""Unit coverage for structure retention and external-provider adapters."""

from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import pytest

from app.chunking.config import ChunkingConfig
from app.chunking.models import ChunkDraft
from app.chunking.semantic_chunker import SemanticChunker
from app.generation.citations import CitationBuilder
from app.generation.gemini_generator import GeneratedAnswer, GenerationPayload
from app.generation.context import ContextSource
from app.ingestion.elements import DocumentElement
from app.ingestion.parser import UnstructuredDocumentParser
from app.ingestion.upload_validation import UploadValidationError, sanitize_filename, validate_file_content
from app.retrieval.bm25 import BM25Retriever
from app.retrieval.dense import DenseRetriever, DenseRetrievalSettings
from app.retrieval.pinecone_store import VectorMatch
from app.reranking.cohere_reranker import CohereReranker, RerankerSettings, RerankingError
from app.retrieval.models import RetrievalCandidate


class HighSimilarity:
    def similarity(self, _left, _right):
        return 0.95


def test_parser_normalization_preserves_page_section_and_table_identity():
    document_id = uuid4()

    class Metadata:
        def __init__(self, values):
            self.values = values

        def to_dict(self):
            return self.values

    raw = [
        SimpleNamespace(category="Title", text="Termination", metadata=Metadata({"page_number": 1})),
        SimpleNamespace(category="NarrativeText", text="Either party may terminate.", metadata=Metadata({"page_number": 1})),
        SimpleNamespace(category="Table", text="Payment schedule", metadata=Metadata({"page_number": 2, "text_as_html": "<table>Payment</table>"})),
    ]

    elements = UnstructuredDocumentParser._normalize(raw, document_id, "agreement.pdf")

    assert elements[0].text == "Termination"
    assert elements[1].text == "Either party may terminate."
    assert elements[1].section_title == "Termination"
    assert elements[1].page_number == 1
    assert elements[2].text == "<table>Payment</table>"
    assert elements[2].section_title == "Termination"
    assert elements[2].metadata["table_id"] == "table-0001"
    assert elements[2].metadata["source_filename"] == "agreement.pdf"


def test_semantic_chunker_keeps_heading_and_table_boundaries():
    document_id = uuid4()
    version_id = uuid4()
    elements = [
        DocumentElement(document_id, 1, "Heading 1", "Termination", "Termination"),
        DocumentElement(document_id, 1, "NarrativeText", "Either party may terminate with notice.", "Termination"),
        DocumentElement(document_id, 2, "Table", "Notice period | 30 days", "Termination", {"table_id": "table-0001"}),
        DocumentElement(document_id, 3, "NarrativeText", "Payment is due on receipt.", "Payment"),
    ]
    chunker = SemanticChunker(
        ChunkingConfig(target_tokens=20, min_tokens=1, max_tokens=30, similarity_threshold=0.7),
        HighSimilarity(),
    )

    chunks = chunker.chunk(elements, version_id)

    assert len(chunks) == 3
    assert chunks[0].text.startswith("Termination\n")
    assert chunks[0].section_title == "Termination"
    assert chunks[1].chunk_type == "Table"
    assert chunks[1].metadata["source_elements"][0]["table_id"] == "table-0001"
    assert chunks[1].page_number == 2
    assert chunks[2].section_title == "Payment"


def test_bm25_indexes_exact_terms_and_persists_incremental_document(tmp_path):
    tenant_id = uuid4()
    document_id = uuid4()
    version_id = uuid4()
    chunk = ChunkDraft(
        id=uuid4(),
        document_id=document_id,
        document_version_id=version_id,
        chunk_index=0,
        text="Section 12.4: payment of $125,000 is due 2025-03-31.",
        page_number=4,
        section_title="Payment",
        chunk_type="NarrativeText",
        token_count=13,
        metadata={},
    )
    retriever = BM25Retriever(tenant_id, tmp_path)
    retriever.index([chunk], "agreement.pdf")

    results = retriever.retrieve("section 12.4 $125,000 2025-03-31", top_k=5)
    reloaded = BM25Retriever(tenant_id, tmp_path).retrieve("12.4", top_k=5)

    assert [item.chunk_id for item in results] == [chunk.id]
    assert reloaded[0].metadata["source_filename"] == "agreement.pdf"
    retriever.remove_document(document_id)
    assert BM25Retriever(tenant_id, tmp_path).retrieve("12.4", top_k=5) == []


def test_dense_retriever_preserves_chunk_association_and_scope():
    tenant_id = uuid4()
    document_id = uuid4()
    chunk_id = uuid4()
    selected_document = uuid4()

    class Embeddings:
        def embed_query(self, _question):
            return [0.1, 0.2]

    class Store:
        def query(self, vector, **scope):
            assert vector == [0.1, 0.2]
            assert scope["tenant_id"] == tenant_id
            assert scope["document_ids"] == [selected_document]
            assert scope["top_k"] == 4
            return [VectorMatch(chunk_id, document_id, 0.91, "Relevant passage", {"page_number": 9})]

    retriever = DenseRetriever(Embeddings(), Store(), DenseRetrievalSettings(top_k=8))
    result = retriever.retrieve(
        "Tell me about termination",
        tenant_id=tenant_id,
        document_ids=[selected_document],
        top_k=4,
    )

    assert result[0].chunk_id == chunk_id
    assert result[0].dense_score == 0.91
    assert result[0].metadata["page_number"] == 9


def test_reranker_maps_scores_and_fallback_never_invents_scores():
    document_id = uuid4()
    first = RetrievalCandidate(uuid4(), document_id, "first", fusion_score=0.4)
    second = RetrievalCandidate(uuid4(), document_id, "second", fusion_score=0.3)

    class SuccessfulClient:
        def rerank(self, **_request):
            return SimpleNamespace(results=[
                SimpleNamespace(index=1, relevance_score=0.92),
                SimpleNamespace(index=0, relevance_score=0.61),
            ])

    reranker = CohereReranker(
        RerankerSettings("key", candidate_k=2, top_k=2),
        SuccessfulClient(),
    )
    ranked = reranker.rerank("query", [first, second])
    assert ranked.candidates[0].candidate.chunk_id == second.chunk_id
    assert ranked.candidates[0].rerank_score == 0.92

    class BrokenClient:
        def rerank(self, **_request):
            raise TimeoutError("service unavailable")

    fallback = CohereReranker(
        RerankerSettings("", candidate_k=2, top_k=2, fallback_enabled=True),
        BrokenClient(),
    ).rerank("query", [first, second])
    assert fallback.used_fallback
    assert all(item.rerank_score is None for item in fallback.candidates)
    with pytest.raises(RerankingError):
        CohereReranker(
            RerankerSettings("key", candidate_k=2, top_k=2),
            BrokenClient(),
        ).rerank("query", [first, second])


def test_citation_builder_whitelists_source_ids_and_deduplicates_chunks():
    document_id = uuid4()
    chunk_id = uuid4()
    valid = ContextSource("S1", document_id, "agreement.pdf", 6, "Notices", chunk_id, "Text")
    duplicate = ContextSource("S2", document_id, "agreement.pdf", 6, "Notices", chunk_id, "Text")
    answer = GeneratedAnswer(
        answer="The notice period is 30 days.",
        source_ids=("S1", "S3"),
        sources=(valid, duplicate),
        model="test",
        prompt_tokens=None,
        output_tokens=None,
        latency_ms=0,
    )

    citations = CitationBuilder.build(answer)

    assert len(citations) == 1
    assert citations[0].chunk_id == chunk_id
    assert GenerationPayload.model_validate_json('{"answer":"Answer","source_ids":["S1"]}').source_ids == ["S1"]
    with pytest.raises(ValueError):
        GenerationPayload.model_validate_json('{"answer":"","source_ids":[]}')


def test_upload_filename_and_content_validation(tmp_path):
    assert sanitize_filename("../../private/contract.txt") == "contract.txt"
    with pytest.raises(UploadValidationError):
        sanitize_filename("contract.exe")
    text_path = tmp_path / "contract.txt"
    text_path.write_bytes(b"valid UTF-8 document")
    validate_file_content(text_path, ".txt")
    text_path.write_bytes(b"\xff\x00bad")
    with pytest.raises(UploadValidationError):
        validate_file_content(text_path, ".txt")
