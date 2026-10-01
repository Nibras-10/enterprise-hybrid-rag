"""Application-level orchestration for retrieval, reranking, and generation."""

from collections.abc import Callable
from dataclasses import dataclass
import os
import re
from time import perf_counter
from uuid import UUID

from dotenv import load_dotenv
from sqlalchemy.orm import Session

from app.generation.citations import Citation, CitationBuilder
from app.generation.gemini_generator import GeneratedAnswer, GeminiAnswerGenerator
from app.reranking.cohere_reranker import CohereReranker
from app.reranking.models import RerankOutcome
from app.repositories.query_repository import QueryRepository, retrieval_records
from app.retrieval.bm25 import BM25Retriever
from app.retrieval.dense import DenseRetriever
from app.retrieval.hybrid import HybridRetriever
from app.retrieval.models import RetrievalCandidate
from app.observability.langfuse_tracer import LangfuseTracer

load_dotenv()

_MAX_QUESTION_LENGTH = int(os.getenv("MAX_QUESTION_LENGTH", "4000"))
_QUESTION_TOKEN_PATTERN = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


class InvalidQuestionError(ValueError):
    """Raised when a question is blank, too long, or contains invalid controls."""


class QueryProcessor:
    """Normalize and bound questions before creating query records or API calls."""

    @staticmethod
    def process(question: str) -> str:
        normalized = question.strip()
        if not normalized:
            raise InvalidQuestionError("Question must not be empty.")
        if len(normalized) > _MAX_QUESTION_LENGTH:
            raise InvalidQuestionError("Question exceeds the configured maximum length.")
        if _QUESTION_TOKEN_PATTERN.search(normalized):
            raise InvalidQuestionError("Question contains unsupported control characters.")
        return normalized


@dataclass(frozen=True, slots=True)
class RAGPipelineMetadata:
    """Public counts, latency, and fallback state for one RAG query."""

    retrieval_count: int
    dense_count: int
    bm25_count: int
    hybrid_count: int
    reranked_count: int
    rerank_fallback_used: bool
    rerank_latency_ms: int
    generation_latency_ms: int
    total_latency_ms: int
    model: str
    prompt_tokens: int | None
    output_tokens: int | None


@dataclass(frozen=True, slots=True)
class RAGQueryResult:
    """Answer, supported citations, and application-level query metadata."""

    query_id: UUID
    answer: str
    sources: tuple[Citation, ...]
    metadata: RAGPipelineMetadata


class RAGPipeline:
    """Run the configured dense + BM25 hybrid RAG pipeline for one tenant scope."""

    def __init__(
        self,
        *,
        session: Session,
        dense_retriever: DenseRetriever,
        hybrid_retriever: HybridRetriever,
        reranker: CohereReranker,
        generator: GeminiAnswerGenerator,
        bm25_retriever_factory: Callable[[UUID], BM25Retriever] | None = None,
        tracer: LangfuseTracer | None = None,
    ) -> None:
        self.session = session
        self.dense_retriever = dense_retriever
        self.hybrid_retriever = hybrid_retriever
        self.reranker = reranker
        self.generator = generator
        self.bm25_retriever_factory = bm25_retriever_factory or BM25Retriever
        self.bm25_top_k = int(os.getenv("BM25_TOP_K", "20"))
        if self.bm25_top_k <= 0:
            raise ValueError("BM25_TOP_K must be greater than zero.")
        self.citation_builder = CitationBuilder()
        self.query_repository = QueryRepository(session)
        self.tracer = tracer or LangfuseTracer()

    def query(
        self,
        question: str,
        *,
        tenant_id: UUID,
        document_ids: list[UUID] | None = None,
        user_id: UUID | None = None,
        collection_id: UUID | None = None,
    ) -> RAGQueryResult:
        """Persist a query, retrieve and fuse candidates, rerank, and generate citations."""
        with self.tracer.span("rag.query", metadata={"question_length": len(question)}) as span:
            try:
                result = self._query_impl(
                    question,
                    tenant_id=tenant_id,
                    document_ids=document_ids,
                    user_id=user_id,
                    collection_id=collection_id,
                )
            except Exception as exc:
                if span is not None:
                    span.update(level="ERROR", status_message=type(exc).__name__)
                raise
            if span is not None:
                span.update(output={
                    "retrieval_count": result.metadata.retrieval_count,
                    "reranked_count": result.metadata.reranked_count,
                    "latency_ms": result.metadata.total_latency_ms,
                    "model": result.metadata.model,
                })
            return result

    def _query_impl(
        self,
        question: str,
        *,
        tenant_id: UUID,
        document_ids: list[UUID] | None = None,
        user_id: UUID | None = None,
        collection_id: UUID | None = None,
    ) -> RAGQueryResult:
        normalized_question = QueryProcessor.process(question)
        started_at = perf_counter()
        query_record = self.query_repository.create(normalized_question, user_id)

        with self.tracer.span("rag.dense_retrieval") as stage:
            dense_results = self.dense_retriever.retrieve(
                normalized_question,
                tenant_id=tenant_id,
                document_ids=document_ids,
                collection_id=collection_id,
            )
            if stage is not None:
                stage.update(output={"result_count": len(dense_results)})
        with self.tracer.span("rag.bm25_retrieval") as stage:
            bm25_results = self.bm25_retriever_factory(tenant_id).retrieve(
                normalized_question,
                top_k=self.bm25_top_k,
                document_ids=document_ids,
            )
            if stage is not None:
                stage.update(output={"result_count": len(bm25_results)})
        with self.tracer.span("rag.hybrid_fusion") as stage:
            hybrid_results = self.hybrid_retriever.fuse(
                dense_results,
                bm25_results,
                top_k=self.reranker.settings.candidate_k,
            )
            if stage is not None:
                stage.update(output={"result_count": len(hybrid_results)})
        with self.tracer.span("rag.reranking", metadata={"candidate_count": len(hybrid_results)}) as stage:
            rerank_outcome: RerankOutcome = self.reranker.rerank(
                normalized_question,
                hybrid_results,
            )
            if stage is not None:
                stage.update(output={
                    "result_count": len(rerank_outcome.candidates),
                    "fallback_used": rerank_outcome.used_fallback,
                    "latency_ms": rerank_outcome.latency_ms,
                })

        generation_started_at = perf_counter()
        with self.tracer.span("rag.generation", metadata={"candidate_count": len(rerank_outcome.candidates)}) as stage:
            generated: GeneratedAnswer = self.generator.generate(
                normalized_question,
                list(rerank_outcome.candidates),
            )
            if stage is not None:
                stage.update(output={
                    "model": generated.model,
                    "latency_ms": generated.latency_ms,
                    "prompt_tokens": generated.prompt_tokens,
                    "output_tokens": generated.output_tokens,
                })
        generation_latency_ms = int((perf_counter() - generation_started_at) * 1000)
        with self.tracer.span("rag.citation_build") as stage:
            citations = self.citation_builder.build(generated)
            if stage is not None:
                stage.update(output={"source_count": len(citations)})

        self.query_repository.set_answer(query_record.id, generated.answer)
        self.query_repository.set_citations(query_record.id, [
            {
                "document_id": str(citation.document_id),
                "document_name": citation.document_name,
                "page_number": citation.page_number,
                "section_title": citation.section_title,
                "chunk_id": str(citation.chunk_id),
            }
            for citation in citations
        ])
        self.query_repository.save_retrieval_results(
            query_record.id,
            retrieval_records(
                dense_results,
                bm25_results,
                hybrid_results,
                list(rerank_outcome.candidates),
            ),
        )
        metadata = RAGPipelineMetadata(
            retrieval_count=len(hybrid_results),
            dense_count=len(dense_results),
            bm25_count=len(bm25_results),
            hybrid_count=len(hybrid_results),
            reranked_count=len(rerank_outcome.candidates),
            rerank_fallback_used=rerank_outcome.used_fallback,
            rerank_latency_ms=rerank_outcome.latency_ms,
            generation_latency_ms=generation_latency_ms,
            total_latency_ms=int((perf_counter() - started_at) * 1000),
            model=generated.model,
            prompt_tokens=generated.prompt_tokens,
            output_tokens=generated.output_tokens,
        )
        return RAGQueryResult(
            query_id=query_record.id,
            answer=generated.answer,
            sources=tuple(citations),
            metadata=metadata,
        )
