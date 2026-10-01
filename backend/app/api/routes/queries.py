"""Grounded document question answering API."""

import logging
import os
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session, selectinload

from app.api.dependencies import get_database_session, get_redis_controls
from app.api.service_factory import create_rag_pipeline
from app.core.security import get_tenant_id
from app.models import Chunk, Query, RetrievalResult
from app.schemas.query import CitationRead, QueryHistoryRead, QueryRequest, QueryResponse
from app.services.redis_controls import RedisControls
from app.services.rag_pipeline import InvalidQuestionError

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api", tags=["queries"])


@router.get("/queries/{query_id}", response_model=QueryHistoryRead)
def get_query(
    query_id: UUID,
    session: Session = Depends(get_database_session),
    _tenant_id: UUID = Depends(get_tenant_id),
) -> QueryHistoryRead:
    """Return saved query text, answer, and its reranked source citations."""
    query = session.scalar(
        select(Query)
        .where(Query.id == query_id)
        .options(selectinload(Query.retrieval_results).selectinload(RetrievalResult.chunk))
    )
    if query is None:
        raise HTTPException(status_code=404, detail="Query was not found.")
    sources = []
    if query.citations is not None:
        for citation_data in query.citations:
            citation = CitationRead.model_validate(citation_data)
            if session.get(Chunk, citation.chunk_id) is not None:
                sources.append(citation)
    else:
        ranked_results = sorted(
            (item for item in query.retrieval_results if item.retrieval_method == "reranked"),
            key=lambda item: item.rank,
        )
        for result in ranked_results:
            chunk = result.chunk
            if chunk is None or chunk.document is None:
                continue
            sources.append(CitationRead(
                document_id=chunk.document_id,
                document_name=chunk.document.filename,
                page_number=chunk.page_number,
                section_title=chunk.section_title,
                chunk_id=chunk.id,
            ))
    return QueryHistoryRead(
        query_id=query.id,
        question=query.question,
        answer=query.answer,
        created_at=query.created_at,
        sources=sources,
    )


@router.post("/query", response_model=QueryResponse)
def query_documents(
    request: QueryRequest,
    session: Session = Depends(get_database_session),
    tenant_id: UUID = Depends(get_tenant_id),
    cache: RedisControls = Depends(get_redis_controls),
) -> QueryResponse:
    """Run retrieval, reranking, and grounded generation for a document question."""
    try:
        config_version = "|".join((
            os.getenv("API_CACHE_VERSION", "v1"),
            os.getenv("GEMINI_EMBEDDING_MODEL", "gemini-embedding-2"),
            os.getenv("GEMINI_MODEL", "gemini-3.8-flash"),
            os.getenv("EMBEDDING_DIMENSION", "1536"),
            os.getenv("DENSE_TOP_K", "20"),
            os.getenv("BM25_TOP_K", "20"),
            os.getenv("RERANK_MODEL", os.getenv("COHERE_RERANK_MODEL", "rerank-v4.0-pro")),
            os.getenv("HYBRID_TOP_K", "20"),
            os.getenv("HYBRID_DENSE_WEIGHT", "0.5"),
            os.getenv("HYBRID_BM25_WEIGHT", "0.5"),
            os.getenv("RERANK_CANDIDATE_K", "30"),
            os.getenv("RERANK_TOP_K", "8"),
            os.getenv("GENERATION_MAX_TOKENS", "2048"),
        ))
        cache_key = cache.cache_key(
            tenant_id=tenant_id,
            question=request.question,
            document_ids=request.document_ids or None,
            config_version=config_version,
        )
        cached = cache.get_json(cache_key)
        if isinstance(cached, dict):
            cached_response = QueryResponse.model_validate(cached)
            cached_response = cached_response.model_copy(update={
                "sources": [
                    citation for citation in cached_response.sources
                    if session.get(Chunk, citation.chunk_id) is not None
                ],
            })
            record = Query(
                question=request.question.strip(),
                answer=cached_response.answer,
                citations=[source.model_dump(mode="json") for source in cached_response.sources],
            )
            session.add(record)
            session.flush()
            for rank, citation in enumerate(cached_response.sources, start=1):
                session.add(RetrievalResult(
                    query_id=record.id,
                    chunk_id=citation.chunk_id,
                    retrieval_method="reranked",
                    rank=rank,
                ))
            session.commit()
            return cached_response.model_copy(update={"query_id": record.id})

        result = create_rag_pipeline(session, tenant_id).query(
            request.question,
            tenant_id=tenant_id,
            document_ids=request.document_ids or None,
        )
        session.commit()
    except InvalidQuestionError as exc:
        session.rollback()
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except SQLAlchemyError as exc:
        session.rollback()
        logger.exception("Query metadata persistence failed")
        raise HTTPException(status_code=503, detail="Query could not be saved.") from exc
    except Exception as exc:
        session.rollback()
        logger.exception("RAG query failed", extra={"tenant_id": str(tenant_id)})
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="The document question could not be answered right now.",
        ) from exc

    response = QueryResponse(
        query_id=result.query_id,
        answer=result.answer,
        sources=[
            CitationRead(
                document_id=citation.document_id,
                document_name=citation.document_name,
                page_number=citation.page_number,
                section_title=citation.section_title,
                chunk_id=citation.chunk_id,
            )
            for citation in result.sources
        ],
        metadata=result.metadata,
    )
    cache.set_json(cache_key, response.model_dump(mode="json"))
    return response
