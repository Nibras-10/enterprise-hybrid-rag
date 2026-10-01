"""Validated query API input and safe public response shapes."""

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class QueryRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question: str = Field(min_length=1, max_length=4000)
    document_ids: list[UUID] = Field(default_factory=list, max_length=100)


class CitationRead(BaseModel):
    document_id: UUID
    document_name: str
    page_number: int | None
    section_title: str | None
    chunk_id: UUID


class QueryMetadataRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

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


class QueryResponse(BaseModel):
    query_id: UUID
    answer: str
    sources: list[CitationRead]
    metadata: QueryMetadataRead


class QueryHistoryRead(BaseModel):
    query_id: UUID
    question: str
    answer: str | None
    created_at: datetime
    sources: list[CitationRead]
