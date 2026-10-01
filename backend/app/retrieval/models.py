"""Shared retrieval candidate record used by dense, BM25, and hybrid search."""

from dataclasses import dataclass, field
from typing import Any
from uuid import UUID


@dataclass(frozen=True, slots=True)
class RetrievalCandidate:
    """Chunk candidate with scores and source metadata from retrieval stages."""

    chunk_id: UUID
    document_id: UUID
    text: str
    dense_score: float | None = None
    bm25_score: float | None = None
    fusion_score: float | None = None
    retrieval_methods: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)
