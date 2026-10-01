"""Weighted reciprocal-rank fusion for dense and BM25 candidates."""

from collections.abc import Sequence
from dataclasses import dataclass, field
import os
from typing import Any
from uuid import UUID

from dotenv import load_dotenv

from app.retrieval.models import RetrievalCandidate

load_dotenv()


@dataclass(frozen=True, slots=True)
class HybridRetrievalSettings:
    """Fusion weights, result count, and reciprocal-rank constant."""

    dense_weight: float = 0.5
    bm25_weight: float = 0.5
    top_k: int = 20
    rank_constant: int = 60

    def __post_init__(self) -> None:
        if self.dense_weight < 0 or self.bm25_weight < 0:
            raise ValueError("Hybrid retrieval weights must not be negative.")
        if self.dense_weight + self.bm25_weight <= 0:
            raise ValueError("At least one hybrid retrieval weight must be positive.")
        if self.top_k <= 0 or self.rank_constant <= 0:
            raise ValueError("Hybrid result count and rank constant must be positive.")

    @classmethod
    def from_environment(cls) -> "HybridRetrievalSettings":
        """Read hybrid fusion settings from environment variables."""
        return cls(
            dense_weight=float(os.getenv("HYBRID_DENSE_WEIGHT", "0.5")),
            bm25_weight=float(os.getenv("HYBRID_BM25_WEIGHT", "0.5")),
            top_k=int(os.getenv("HYBRID_TOP_K", "20")),
        )


@dataclass(slots=True)
class _FusionEntry:
    """Mutable accumulator for one deduplicated candidate."""

    document_id: UUID
    text: str
    dense_score: float | None = None
    bm25_score: float | None = None
    fusion_score: float = 0.0
    retrieval_methods: set[str] = field(default_factory=set)
    metadata: dict[str, Any] = field(default_factory=dict)


class HybridRetriever:
    """Merge independently retrieved candidates using weighted RRF scores."""

    def __init__(self, settings: HybridRetrievalSettings) -> None:
        self.settings = settings

    def fuse(
        self,
        dense_results: Sequence[RetrievalCandidate],
        bm25_results: Sequence[RetrievalCandidate],
        *,
        top_k: int | None = None,
    ) -> list[RetrievalCandidate]:
        """Deduplicate candidates while retaining source scores and retrieval methods."""
        result_limit = self.settings.top_k if top_k is None else top_k
        if result_limit <= 0:
            raise ValueError("top_k must be greater than zero.")
        fused: dict[UUID, _FusionEntry] = {}
        self._add_results(fused, dense_results, "dense", self.settings.dense_weight)
        self._add_results(fused, bm25_results, "bm25", self.settings.bm25_weight)

        candidates = [
            RetrievalCandidate(
                chunk_id=chunk_id,
                document_id=entry.document_id,
                text=entry.text,
                dense_score=entry.dense_score,
                bm25_score=entry.bm25_score,
                fusion_score=entry.fusion_score,
                retrieval_methods=tuple(sorted(entry.retrieval_methods)),
                metadata=entry.metadata,
            )
            for chunk_id, entry in fused.items()
        ]
        candidates.sort(key=lambda item: (-float(item.fusion_score or 0.0), str(item.chunk_id)))
        return candidates[:result_limit]

    def _add_results(
        self,
        fused: dict[UUID, _FusionEntry],
        results: Sequence[RetrievalCandidate],
        method: str,
        weight: float,
    ) -> None:
        if weight == 0:
            return
        for rank, candidate in enumerate(results, start=1):
            entry = fused.setdefault(
                candidate.chunk_id,
                _FusionEntry(document_id=candidate.document_id, text=candidate.text),
            )
            if entry.document_id != candidate.document_id:
                raise ValueError("A chunk ID cannot map to multiple document IDs.")
            entry.fusion_score += weight / (self.settings.rank_constant + rank)
            if method == "dense":
                entry.dense_score = candidate.dense_score
            else:
                entry.bm25_score = candidate.bm25_score
            entry.retrieval_methods.add(method)
            if candidate.text and not entry.text:
                entry.text = candidate.text
            entry.metadata.update(candidate.metadata)
