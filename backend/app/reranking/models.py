"""Reranking output records with original and reranked positions."""

from dataclasses import dataclass

from app.retrieval.models import RetrievalCandidate


@dataclass(frozen=True, slots=True)
class RerankedCandidate:
    """One candidate with its retrieval rank and optional Cohere relevance score."""

    candidate: RetrievalCandidate
    original_rank: int
    original_score: float | None
    rerank_score: float | None
    final_rank: int


@dataclass(frozen=True, slots=True)
class RerankOutcome:
    """Reranked candidates plus transparent fallback and latency metadata."""

    candidates: tuple[RerankedCandidate, ...]
    used_fallback: bool
    latency_ms: int
