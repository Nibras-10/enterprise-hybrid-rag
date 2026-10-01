"""Persistence helpers for query records and ranked retrieval metadata."""

from dataclasses import dataclass
from collections.abc import Sequence
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models import Query, RetrievalResult
from app.reranking.models import RerankedCandidate
from app.retrieval.models import RetrievalCandidate


@dataclass(frozen=True, slots=True)
class RetrievalRecord:
    """One stage-specific rank and score to persist for a query."""

    chunk_id: UUID
    retrieval_method: str
    rank: int
    score: float | None
    rerank_score: float | None = None


class QueryRepository:
    """Store query history and retrieval candidates using SQLAlchemy parameters."""

    def __init__(self, session: Session) -> None:
        self.session = session

    def create(self, question: str, user_id: UUID | None = None) -> Query:
        query = Query(question=question, user_id=user_id)
        self.session.add(query)
        self.session.flush()
        return query

    def set_answer(self, query_id: UUID, answer: str) -> None:
        """Persist the generated answer inside the current query transaction."""
        query = self.session.get(Query, query_id)
        if query is None:
            raise ValueError("Query record does not exist.")
        query.answer = answer

    def set_citations(self, query_id: UUID, citations: list[dict[str, object]]) -> None:
        """Persist the verified citations returned with the generated answer."""
        query = self.session.get(Query, query_id)
        if query is None:
            raise ValueError("Query record does not exist.")
        query.citations = citations

    def save_retrieval_results(
        self,
        query_id: UUID,
        records: list[RetrievalRecord],
    ) -> None:
        """Persist stage-specific retrieval ranks and scores for one query."""
        self.session.add_all(
            [
                RetrievalResult(
                    query_id=query_id,
                    chunk_id=record.chunk_id,
                    retrieval_method=record.retrieval_method,
                    rank=record.rank,
                    score=record.score,
                    rerank_score=record.rerank_score,
                )
                for record in records
            ]
        )
        self.session.flush()

    def get(self, query_id: UUID) -> Query | None:
        """Fetch one saved query with ranked source records."""
        return self.session.scalar(select(Query).where(Query.id == query_id))


def retrieval_records(
    dense_results: list[RetrievalCandidate],
    bm25_results: list[RetrievalCandidate],
    hybrid_results: list[RetrievalCandidate],
    reranked_results: Sequence[RerankedCandidate],
) -> list[RetrievalRecord]:
    """Convert pipeline stages to rows while retaining scores at each stage."""
    records: list[RetrievalRecord] = []
    for method, candidates, score_field in (
        ("dense", dense_results, "dense_score"),
        ("bm25", bm25_results, "bm25_score"),
        ("hybrid", hybrid_results, "fusion_score"),
    ):
        records.extend(
            RetrievalRecord(
                chunk_id=candidate.chunk_id,
                retrieval_method=method,
                rank=rank,
                score=getattr(candidate, score_field),
            )
            for rank, candidate in enumerate(candidates, start=1)
        )
    records.extend(
        RetrievalRecord(
            chunk_id=result.candidate.chunk_id,
            retrieval_method="reranked",
            rank=result.final_rank,
            score=result.original_score,
            rerank_score=result.rerank_score,
        )
        for result in reranked_results
    )
    return records
