"""Cohere reranking adapter with explicit, score-free failure fallback."""

from collections.abc import Sequence
from dataclasses import dataclass
import math
import os
import time
from typing import Any

from dotenv import load_dotenv

from app.reranking.models import RerankOutcome, RerankedCandidate
from app.retrieval.models import RetrievalCandidate

load_dotenv()


class RerankingError(RuntimeError):
    """Raised when reranking fails and no configured fallback is available."""


@dataclass(frozen=True, slots=True)
class RerankerSettings:
    """Cohere model, candidate limits, timeout, and fallback behavior."""

    api_key: str
    model: str = "rerank-v4.0-pro"
    candidate_k: int = 30
    top_k: int = 8
    timeout_seconds: float = 30.0
    fallback_enabled: bool = False

    def __post_init__(self) -> None:
        if not self.api_key and not self.fallback_enabled:
            raise ValueError("COHERE_API_KEY must be configured before reranking.")
        if not self.model.strip():
            raise ValueError("RERANK_MODEL must not be empty.")
        if self.candidate_k <= 0 or self.top_k <= 0 or self.top_k > self.candidate_k:
            raise ValueError("Reranker candidate/result limits are invalid.")
        if self.timeout_seconds <= 0:
            raise ValueError("Reranker timeout must be greater than zero.")

    @classmethod
    def from_environment(cls) -> "RerankerSettings":
        """Read reranker settings and API key from environment variables."""
        return cls(
            api_key=os.getenv("COHERE_API_KEY", ""),
            model=os.getenv("RERANK_MODEL") or os.getenv("COHERE_RERANK_MODEL") or "rerank-v4.0-pro",
            candidate_k=int(os.getenv("RERANK_CANDIDATE_K", "30")),
            top_k=int(os.getenv("RERANK_TOP_K", "8")),
            timeout_seconds=float(os.getenv("RERANK_TIMEOUT_SECONDS", "30")),
            fallback_enabled=os.getenv("RERANK_FALLBACK_ENABLED", "false").casefold() == "true",
        )


class CohereReranker:
    """Rerank a hybrid candidate list with Cohere and retain its source scores."""

    def __init__(self, settings: RerankerSettings, client: Any | None = None) -> None:
        self.settings = settings
        self._client = client

    def rerank(self, question: str, candidates: Sequence[RetrievalCandidate]) -> RerankOutcome:
        """Return top candidates in Cohere rank order or an explicitly marked fallback."""
        selected = list(candidates[: self.settings.candidate_k])
        if not selected:
            return RerankOutcome(candidates=(), used_fallback=False, latency_ms=0)

        started_at = time.perf_counter()
        try:
            response = self._get_client().rerank(
                model=self.settings.model,
                query=question,
                documents=[candidate.text for candidate in selected],
                top_n=min(self.settings.top_k, len(selected)),
            )
            reranked = self._map_results(response, selected)
            latency_ms = int((time.perf_counter() - started_at) * 1000)
            return RerankOutcome(
                candidates=tuple(reranked),
                used_fallback=False,
                latency_ms=latency_ms,
            )
        except RerankingError:
            raise
        except Exception as exc:
            if not self.settings.fallback_enabled:
                raise RerankingError("Cohere reranking failed and fallback is disabled.") from exc
            latency_ms = int((time.perf_counter() - started_at) * 1000)
            fallback = self._fallback(selected)
            return RerankOutcome(
                candidates=tuple(fallback),
                used_fallback=True,
                latency_ms=latency_ms,
            )

    def _get_client(self) -> Any:
        if self._client is None:
            if not self.settings.api_key:
                raise RuntimeError("COHERE_API_KEY is not configured.")
            import cohere

            self._client = cohere.ClientV2(
                api_key=self.settings.api_key,
                timeout=self.settings.timeout_seconds,
            )
        return self._client

    def _map_results(
        self,
        response: Any,
        candidates: Sequence[RetrievalCandidate],
    ) -> list[RerankedCandidate]:
        results = getattr(response, "results", None)
        if results is None:
            raise RerankingError("Cohere returned an invalid rerank response.")

        mapped: list[RerankedCandidate] = []
        seen_indexes: set[int] = set()
        for final_rank, result in enumerate(results[: self.settings.top_k], start=1):
            index = getattr(result, "index", None)
            score = getattr(result, "relevance_score", None)
            if not isinstance(index, int) or not 0 <= index < len(candidates) or index in seen_indexes:
                raise RerankingError("Cohere returned an invalid candidate index.")
            if not isinstance(score, (int, float)) or not math.isfinite(float(score)):
                raise RerankingError("Cohere returned a result without a relevance score.")
            seen_indexes.add(index)
            candidate = candidates[index]
            mapped.append(
                RerankedCandidate(
                    candidate=candidate,
                    original_rank=index + 1,
                    original_score=candidate.fusion_score,
                    rerank_score=float(score),
                    final_rank=final_rank,
                )
            )
        return mapped

    def _fallback(self, candidates: Sequence[RetrievalCandidate]) -> list[RerankedCandidate]:
        return [
            RerankedCandidate(
                candidate=candidate,
                original_rank=rank,
                original_score=candidate.fusion_score,
                rerank_score=None,
                final_rank=rank,
            )
            for rank, candidate in enumerate(candidates[: self.settings.top_k], start=1)
        ]
