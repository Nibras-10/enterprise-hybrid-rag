"""Gemini Embedding 2 adapter with consistent dimensions and bounded retries."""

from collections.abc import Sequence
from dataclasses import dataclass
import math
import os
import time
from typing import Any
from uuid import UUID, uuid4

from dotenv import load_dotenv

from app.embeddings.models import EmbeddingInput, EmbeddingResult

load_dotenv()


class EmbeddingError(RuntimeError):
    """Raised when Gemini cannot return a complete, dimensionally valid batch."""


@dataclass(frozen=True, slots=True)
class EmbeddingSettings:
    """Gemini embedding model, dimension, batching, and retry settings."""

    api_key: str
    model: str = "gemini-embedding-2"
    dimension: int = 1536
    batch_size: int = 100
    max_retries: int = 3
    timeout_seconds: float = 30.0

    def __post_init__(self) -> None:
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY must be configured before generating embeddings.")
        if not self.model.strip():
            raise ValueError("GEMINI_EMBEDDING_MODEL must not be empty.")
        if not 128 <= self.dimension <= 3072:
            raise ValueError("EMBEDDING_DIMENSION must be between 128 and 3072.")
        if self.batch_size <= 0 or self.max_retries < 0 or self.timeout_seconds <= 0:
            raise ValueError("Embedding batch, retry, and timeout settings are invalid.")

    @classmethod
    def from_environment(cls) -> "EmbeddingSettings":
        """Load the embedding integration configuration from environment variables."""
        return cls(
            api_key=os.getenv("GEMINI_API_KEY", ""),
            model=os.getenv("GEMINI_EMBEDDING_MODEL") or "gemini-embedding-2",
            dimension=int(os.getenv("EMBEDDING_DIMENSION", "1536")),
            batch_size=int(os.getenv("EMBEDDING_BATCH_SIZE", "100")),
            max_retries=int(os.getenv("EMBEDDING_MAX_RETRIES", "3")),
            timeout_seconds=float(os.getenv("EMBEDDING_TIMEOUT_SECONDS", "30")),
        )


class GeminiEmbeddingService:
    """Create document, query, and symmetric similarity embeddings with Gemini."""

    def __init__(self, settings: EmbeddingSettings, client: Any | None = None) -> None:
        self.settings = settings
        self._client = client

    def embed_documents(self, inputs: Sequence[EmbeddingInput]) -> list[EmbeddingResult]:
        """Embed retrieval documents using their titles and source text."""
        prepared = [
            f"title: {item.title.strip() if item.title and item.title.strip() else 'none'} | text: {item.text}"
            for item in inputs
        ]
        return self._embed_inputs(inputs, prepared)

    def embed_query(self, question: str) -> list[float]:
        """Embed one RAG question using the question-answering task prefix."""
        if not question.strip():
            raise ValueError("The embedding query must not be empty.")
        item = EmbeddingInput(id=uuid4(), text=question)
        [result] = self._embed_inputs(
            [item], [f"task: question answering | query: {question.strip()}"]
        )
        return list(result.values)

    def embed_for_similarity(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed text for symmetric semantic similarity comparisons."""
        inputs = [EmbeddingInput(id=uuid4(), text=text) for text in texts]
        prepared = [f"task: sentence similarity | query: {text}" for text in texts]
        return [list(result.values) for result in self._embed_inputs(inputs, prepared)]

    def _embed_inputs(
        self,
        inputs: Sequence[EmbeddingInput],
        prepared_texts: Sequence[str],
    ) -> list[EmbeddingResult]:
        if len(inputs) != len(prepared_texts):
            raise ValueError("Embedding inputs and prepared text must have matching lengths.")
        if any(not item.text.strip() for item in inputs):
            raise ValueError("Embedding document text must not be empty.")
        if not inputs:
            return []

        results: list[EmbeddingResult] = []
        for start in range(0, len(inputs), self.settings.batch_size):
            stop = min(start + self.settings.batch_size, len(inputs))
            vectors = self._request_vectors(prepared_texts[start:stop], start)
            if len(vectors) != stop - start:
                raise EmbeddingError("Gemini returned an incomplete embedding batch.")
            results.extend(
                EmbeddingResult(id=inputs[index].id, values=tuple(vector))
                for index, vector in zip(range(start, stop), vectors, strict=True)
            )
        return results

    def _request_vectors(self, texts: Sequence[str], batch_offset: int) -> list[list[float]]:
        client = self._get_client()
        from google.genai import types

        contents = [
            types.Content(parts=[types.Part.from_text(text=text)]) for text in texts
        ]
        config = types.EmbedContentConfig(
            output_dimensionality=self.settings.dimension,
        )

        for attempt in range(self.settings.max_retries + 1):
            try:
                response = client.models.embed_content(
                    model=self.settings.model,
                    contents=contents,
                    config=config,
                )
                embeddings = getattr(response, "embeddings", None)
                if embeddings is None or len(embeddings) != len(texts):
                    raise EmbeddingError("Gemini returned an incomplete embedding batch.")
                return [self._validate_vector(item) for item in embeddings]
            except EmbeddingError:
                raise
            except Exception as exc:
                if attempt >= self.settings.max_retries or not self._is_retryable(exc):
                    raise EmbeddingError(
                        f"Gemini embedding request failed for batch starting at {batch_offset}."
                    ) from exc
                time.sleep(min(0.5 * (2**attempt), 8.0))

        raise EmbeddingError("Gemini embedding request exhausted its retry budget.")

    def _get_client(self) -> Any:
        if self._client is None:
            from google import genai
            from google.genai import types

            self._client = genai.Client(
                api_key=self.settings.api_key,
                http_options=types.HttpOptions(
                    timeout=int(self.settings.timeout_seconds * 1000)
                ),
            )
        return self._client

    def _validate_vector(self, embedding: Any) -> list[float]:
        raw_values = getattr(embedding, "values", None)
        if raw_values is None and isinstance(embedding, dict):
            raw_values = embedding.get("values")
        if raw_values is None:
            raise EmbeddingError("Gemini returned an embedding without vector values.")

        values = [float(value) for value in raw_values]
        if len(values) != self.settings.dimension:
            raise EmbeddingError(
                f"Gemini returned {len(values)} dimensions; expected {self.settings.dimension}."
            )
        if not all(math.isfinite(value) for value in values):
            raise EmbeddingError("Gemini returned non-finite embedding values.")
        return values

    @staticmethod
    def _is_retryable(error: Exception) -> bool:
        if isinstance(error, (TimeoutError, ConnectionError)):
            return True
        status_code = getattr(error, "status_code", getattr(error, "code", None))
        return status_code == 429 or (
            isinstance(status_code, int) and 500 <= status_code < 600
        )


class GeminiSimilarityScorer:
    """Compare adjacent chunk units using sentence-similarity embeddings."""

    def __init__(self, embedding_service: GeminiEmbeddingService) -> None:
        self.embedding_service = embedding_service
        self._vectors: dict[str, list[float]] = {}

    def similarity(self, left: str, right: str) -> float:
        """Return cosine similarity for two symmetric text vectors."""
        left_vector = self._vector(left)
        right_vector = self._vector(right)
        left_norm = math.sqrt(sum(value * value for value in left_vector))
        right_norm = math.sqrt(sum(value * value for value in right_vector))
        if left_norm == 0 or right_norm == 0:
            raise EmbeddingError("Gemini returned a zero-length similarity vector.")
        dot_product = sum(
            left_value * right_value
            for left_value, right_value in zip(left_vector, right_vector, strict=True)
        )
        return max(-1.0, min(1.0, dot_product / (left_norm * right_norm)))

    def _vector(self, text: str) -> list[float]:
        vector = self._vectors.get(text)
        if vector is None:
            [vector] = self.embedding_service.embed_for_similarity([text])
            self._vectors[text] = vector
        return vector
