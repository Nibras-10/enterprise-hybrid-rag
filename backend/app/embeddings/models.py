"""Input and output records for embedding generation."""

from dataclasses import dataclass
from uuid import UUID


@dataclass(frozen=True, slots=True)
class EmbeddingInput:
    """Text input associated with its source chunk or document identity."""

    id: UUID
    text: str
    title: str | None = None


@dataclass(frozen=True, slots=True)
class EmbeddingResult:
    """Embedding vector associated with the originating input ID."""

    id: UUID
    values: tuple[float, ...]
