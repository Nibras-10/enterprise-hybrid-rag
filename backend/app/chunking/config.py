"""Validated environment configuration for structure-aware chunking."""

from dataclasses import dataclass
import os

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True, slots=True)
class ChunkingConfig:
    """Size and semantic-boundary settings for generated chunks."""

    target_tokens: int = 600
    min_tokens: int = 120
    max_tokens: int = 900
    similarity_threshold: float = 0.72

    def __post_init__(self) -> None:
        if not 0 < self.min_tokens <= self.target_tokens <= self.max_tokens:
            raise ValueError("Chunk token limits must satisfy 0 < min <= target <= max.")
        if not -1.0 <= self.similarity_threshold <= 1.0:
            raise ValueError("Cosine similarity threshold must be between minus one and one.")

    @classmethod
    def from_environment(cls) -> "ChunkingConfig":
        """Read chunking options from environment variables."""
        return cls(
            target_tokens=int(os.getenv("CHUNK_TARGET_TOKENS", "600")),
            min_tokens=int(os.getenv("CHUNK_MIN_TOKENS", "120")),
            max_tokens=int(os.getenv("CHUNK_MAX_TOKENS", "900")),
            similarity_threshold=float(os.getenv("SEMANTIC_SIMILARITY_THRESHOLD", "0.72")),
        )
