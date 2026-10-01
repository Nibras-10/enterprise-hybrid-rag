"""Chunk candidate value objects shared by chunking and persistence layers."""

from dataclasses import dataclass, field
from typing import Any
from uuid import UUID


@dataclass(frozen=True, slots=True)
class ChunkDraft:
    """A metadata-complete chunk before it is persisted or embedded."""

    id: UUID
    document_id: UUID
    document_version_id: UUID
    chunk_index: int
    text: str
    page_number: int | None
    section_title: str | None
    chunk_type: str
    token_count: int
    metadata: dict[str, Any] = field(default_factory=dict)
