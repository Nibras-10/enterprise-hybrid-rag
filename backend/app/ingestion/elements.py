"""Normalized document structure emitted by the parsing layer."""

from dataclasses import dataclass, field
from typing import Any
from uuid import UUID


@dataclass(frozen=True, slots=True)
class DocumentElement:
    """A single structured element that retains page and section context."""

    document_id: UUID
    page_number: int | None
    element_type: str
    text: str
    section_title: str | None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ParsedDocument:
    """Document elements plus the highest observed page number."""

    elements: tuple[DocumentElement, ...]
    page_count: int
