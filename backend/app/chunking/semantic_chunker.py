"""Semantic chunker that respects headings, paragraphs, lists, and table boundaries."""

from collections.abc import Sequence
from dataclasses import dataclass, field
import re
from typing import Any, Protocol
from uuid import UUID, uuid4

from app.chunking.config import ChunkingConfig
from app.chunking.models import ChunkDraft
from app.ingestion.elements import DocumentElement


class SimilarityScorer(Protocol):
    """Score semantic similarity for adjacent text units on a zero-to-one scale."""

    def similarity(self, left: str, right: str) -> float:
        """Return the semantic similarity between two text inputs."""


@dataclass(frozen=True, slots=True)
class _SemanticUnit:
    """An atomic paragraph, list, or table together with its source metadata."""

    document_id: UUID
    text: str
    page_number: int | None
    section_title: str | None
    element_type: str
    metadata: dict[str, Any] = field(default_factory=dict)


class SemanticChunker:
    """Group related parsed elements while preserving structure and citations."""

    _SECTION_TYPES = {"Title", "Heading", "Heading 1", "Heading 2", "Heading 3"}
    _TOKEN_PATTERN = re.compile(r"\w+|[^\w\s]", re.UNICODE)

    def __init__(self, config: ChunkingConfig, similarity_scorer: SimilarityScorer) -> None:
        self.config = config
        self.similarity_scorer = similarity_scorer

    def chunk(
        self,
        elements: Sequence[DocumentElement],
        document_version_id: UUID,
    ) -> list[ChunkDraft]:
        """Create chunks at structural or semantic boundaries and retain source metadata."""
        if not elements:
            return []

        document_id = elements[0].document_id
        if any(element.document_id != document_id for element in elements):
            raise ValueError("All parsed elements must belong to the same document.")

        units = self._semantic_units(elements)
        drafts: list[ChunkDraft] = []
        current_units: list[_SemanticUnit] = []

        def flush() -> None:
            if current_units:
                drafts.append(self._build_chunk(current_units, document_version_id, len(drafts)))
                current_units.clear()

        for unit in units:
            if unit.element_type == "Table":
                flush()
                drafts.append(self._build_chunk([unit], document_version_id, len(drafts)))
                continue

            if not current_units:
                current_units.append(unit)
                continue

            current_text = "\n\n".join(item.text for item in current_units)
            current_tokens = self.count_tokens(current_text)
            unit_tokens = self.count_tokens(unit.text)
            section_changed = current_units[-1].section_title != unit.section_title
            exceeds_maximum = current_tokens + unit_tokens > self.config.max_tokens
            reaches_target = (
                current_tokens >= self.config.min_tokens
                and current_tokens + unit_tokens > self.config.target_tokens
            )
            semantic_boundary = False
            should_check_similarity = not (
                section_changed or exceeds_maximum or reaches_target
            ) and current_tokens >= self.config.min_tokens
            if should_check_similarity:
                similarity = self.similarity_scorer.similarity(
                    current_units[-1].text,
                    unit.text,
                )
                if not -1.0 <= similarity <= 1.0:
                    raise ValueError("Similarity scorer must return cosine similarity between minus one and one.")
                semantic_boundary = similarity < self.config.similarity_threshold

            if section_changed or exceeds_maximum or reaches_target or semantic_boundary:
                flush()
            current_units.append(unit)

        flush()
        return drafts

    @classmethod
    def count_tokens(cls, text: str) -> int:
        """Estimate tokens using word and punctuation boundaries."""
        return len(cls._TOKEN_PATTERN.findall(text))

    @classmethod
    def _semantic_units(cls, elements: Sequence[DocumentElement]) -> list[_SemanticUnit]:
        """Attach heading elements to their following content and isolate tables."""
        units: list[_SemanticUnit] = []
        pending_headings: list[str] = []

        for element in elements:
            if element.element_type in cls._SECTION_TYPES:
                pending_headings.append(element.text)
                continue

            text = element.text
            if pending_headings:
                text = "\n".join((*pending_headings, text))
                pending_headings.clear()

            units.append(
                _SemanticUnit(
                    document_id=element.document_id,
                    text=text,
                    page_number=element.page_number,
                    section_title=element.section_title,
                    element_type=element.element_type,
                    metadata=dict(element.metadata),
                )
            )

        if pending_headings:
            heading = "\n".join(pending_headings)
            last_element = elements[-1]
            units.append(
                _SemanticUnit(
                    document_id=last_element.document_id,
                    text=heading,
                    page_number=last_element.page_number,
                    section_title=last_element.section_title or heading,
                    element_type="Heading",
                )
            )

        return units

    def _build_chunk(
        self,
        units: Sequence[_SemanticUnit],
        document_version_id: UUID,
        chunk_index: int,
    ) -> ChunkDraft:
        """Combine source units and preserve per-element/page metadata."""
        text = "\n\n".join(unit.text for unit in units)
        element_types = {unit.element_type for unit in units}
        chunk_type = next(iter(element_types)) if len(element_types) == 1 else "Mixed"
        page_numbers = list(dict.fromkeys(unit.page_number for unit in units if unit.page_number is not None))
        metadata: dict[str, Any] = {
            "page_numbers": page_numbers,
            "source_elements": [
                {
                    "element_type": unit.element_type,
                    "page_number": unit.page_number,
                    **unit.metadata,
                }
                for unit in units
            ],
        }
        token_count = self.count_tokens(text)
        if token_count > self.config.max_tokens:
            metadata["oversized_atomic_unit"] = True

        return ChunkDraft(
            id=uuid4(),
            document_id=units[0].document_id,
            document_version_id=document_version_id,
            chunk_index=chunk_index,
            text=text,
            page_number=page_numbers[0] if page_numbers else None,
            section_title=units[0].section_title,
            chunk_type=chunk_type,
            token_count=token_count,
            metadata=metadata,
        )
