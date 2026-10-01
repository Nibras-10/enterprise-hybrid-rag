"""Grounding context and source identifiers for answer generation."""

from dataclasses import dataclass
import json
from typing import Any
from uuid import UUID

from app.retrieval.models import RetrievalCandidate


@dataclass(frozen=True, slots=True)
class ContextSource:
    """One retrieved chunk and its server-controlled citation identity."""

    source_id: str
    document_id: UUID
    document_name: str
    page_number: int | None
    section_title: str | None
    chunk_id: UUID
    text: str


@dataclass(frozen=True, slots=True)
class GenerationContext:
    """Serialized user question and document data plus citation source mapping."""

    serialized_request_data: str
    sources: tuple[ContextSource, ...]
    source_ids: frozenset[str]


class ContextBuilder:
    """Serialize retrieved data separately from model system instructions."""

    def build(
        self,
        question: str,
        candidates: list[RetrievalCandidate],
    ) -> GenerationContext:
        """Create source IDs from retrieved chunks and JSON-escape untrusted text."""
        sources: list[ContextSource] = []
        for index, candidate in enumerate(candidates, start=1):
            metadata: dict[str, Any] = candidate.metadata
            page_number = metadata.get("page_number")
            if not isinstance(page_number, int):
                page_number = None
            section_title = metadata.get("section_title")
            if not isinstance(section_title, str):
                section_title = None
            filename = metadata.get("source_filename")
            if not isinstance(filename, str) or not filename:
                filename = "Unknown document"
            sources.append(
                ContextSource(
                    source_id=f"S{index}",
                    document_id=candidate.document_id,
                    document_name=filename,
                    page_number=page_number,
                    section_title=section_title,
                    chunk_id=candidate.chunk_id,
                    text=candidate.text,
                )
            )

        request_data = {
            "user_question": question,
            "retrieved_document_data": [
                {
                    "source_id": source.source_id,
                    "document_name": source.document_name,
                    "document_id": str(source.document_id),
                    "page_number": source.page_number,
                    "section_title": source.section_title,
                    "chunk_id": str(source.chunk_id),
                    "text": source.text,
                }
                for source in sources
            ],
        }
        return GenerationContext(
            serialized_request_data=json.dumps(request_data, ensure_ascii=False),
            sources=tuple(sources),
            source_ids=frozenset(source.source_id for source in sources),
        )
