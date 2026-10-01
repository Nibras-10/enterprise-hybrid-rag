"""Build citation responses only from sources supplied to generation."""

from dataclasses import dataclass
from uuid import UUID

from app.generation.gemini_generator import GeneratedAnswer


@dataclass(frozen=True, slots=True)
class Citation:
    """Public source citation for a retrieved document chunk."""

    document_id: UUID
    document_name: str
    page_number: int | None
    section_title: str | None
    chunk_id: UUID


class CitationBuilder:
    """Whitelist and deduplicate cited chunks before returning them to clients."""

    @staticmethod
    def build(answer: GeneratedAnswer) -> list[Citation]:
        cited_ids = set(answer.source_ids)
        citations: list[Citation] = []
        seen_chunks: set[UUID] = set()
        for source in answer.sources:
            if source.source_id not in cited_ids or source.chunk_id in seen_chunks:
                continue
            seen_chunks.add(source.chunk_id)
            citations.append(
                Citation(
                    document_id=source.document_id,
                    document_name=source.document_name,
                    page_number=source.page_number,
                    section_title=source.section_title,
                    chunk_id=source.chunk_id,
                )
            )
        return citations
