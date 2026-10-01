"""Persistence helpers for version-scoped semantic chunks."""

from collections.abc import Sequence
from uuid import UUID

from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from app.chunking.models import ChunkDraft
from app.models import Chunk, DocumentVersion, RetrievalResult


class ChunkRepository:
    """Replace one document version's chunks atomically."""

    def __init__(self, session: Session) -> None:
        self.session = session

    def replace_version_chunks(
        self,
        document_version_id: UUID,
        drafts: Sequence[ChunkDraft],
    ) -> list[Chunk]:
        """Replace persisted chunks without leaving retrieval rows for stale chunks."""
        version = self.session.get(DocumentVersion, document_version_id)
        if version is None:
            raise LookupError("Document version was not found.")
        if any(draft.document_version_id != document_version_id for draft in drafts):
            raise ValueError("All chunk drafts must belong to the requested document version.")
        if any(draft.document_id != version.document_id for draft in drafts):
            raise ValueError("Chunk drafts must belong to the document that owns the version.")

        existing_ids = self.session.scalars(
            select(Chunk.id).where(Chunk.document_version_id == document_version_id)
        ).all()
        if existing_ids:
            self.session.execute(
                delete(RetrievalResult).where(RetrievalResult.chunk_id.in_(existing_ids))
            )
            self.session.execute(
                delete(Chunk).where(Chunk.document_version_id == document_version_id)
            )

        chunks = [
            Chunk(
                id=draft.id,
                document_id=draft.document_id,
                document_version_id=draft.document_version_id,
                chunk_index=draft.chunk_index,
                text=draft.text,
                page_number=draft.page_number,
                section_title=draft.section_title,
                chunk_type=draft.chunk_type,
                token_count=draft.token_count,
                chunk_metadata=draft.metadata,
            )
            for draft in drafts
        ]
        self.session.add_all(chunks)
        self.session.flush()
        return chunks
