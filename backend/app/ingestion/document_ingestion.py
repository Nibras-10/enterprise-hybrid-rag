"""Document processing orchestration for status tracking and structural parsing."""

from uuid import UUID

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.settings import UploadSettings
from app.ingestion.elements import ParsedDocument
from app.ingestion.lifecycle import (
    DocumentLifecycleService,
    DocumentStatus,
    IngestionStatus,
)
from app.ingestion.parser import DocumentParsingError, UnstructuredDocumentParser
from app.ingestion.storage import LocalDocumentStorage
from app.models import Document, DocumentVersion, IngestionJob


class DocumentIngestionService:
    """Start queued jobs, parse source files, and record completion or failure."""

    def __init__(
        self,
        session: Session,
        settings: UploadSettings,
        parser: UnstructuredDocumentParser | None = None,
    ) -> None:
        self.session = session
        self.storage = LocalDocumentStorage(settings.storage_path, settings.max_upload_size_bytes)
        self.parser = parser or UnstructuredDocumentParser()
        self.lifecycle = DocumentLifecycleService(session)

    def parse(self, document_id: UUID) -> ParsedDocument:
        """Parse a queued document while persisting processing and failure status."""
        document = self.session.get(Document, document_id)
        if document is None:
            raise LookupError("Document was not found.")

        job = self.session.scalar(
            select(IngestionJob)
            .where(
                IngestionJob.document_id == document_id,
                IngestionJob.status == IngestionStatus.QUEUED,
            )
            .order_by(IngestionJob.created_at.desc())
        )
        if job is None:
            raise LookupError("No queued ingestion job exists for this document.")

        version = max(document.versions, key=lambda item: item.version)
        self.lifecycle.start(document, job)

        try:
            source_path = self.storage.resolve(document.file_reference or "")
            parsed = self.parser.parse(source_path, document.id)
            version.page_count = parsed.page_count
            version.status = DocumentStatus.PARSED
            self.lifecycle.advance(document, DocumentStatus.PARSED)
            return parsed
        except DocumentParsingError as exc:
            self.lifecycle.fail(document, job, str(exc))
            raise
