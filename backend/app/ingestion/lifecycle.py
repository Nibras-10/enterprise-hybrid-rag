"""Explicit document and ingestion-job lifecycle transitions."""

from datetime import UTC, datetime
from enum import StrEnum

from sqlalchemy.orm import Session

from app.ingestion.upload_validation import UploadValidationError
from app.models import Document, IngestionJob


class DocumentStatus(StrEnum):
    """Supported document processing states."""

    UPLOADED = "UPLOADED"
    PROCESSING = "PROCESSING"
    PARSED = "PARSED"
    CHUNKED = "CHUNKED"
    EMBEDDED = "EMBEDDED"
    INDEXED = "INDEXED"
    READY = "READY"
    FAILED = "FAILED"


class IngestionStatus(StrEnum):
    """Supported ingestion-job states."""

    QUEUED = "QUEUED"
    PROCESSING = "PROCESSING"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"


_ALLOWED_TRANSITIONS: dict[DocumentStatus, set[DocumentStatus]] = {
    DocumentStatus.UPLOADED: {DocumentStatus.PROCESSING, DocumentStatus.FAILED},
    DocumentStatus.PROCESSING: {DocumentStatus.PARSED, DocumentStatus.FAILED},
    DocumentStatus.PARSED: {DocumentStatus.CHUNKED, DocumentStatus.FAILED},
    DocumentStatus.CHUNKED: {DocumentStatus.EMBEDDED, DocumentStatus.FAILED},
    DocumentStatus.EMBEDDED: {DocumentStatus.INDEXED, DocumentStatus.FAILED},
    DocumentStatus.INDEXED: {DocumentStatus.READY, DocumentStatus.FAILED},
    DocumentStatus.READY: set(),
    DocumentStatus.FAILED: set(),
}


class DocumentLifecycleService:
    """Persist validated lifecycle changes for a document and its active job."""

    def __init__(self, session: Session) -> None:
        self.session = session

    def start(self, document: Document, job: IngestionJob) -> None:
        """Move a newly uploaded document and queued job into processing."""
        if job.status != IngestionStatus.QUEUED:
            raise UploadValidationError("Only queued ingestion jobs can be started.")
        self._transition(document, DocumentStatus.PROCESSING)
        now = datetime.now(UTC)
        job.status = IngestionStatus.PROCESSING
        job.started_at = now
        self.session.commit()

    def advance(self, document: Document, next_status: DocumentStatus) -> None:
        """Advance one processing step and complete the job when ready."""
        job = self._active_job(document) if next_status is DocumentStatus.READY else None
        self._transition(document, next_status)
        if job is not None:
            job.status = IngestionStatus.SUCCEEDED
            job.completed_at = datetime.now(UTC)
        self.session.commit()

    def fail(self, document: Document, job: IngestionJob, error_message: str) -> None:
        """Record a bounded failure message on both document and job."""
        if DocumentStatus.FAILED not in _ALLOWED_TRANSITIONS[DocumentStatus(document.status)]:
            raise UploadValidationError(f"Cannot fail a document in {document.status} state.")
        now = datetime.now(UTC)
        document.status = DocumentStatus.FAILED
        job.status = IngestionStatus.FAILED
        job.error_message = error_message[:4000]
        job.completed_at = now
        self.session.commit()

    @staticmethod
    def _transition(document: Document, next_status: DocumentStatus) -> None:
        current_status = DocumentStatus(document.status)
        if next_status not in _ALLOWED_TRANSITIONS[current_status]:
            raise UploadValidationError(
                f"Invalid document lifecycle transition: {current_status} -> {next_status}."
            )
        document.status = next_status

    @staticmethod
    def _active_job(document: Document) -> IngestionJob:
        for job in reversed(document.ingestion_jobs):
            if job.status in {IngestionStatus.QUEUED, IngestionStatus.PROCESSING}:
                return job
        raise UploadValidationError("No active ingestion job exists for this document.")
