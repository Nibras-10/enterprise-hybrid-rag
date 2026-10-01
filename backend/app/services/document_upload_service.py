"""Validate, store, deduplicate, and register uploaded documents."""

from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO
from uuid import UUID, uuid4

from sqlalchemy.orm import Session

from app.core.settings import UploadSettings
from app.ingestion.storage import LocalDocumentStorage, UploadTooLargeError
from app.ingestion.upload_validation import (
    UploadValidationError,
    sanitize_filename,
    validate_file_content,
)
from app.models import Document, IngestionJob
from app.repositories.document_repository import DocumentRepository


class DuplicateDocumentError(ValueError):
    """Raised when identical bytes already exist in the document collection."""


@dataclass(frozen=True, slots=True)
class UploadedDocument:
    """Persisted document and its initial queued ingestion job."""

    document: Document
    ingestion_job: IngestionJob


class DocumentUploadService:
    """Coordinate content checks, storage, deduplication, and metadata persistence."""

    def __init__(self, session: Session, settings: UploadSettings) -> None:
        self.repository = DocumentRepository(session)
        self.session = session
        self.storage = LocalDocumentStorage(settings.storage_path, settings.max_upload_size_bytes)

    def upload(self, filename: str | None, source: BinaryIO) -> UploadedDocument:
        """Register one validated file or clean up all partial state on failure."""
        clean_filename = sanitize_filename(filename)
        extension = Path(clean_filename).suffix.lower()
        staged = self.storage.stage(source)
        storage_reference: str | None = None

        try:
            validate_file_content(staged.path, extension)
            if self.repository.content_hash_exists(staged.sha256):
                raise DuplicateDocumentError("A document with identical content already exists.")

            document_id = uuid4()
            storage_reference = self.storage.commit(staged, document_id, extension)
            document, job = self.repository.create_uploaded(
                document_id=document_id,
                filename=clean_filename,
                storage_reference=storage_reference,
                content_hash=staged.sha256,
            )
            self.session.commit()
            return UploadedDocument(document=document, ingestion_job=job)
        except Exception:
            self.session.rollback()
            self.storage.discard(staged)
            if storage_reference is not None:
                self.storage.remove(storage_reference)
            raise
