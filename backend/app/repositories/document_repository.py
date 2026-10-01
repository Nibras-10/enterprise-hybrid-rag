"""Database operations for document uploads and ingestion jobs."""

from uuid import UUID

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models import Document, DocumentVersion, IngestionJob


class DocumentRepository:
    """Encapsulate document persistence queries used by ingestion services."""

    def __init__(self, session: Session) -> None:
        self.session = session

    def list_documents(self, *, offset: int = 0, limit: int = 50) -> list[Document]:
        """List the newest documents using bounded pagination."""
        if offset < 0 or not 1 <= limit <= 200:
            raise ValueError("Document pagination is invalid.")
        return list(self.session.scalars(
            select(Document).order_by(Document.created_at.desc()).offset(offset).limit(limit)
        ).all())

    def get_document(self, document_id: UUID) -> Document | None:
        """Fetch one document with versions and jobs available for response mapping."""
        return self.session.scalar(
            select(Document).where(Document.id == document_id)
        )

    def delete_document(self, document: Document) -> None:
        """Delete the document and its relationally-cascaded metadata."""
        self.session.delete(document)
        self.session.flush()

    def content_hash_exists(self, content_hash: str) -> bool:
        """Return whether this exact file content has already been uploaded."""
        statement = select(DocumentVersion.id).where(DocumentVersion.content_hash == content_hash).limit(1)
        return self.session.scalar(statement) is not None

    def create_uploaded(
        self,
        *,
        document_id: UUID,
        filename: str,
        storage_reference: str,
        content_hash: str,
    ) -> tuple[Document, IngestionJob]:
        """Add document, initial version, and queued ingestion job in one transaction."""
        suffix = filename.rsplit(".", maxsplit=1)[-1].lower()
        document = Document(
            id=document_id,
            filename=filename,
            title=filename.rsplit(".", maxsplit=1)[0],
            document_type=suffix,
            source="upload",
            status="UPLOADED",
            file_reference=storage_reference,
        )
        version = DocumentVersion(version=1, content_hash=content_hash, status="UPLOADED")
        job = IngestionJob(status="QUEUED")
        document.versions.append(version)
        document.ingestion_jobs.append(job)
        self.session.add(document)
        self.session.flush()
        return document, job
