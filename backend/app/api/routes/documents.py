"""Document upload endpoints."""

import logging
import os
from pathlib import Path
from uuid import UUID

from fastapi import APIRouter, BackgroundTasks, Depends, File, HTTPException, Query, UploadFile, status
from sqlalchemy import select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from app.api.dependencies import _session_factory, get_database_session, get_redis_controls
from app.api.service_factory import create_indexing_pipeline
from app.core.security import get_tenant_id
from app.core.settings import UploadSettings
from app.models import Document
from app.models import IngestionJob
from app.ingestion.lifecycle import DocumentLifecycleService, DocumentStatus
from app.repositories.document_repository import DocumentRepository
from app.ingestion.storage import UploadTooLargeError
from app.ingestion.upload_validation import UploadValidationError
from app.schemas.document import DocumentRead, DocumentUploadResponse, IngestionJobRead
from app.services.document_upload_service import (
    DocumentUploadService,
    DuplicateDocumentError,
)

router = APIRouter(prefix="/api/documents", tags=["documents"])
logger = logging.getLogger(__name__)


def _process_uploaded_document(document_id: UUID, tenant_id: UUID) -> None:
    """Process a queued document with a separate session after upload response."""
    try:
        with _session_factory()() as session:
            try:
                create_indexing_pipeline(session).ingest(document_id, tenant_id=tenant_id)
            except Exception as exc:
                session.rollback()
                document = session.get(Document, document_id)
                job = session.scalar(
                    select(IngestionJob)
                    .where(IngestionJob.document_id == document_id)
                    .order_by(IngestionJob.created_at.desc())
                )
                if document is not None and job is not None and document.status not in {
                    DocumentStatus.READY,
                    DocumentStatus.FAILED,
                }:
                    try:
                        DocumentLifecycleService(session).fail(
                            document,
                            job,
                            f"Document processing failed ({type(exc).__name__}).",
                        )
                    except Exception:
                        session.rollback()
                        logger.exception("Could not record background ingestion failure")
                raise
    except Exception:
        logger.exception("Background document processing failed", extra={"document_id": str(document_id)})
    finally:
        try:
            get_redis_controls().invalidate_tenant_cache(tenant_id)
        except Exception:
            logger.exception("Query cache invalidation failed after document processing")


@router.post("", response_model=DocumentUploadResponse, status_code=status.HTTP_201_CREATED)
def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    session: Session = Depends(get_database_session),
    tenant_id: UUID = Depends(get_tenant_id),
) -> DocumentUploadResponse:
    """Validate and register a document, then queue it for ingestion."""
    try:
        settings = UploadSettings.from_environment()
        service = DocumentUploadService(session, settings)
        uploaded = service.upload(file.filename, file.file)
    except UploadTooLargeError as exc:
        raise HTTPException(status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, detail=str(exc)) from exc
    except DuplicateDocumentError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc
    except UploadValidationError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except SQLAlchemyError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Document metadata could not be saved.",
        ) from exc
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Document upload settings are invalid.",
        ) from exc
    finally:
        file.file.close()

    background_tasks.add_task(_process_uploaded_document, uploaded.document.id, tenant_id)
    return DocumentUploadResponse(
        id=uploaded.document.id,
        filename=uploaded.document.filename,
        title=uploaded.document.title,
        document_type=uploaded.document.document_type,
        status=uploaded.document.status,
        created_at=uploaded.document.created_at,
        ingestion_job=IngestionJobRead(
            id=uploaded.ingestion_job.id,
            status=uploaded.ingestion_job.status,
            created_at=uploaded.ingestion_job.created_at,
        ),
    )


@router.get("", response_model=list[DocumentRead])
def list_documents(
    offset: int = Query(default=0, ge=0),
    limit: int = Query(default=50, ge=1, le=200),
    session: Session = Depends(get_database_session),
    _tenant_id: UUID = Depends(get_tenant_id),
) -> list[Document]:
    """List uploaded documents with bounded pagination."""
    return DocumentRepository(session).list_documents(offset=offset, limit=limit)


@router.get("/{document_id}", response_model=DocumentRead)
def get_document(
    document_id: UUID,
    session: Session = Depends(get_database_session),
    _tenant_id: UUID = Depends(get_tenant_id),
) -> Document:
    """Read document metadata, versions, and ingestion status."""
    document = DocumentRepository(session).get_document(document_id)
    if document is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document was not found.")
    return document


@router.delete("/{document_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_document(
    document_id: UUID,
    session: Session = Depends(get_database_session),
    tenant_id: UUID = Depends(get_tenant_id),
) -> None:
    """Remove document metadata, local bytes, and configured search indexes."""
    repository = DocumentRepository(session)
    document = repository.get_document(document_id)
    if document is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document was not found.")
    if document.status in {"UPLOADED", "PROCESSING"}:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Document processing must finish before deletion.",
        )

    storage = UploadSettings.from_environment().storage_path
    file_path: Path | None = None
    if document.file_reference:
        candidate = (storage / document.file_reference).resolve()
        try:
            candidate.relative_to(storage.resolve())
        except ValueError as exc:
            raise HTTPException(status_code=500, detail="Stored document path is invalid.") from exc
        file_path = candidate

    try:
        from app.retrieval.bm25 import BM25Retriever

        BM25Retriever(tenant_id).remove_document(document_id)
        if os.getenv("PINECONE_API_KEY") and os.getenv("PINECONE_INDEX_NAME"):
            from app.retrieval.pinecone_store import PineconeSettings, PineconeVectorStore

            store = PineconeVectorStore(PineconeSettings.from_environment())
            store.delete_document(document_id, tenant_id=tenant_id)
        repository.delete_document(document)
        session.commit()
    except Exception as exc:
        session.rollback()
        raise HTTPException(status_code=503, detail="Document indexes or metadata could not be removed.") from exc

    if file_path is not None:
        file_path.unlink(missing_ok=True)
    try:
        get_redis_controls().invalidate_tenant_cache(tenant_id)
    except Exception:
        logger.exception("Query cache invalidation failed after document deletion")
