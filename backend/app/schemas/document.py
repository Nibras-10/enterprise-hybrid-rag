"""Document API response schemas."""

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict


class IngestionJobRead(BaseModel):
    """Public status for the initial document-ingestion job."""

    model_config = ConfigDict(from_attributes=True)

    id: UUID
    status: str
    created_at: datetime


class DocumentUploadResponse(BaseModel):
    """Public document metadata returned after an upload is registered."""

    model_config = ConfigDict(from_attributes=True)

    id: UUID
    filename: str
    title: str | None
    document_type: str | None
    status: str
    created_at: datetime
    ingestion_job: IngestionJobRead


class DocumentRead(BaseModel):
    """Document metadata and latest processing state exposed by the API."""

    model_config = ConfigDict(from_attributes=True)

    id: UUID
    filename: str
    title: str | None
    document_type: str | None
    source: str | None
    status: str
    created_at: datetime
    updated_at: datetime
    versions: list["DocumentVersionRead"]
    ingestion_jobs: list[IngestionJobRead]


class DocumentVersionRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    version: int
    page_count: int | None
    status: str
    created_at: datetime
