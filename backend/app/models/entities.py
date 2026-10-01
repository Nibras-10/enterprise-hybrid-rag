"""Relational models for users, document ingestion, queries, and evaluation."""

from datetime import datetime
from typing import Any
from uuid import UUID, uuid4

from sqlalchemy import DateTime, ForeignKey, Index, Integer, JSON, String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.types import Uuid

from app.models.base import Base, TimestampMixin, utc_now


class User(TimestampMixin, Base):
    __tablename__ = "users"

    id: Mapped[UUID] = mapped_column(Uuid, primary_key=True, default=uuid4)
    email: Mapped[str] = mapped_column(String(320), unique=True, index=True)
    documents: Mapped[list["Document"]] = relationship(back_populates="owner")
    queries: Mapped[list["Query"]] = relationship(back_populates="user")


class Document(TimestampMixin, Base):
    __tablename__ = "documents"
    __table_args__ = (Index("ix_documents_status_created_at", "status", "created_at"),)

    id: Mapped[UUID] = mapped_column(Uuid, primary_key=True, default=uuid4)
    filename: Mapped[str] = mapped_column(String(255), nullable=False)
    title: Mapped[str | None] = mapped_column(String(500))
    document_type: Mapped[str | None] = mapped_column(String(50))
    source: Mapped[str | None] = mapped_column(String(500))
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="UPLOADED")
    file_reference: Mapped[str | None] = mapped_column(String(1024))
    owner_id: Mapped[UUID | None] = mapped_column(Uuid, ForeignKey("users.id", ondelete="SET NULL"), index=True)
    owner: Mapped[User | None] = relationship(back_populates="documents")
    versions: Mapped[list["DocumentVersion"]] = relationship(back_populates="document", cascade="all, delete-orphan")
    ingestion_jobs: Mapped[list["IngestionJob"]] = relationship(back_populates="document", cascade="all, delete-orphan")
    chunks: Mapped[list["Chunk"]] = relationship(back_populates="document", cascade="all, delete-orphan")


class DocumentVersion(Base):
    __tablename__ = "document_versions"
    __table_args__ = (UniqueConstraint("document_id", "version"),)

    id: Mapped[UUID] = mapped_column(Uuid, primary_key=True, default=uuid4)
    document_id: Mapped[UUID] = mapped_column(Uuid, ForeignKey("documents.id", ondelete="CASCADE"), nullable=False, index=True)
    version: Mapped[int] = mapped_column(Integer, nullable=False)
    content_hash: Mapped[str] = mapped_column(String(128), nullable=False)
    page_count: Mapped[int | None] = mapped_column(Integer)
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="CREATED")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now, nullable=False)
    document: Mapped[Document] = relationship(back_populates="versions")
    chunks: Mapped[list["Chunk"]] = relationship(back_populates="document_version", cascade="all, delete-orphan")


class Chunk(Base):
    __tablename__ = "chunks"
    __table_args__ = (UniqueConstraint("document_version_id", "chunk_index"),)

    id: Mapped[UUID] = mapped_column(Uuid, primary_key=True, default=uuid4)
    document_id: Mapped[UUID] = mapped_column(Uuid, ForeignKey("documents.id", ondelete="CASCADE"), nullable=False, index=True)
    document_version_id: Mapped[UUID] = mapped_column(Uuid, ForeignKey("document_versions.id", ondelete="CASCADE"), nullable=False, index=True)
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    page_number: Mapped[int | None] = mapped_column(Integer)
    section_title: Mapped[str | None] = mapped_column(String(500))
    chunk_type: Mapped[str | None] = mapped_column(String(50))
    token_count: Mapped[int | None] = mapped_column(Integer)
    pinecone_vector_id: Mapped[str | None] = mapped_column(String(256), unique=True)
    chunk_metadata: Mapped[dict[str, Any]] = mapped_column("metadata", JSON, nullable=False, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now, nullable=False)
    document: Mapped[Document] = relationship(back_populates="chunks")
    document_version: Mapped[DocumentVersion] = relationship(back_populates="chunks")
    retrieval_results: Mapped[list["RetrievalResult"]] = relationship(back_populates="chunk", cascade="all, delete-orphan")


class IngestionJob(Base):
    __tablename__ = "ingestion_jobs"
    __table_args__ = (Index("ix_ingestion_jobs_status_created_at", "status", "created_at"),)

    id: Mapped[UUID] = mapped_column(Uuid, primary_key=True, default=uuid4)
    document_id: Mapped[UUID] = mapped_column(Uuid, ForeignKey("documents.id", ondelete="CASCADE"), nullable=False, index=True)
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="QUEUED")
    error_message: Mapped[str | None] = mapped_column(Text)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now, nullable=False)
    document: Mapped[Document] = relationship(back_populates="ingestion_jobs")


class Query(Base):
    __tablename__ = "queries"

    id: Mapped[UUID] = mapped_column(Uuid, primary_key=True, default=uuid4)
    question: Mapped[str] = mapped_column(Text, nullable=False)
    answer: Mapped[str | None] = mapped_column(Text)
    citations: Mapped[list[dict[str, Any]] | None] = mapped_column(JSON)
    user_id: Mapped[UUID | None] = mapped_column(Uuid, ForeignKey("users.id", ondelete="SET NULL"), index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now, nullable=False)
    user: Mapped[User | None] = relationship(back_populates="queries")
    retrieval_results: Mapped[list["RetrievalResult"]] = relationship(back_populates="query", cascade="all, delete-orphan")


class RetrievalResult(Base):
    __tablename__ = "retrieval_results"
    __table_args__ = (
        UniqueConstraint("query_id", "chunk_id", "retrieval_method"),
        Index("ix_retrieval_results_query_rank", "query_id", "rank"),
    )

    id: Mapped[UUID] = mapped_column(Uuid, primary_key=True, default=uuid4)
    query_id: Mapped[UUID] = mapped_column(Uuid, ForeignKey("queries.id", ondelete="CASCADE"), nullable=False, index=True)
    chunk_id: Mapped[UUID] = mapped_column(Uuid, ForeignKey("chunks.id", ondelete="CASCADE"), nullable=False, index=True)
    retrieval_method: Mapped[str] = mapped_column(String(32), nullable=False)
    rank: Mapped[int] = mapped_column(Integer, nullable=False)
    score: Mapped[float | None] = mapped_column()
    rerank_score: Mapped[float | None] = mapped_column()
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now, nullable=False)
    query: Mapped[Query] = relationship(back_populates="retrieval_results")
    chunk: Mapped[Chunk] = relationship(back_populates="retrieval_results")


class EvaluationDataset(Base):
    __tablename__ = "evaluation_datasets"

    id: Mapped[UUID] = mapped_column(Uuid, primary_key=True, default=uuid4)
    name: Mapped[str] = mapped_column(String(255), nullable=False, unique=True)
    description: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now, nullable=False)
    questions: Mapped[list["EvaluationQuestion"]] = relationship(back_populates="dataset", cascade="all, delete-orphan")
    runs: Mapped[list["EvaluationRun"]] = relationship(back_populates="dataset", cascade="all, delete-orphan")


class EvaluationQuestion(Base):
    __tablename__ = "evaluation_questions"
    __table_args__ = (Index("ix_evaluation_questions_dataset_id", "dataset_id"),)

    id: Mapped[UUID] = mapped_column(Uuid, primary_key=True, default=uuid4)
    dataset_id: Mapped[UUID] = mapped_column(Uuid, ForeignKey("evaluation_datasets.id", ondelete="CASCADE"), nullable=False)
    question: Mapped[str] = mapped_column(Text, nullable=False)
    ground_truth: Mapped[str | None] = mapped_column(Text)
    relevant_chunk_ids: Mapped[list[str]] = mapped_column(JSON, nullable=False, default=list)
    relevant_contexts: Mapped[list[str]] = mapped_column(JSON, nullable=False, default=list)
    expected_document: Mapped[str | None] = mapped_column(String(500))
    expected_page: Mapped[int | None] = mapped_column(Integer)
    expected_chunk: Mapped[str | None] = mapped_column(String(100))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now, nullable=False)
    dataset: Mapped[EvaluationDataset] = relationship(back_populates="questions")
    results: Mapped[list["EvaluationResult"]] = relationship(back_populates="question")


class EvaluationRun(Base):
    __tablename__ = "evaluation_runs"

    id: Mapped[UUID] = mapped_column(Uuid, primary_key=True, default=uuid4)
    dataset_id: Mapped[UUID] = mapped_column(Uuid, ForeignKey("evaluation_datasets.id", ondelete="RESTRICT"), nullable=False, index=True)
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="PENDING")
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    dataset: Mapped[EvaluationDataset] = relationship(back_populates="runs")
    results: Mapped[list["EvaluationResult"]] = relationship(back_populates="evaluation_run", cascade="all, delete-orphan")


class EvaluationResult(Base):
    __tablename__ = "evaluation_results"

    id: Mapped[UUID] = mapped_column(Uuid, primary_key=True, default=uuid4)
    evaluation_run_id: Mapped[UUID] = mapped_column(Uuid, ForeignKey("evaluation_runs.id", ondelete="CASCADE"), nullable=False, index=True)
    question_id: Mapped[UUID] = mapped_column(Uuid, ForeignKey("evaluation_questions.id", ondelete="CASCADE"), nullable=False, index=True)
    answer: Mapped[str | None] = mapped_column(Text)
    metrics: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now, nullable=False)
    evaluation_run: Mapped[EvaluationRun] = relationship(back_populates="results")
    question: Mapped[EvaluationQuestion] = relationship(back_populates="results")
