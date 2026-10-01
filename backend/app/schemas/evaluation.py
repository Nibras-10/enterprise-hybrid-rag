"""Evaluation request and read models for the API."""

from datetime import datetime
from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class EvaluationQuestionInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question: str = Field(min_length=1, max_length=4000)
    ground_truth: str | None = Field(default=None, max_length=12000)
    relevant_chunk_ids: list[str] = Field(default_factory=list, max_length=100)
    relevant_contexts: list[str] = Field(default_factory=list, max_length=100)
    expected_document: str | None = Field(default=None, max_length=500)
    expected_page: int | None = Field(default=None, ge=1)
    expected_chunk: str | None = Field(default=None, max_length=100)
    document_ids: list[UUID] = Field(default_factory=list, max_length=100)


class EvaluationRunRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1, max_length=200)
    description: str = Field(default="", max_length=2000)
    questions: list[EvaluationQuestionInput] = Field(min_length=1, max_length=100)
    include_ragas: bool = False
    retrieval_k: int = Field(default=10, ge=1, le=100)


class EvaluationRunSummary(BaseModel):
    id: UUID
    dataset_id: UUID
    dataset_name: str
    status: str
    started_at: datetime | None
    completed_at: datetime | None


class EvaluationRunDetail(EvaluationRunSummary):
    results: list[dict[str, Any]]
