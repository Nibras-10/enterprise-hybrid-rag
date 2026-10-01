"""Persistence operations for evaluation datasets, runs, and per-question metrics."""

from datetime import UTC, datetime
from typing import Any
from uuid import UUID, uuid4

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models import (
    EvaluationDataset as DatasetRecord,
    EvaluationQuestion as QuestionRecord,
    EvaluationResult,
    EvaluationRun,
)


class EvaluationRepository:
    """Store completed evaluation reports in the application metadata database."""

    def __init__(self, session: Session) -> None:
        self.session = session

    def create_run(
        self,
        name: str,
        description: str,
        rows: list[dict[str, Any]],
    ) -> tuple[EvaluationRun, list[UUID]]:
        dataset = DatasetRecord(name=f"{name[:220]}-{uuid4().hex}", description=description)
        self.session.add(dataset)
        self.session.flush()

        question_records: list[QuestionRecord] = []
        for row in rows:
            question = QuestionRecord(
                dataset_id=dataset.id,
                question=row["question"],
                ground_truth=row.get("ground_truth"),
                relevant_chunk_ids=row.get("relevant_chunk_ids", []),
                relevant_contexts=row.get("relevant_contexts", []),
                expected_document=row.get("expected_document"),
                expected_page=row.get("expected_page"),
                expected_chunk=row.get("expected_chunk"),
            )
            self.session.add(question)
            question_records.append(question)
        run = EvaluationRun(dataset_id=dataset.id, status="SUCCEEDED", started_at=datetime.now(UTC))
        self.session.add(run)
        self.session.flush()
        return run, [question.id for question in question_records]

    def save_results(self, run_id: UUID, question_ids: list[UUID], rows: list[dict[str, Any]]) -> None:
        run = self.session.get(EvaluationRun, run_id)
        if run is None:
            raise ValueError("Evaluation run does not exist.")
        if len(question_ids) != len(rows):
            raise ValueError("Evaluation question and result counts do not match.")
        for question_id, row in zip(question_ids, rows, strict=True):
            self.session.add(EvaluationResult(
                evaluation_run_id=run_id,
                question_id=question_id,
                answer=row.get("answer"),
                metrics=row.get("metrics", {}),
            ))
        run.completed_at = datetime.now(UTC)

    def list_runs(self, limit: int = 50) -> list[EvaluationRun]:
        if not 1 <= limit <= 200:
            raise ValueError("limit must be between 1 and 200.")
        return list(self.session.scalars(
            select(EvaluationRun).order_by(EvaluationRun.started_at.desc()).limit(limit)
        ).all())

    def get_run(self, run_id: UUID) -> EvaluationRun | None:
        return self.session.get(EvaluationRun, run_id)
