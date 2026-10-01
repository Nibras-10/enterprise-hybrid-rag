"""Evaluation execution and persisted run history endpoints."""

import logging
import os
from pathlib import Path
from typing import Any
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import select
from sqlalchemy.orm import Session, selectinload

from app.api.dependencies import get_database_session
from app.api.service_factory import get_rag_components
from app.core.security import get_tenant_id
from app.evaluation.models import EvaluationDataset, EvaluationQuestion, StrategyResult
from app.evaluation.runner import EvaluationRunner
from app.models import EvaluationResult, EvaluationRun
from app.repositories.evaluation_repository import EvaluationRepository
from app.retrieval.bm25 import BM25Retriever
from app.schemas.evaluation import EvaluationRunDetail, EvaluationRunRequest, EvaluationRunSummary

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/evaluations", tags=["evaluations"])


@router.post("/run", status_code=status.HTTP_201_CREATED)
def run_evaluation(
    request: EvaluationRunRequest,
    session: Session = Depends(get_database_session),
    tenant_id: UUID = Depends(get_tenant_id),
) -> dict[str, Any]:
    """Compare dense, BM25, hybrid, and reranked retrieval on a labeled dataset."""
    dataset = EvaluationDataset(
        name=request.name,
        description=request.description,
        questions=tuple(EvaluationQuestion(
            question=item.question,
            ground_truth=item.ground_truth,
            relevant_chunk_ids=tuple(item.relevant_chunk_ids),
            relevant_contexts=tuple(item.relevant_contexts),
            expected_document=item.expected_document,
            expected_page=item.expected_page,
            expected_chunk=item.expected_chunk,
        ) for item in request.questions),
    )
    try:
        dense, hybrid, reranker, generator = get_rag_components(tenant_id)
        bm25 = BM25Retriever(tenant_id)

        def run_strategy(strategy_name: str):
            position = 0

            def run(question: str) -> StrategyResult:
                nonlocal position
                input_row = request.questions[position]
                position += 1
                if input_row.question != question:
                    raise ValueError("Evaluation strategy question order changed unexpectedly.")
                document_ids = input_row.document_ids or None
                dense_results = dense.retrieve(
                    question,
                    tenant_id=tenant_id,
                    document_ids=document_ids,
                    top_k=request.retrieval_k,
                )
                bm25_results = bm25.retrieve(
                    question,
                    top_k=max(request.retrieval_k, reranker.settings.candidate_k),
                    document_ids=document_ids,
                )
                if strategy_name == "dense":
                    candidates = dense_results
                elif strategy_name == "bm25":
                    candidates = bm25_results
                else:
                    candidates = hybrid.fuse(
                        dense_results,
                        bm25_results,
                        top_k=reranker.settings.candidate_k,
                    )
                    if strategy_name == "hybrid_reranked":
                        candidates = list(reranker.rerank(question, candidates).candidates)
                generated = generator.generate(question, candidates)
                plain_candidates = [getattr(item, "candidate", item) for item in candidates]
                return StrategyResult(
                    answer=generated.answer,
                    retrieved_contexts=tuple(candidate.text for candidate in plain_candidates),
                    retrieved_chunk_ids=tuple(str(candidate.chunk_id) for candidate in plain_candidates),
                    metadata={
                        "candidate_count": len(plain_candidates),
                        "model": generated.model,
                        "retrieved_sources": [
                            {
                                "chunk_id": str(candidate.chunk_id),
                                "document_id": str(candidate.document_id),
                                "document_name": candidate.metadata.get("source_filename"),
                                "page_number": candidate.metadata.get("page_number"),
                            }
                            for candidate in plain_candidates
                        ],
                    },
                )
            return run

        strategy_names = ("dense", "bm25", "hybrid", "hybrid_reranked")
        strategies = {name: run_strategy(name) for name in strategy_names}
        run_directory = Path(os.getenv("EVALUATION_RUNS_PATH", "../evaluation/runs")).resolve()
        report = EvaluationRunner(run_directory).run(
            dataset,
            strategies,
            configuration={
                "embedding_model": dense.embedding_service.settings.model,
                "generation_model": generator.settings.model,
                "rerank_model": reranker.settings.model,
                "retrieval_k": request.retrieval_k,
                "hybrid_dense_weight": hybrid.settings.dense_weight,
                "hybrid_bm25_weight": hybrid.settings.bm25_weight,
                "hybrid_rank_constant": hybrid.settings.rank_constant,
                "rerank_candidate_k": reranker.settings.candidate_k,
                "rerank_top_k": reranker.settings.top_k,
            },
            include_ragas=request.include_ragas,
            retrieval_k=request.retrieval_k,
        )

        repository = EvaluationRepository(session)
        for strategy_name, strategy_report in report["strategies"].items():
            question_rows = [
                {
                    "question": item.question,
                    "ground_truth": item.ground_truth,
                    "relevant_chunk_ids": list(item.relevant_chunk_ids),
                    "relevant_contexts": list(item.relevant_contexts),
                    "expected_document": item.expected_document,
                    "expected_page": item.expected_page,
                    "expected_chunk": item.expected_chunk,
                }
                for item in dataset.questions
            ]
            database_run, question_ids = repository.create_run(
                f"{dataset.name}-{strategy_name}",
                dataset.description,
                question_rows,
            )
            result_rows = [
                {"answer": item["answer"], "metrics": item["metrics"]}
                for item in strategy_report["results"]
            ]
            repository.save_results(database_run.id, question_ids, result_rows)
        session.commit()
        return report
    except ValueError as exc:
        session.rollback()
        raise HTTPException(status_code=400, detail="Evaluation request configuration is invalid.") from exc
    except Exception as exc:
        session.rollback()
        logger.exception("Evaluation run failed")
        raise HTTPException(status_code=502, detail="The evaluation run could not be completed.") from exc


@router.get("", response_model=list[EvaluationRunSummary])
def list_evaluations(
    limit: int = Query(default=50, ge=1, le=200),
    session: Session = Depends(get_database_session),
    _tenant_id: UUID = Depends(get_tenant_id),
) -> list[EvaluationRunSummary]:
    """List recent evaluation strategy runs."""
    runs = EvaluationRepository(session).list_runs(limit)
    return [EvaluationRunSummary(
        id=run.id,
        dataset_id=run.dataset_id,
        dataset_name=run.dataset.name,
        status=run.status,
        started_at=run.started_at,
        completed_at=run.completed_at,
    ) for run in runs]


@router.get("/{run_id}", response_model=EvaluationRunDetail)
def get_evaluation(
    run_id: UUID,
    session: Session = Depends(get_database_session),
    _tenant_id: UUID = Depends(get_tenant_id),
) -> EvaluationRunDetail:
    """Return per-question results for a persisted strategy run."""
    run = session.scalar(
        select(EvaluationRun)
        .where(EvaluationRun.id == run_id)
        .options(
            selectinload(EvaluationRun.dataset),
            selectinload(EvaluationRun.results).selectinload(EvaluationResult.question),
        )
    )
    if run is None:
        raise HTTPException(status_code=404, detail="Evaluation run was not found.")
    return EvaluationRunDetail(
        id=run.id,
        dataset_id=run.dataset_id,
        dataset_name=run.dataset.name,
        status=run.status,
        started_at=run.started_at,
        completed_at=run.completed_at,
        results=[{
            "question": result.question.question,
            "ground_truth": result.question.ground_truth,
            "relevant_chunk_ids": result.question.relevant_chunk_ids,
            "relevant_contexts": result.question.relevant_contexts,
            "expected_document": result.question.expected_document,
            "expected_page": result.question.expected_page,
            "expected_chunk": result.question.expected_chunk,
            "answer": result.answer,
            "metrics": result.metrics,
        } for result in run.results],
    )
