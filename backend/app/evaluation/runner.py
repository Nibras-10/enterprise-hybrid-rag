"""Run a dataset against named retrieval strategies and write comparable reports."""

from collections.abc import Callable, Mapping
from datetime import UTC, datetime
import json
from pathlib import Path
from typing import Any
from uuid import uuid4

from app.evaluation.metrics import retrieval_metrics, score_with_ragas
from app.evaluation.models import EvaluationDataset, StrategyResult

StrategyRunner = Callable[[str], StrategyResult]


class EvaluationRunner:
    """Execute deterministic strategy comparisons and persist machine-readable reports."""

    def __init__(self, output_directory: Path) -> None:
        self.output_directory = output_directory

    def run(
        self,
        dataset: EvaluationDataset,
        strategies: Mapping[str, StrategyRunner],
        *,
        configuration: dict[str, Any] | None = None,
        include_ragas: bool = False,
        retrieval_k: int = 10,
    ) -> dict[str, Any]:
        if not strategies:
            raise ValueError("At least one retrieval strategy is required.")
        run_id = str(uuid4())
        strategy_reports: dict[str, Any] = {}
        for strategy_name, strategy in strategies.items():
            row_results: list[dict[str, Any]] = []
            ragas_rows: list[dict[str, Any]] = []
            for question in dataset.questions:
                result = strategy(question.question)
                metrics = retrieval_metrics(
                    result.retrieved_chunk_ids,
                    question.relevant_chunk_ids,
                    retrieval_k,
                )
                retrieved_sources = result.metadata.get("retrieved_sources", [])
                matching_sources = retrieved_sources
                if question.expected_document is not None:
                    matching_sources = [
                        source for source in retrieved_sources
                        if source.get("document_name") == question.expected_document
                        or source.get("document_id") == question.expected_document
                    ]
                    metrics["expected_document_hit"] = float(bool(matching_sources))
                if question.expected_page is not None:
                    metrics["expected_page_hit"] = float(any(
                        source.get("page_number") == question.expected_page
                        for source in matching_sources
                    ))
                if question.expected_chunk is not None:
                    metrics["expected_chunk_hit"] = float(
                        question.expected_chunk in result.retrieved_chunk_ids
                    )
                row = {
                    "question": question.question,
                    "ground_truth": question.ground_truth,
                    "relevant_chunk_ids": list(question.relevant_chunk_ids),
                    "relevant_contexts": list(question.relevant_contexts),
                    "expected_document": question.expected_document,
                    "expected_page": question.expected_page,
                    "expected_chunk": question.expected_chunk,
                    "answer": result.answer,
                    "retrieved_chunk_ids": list(result.retrieved_chunk_ids),
                    "metrics": metrics,
                    "metadata": result.metadata,
                }
                row_results.append(row)
                if question.ground_truth is not None:
                    ragas_rows.append({
                        "user_input": question.question,
                        "retrieved_contexts": list(result.retrieved_contexts),
                        "response": result.answer,
                        "reference": question.ground_truth,
                    })
            aggregates = _aggregate(row["metrics"] for row in row_results)
            if include_ragas:
                if not ragas_rows:
                    raise ValueError("Ragas scoring requires ground_truth for at least one question.")
                aggregates.update(score_with_ragas(ragas_rows))
            strategy_reports[strategy_name] = {"metrics": aggregates, "results": row_results}

        report = {
            "run_id": run_id,
            "dataset": {"name": dataset.name, "description": dataset.description, "size": len(dataset.questions)},
            "created_at": datetime.now(UTC).isoformat(),
            "configuration": configuration or {},
            "strategies": strategy_reports,
            "conclusions": _conclusions(strategy_reports),
        }
        self.output_directory.mkdir(parents=True, exist_ok=True)
        path = self.output_directory / f"{run_id}.json"
        path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        return report


def _aggregate(rows: Any) -> dict[str, float | None]:
    metrics: dict[str, list[float]] = {}
    for row in rows:
        for name, value in row.items():
            if value is not None:
                metrics.setdefault(name, []).append(float(value))
    return {
        name: sum(values) / len(values) if values else None
        for name, values in metrics.items()
    }


def _conclusions(strategies: dict[str, Any]) -> list[str]:
    """Summarize measured retrieval metrics without inferring unmeasured quality."""
    recalls = {
        name: details["metrics"].get("recall_at_k")
        for name, details in strategies.items()
        if details["metrics"].get("recall_at_k") is not None
    }
    if not recalls:
        return ["No relevant chunk labels were supplied, so retrieval quality cannot be compared from this run."]

    best_recall = max(float(score) for score in recalls.values())
    recall_leaders = [name for name, score in recalls.items() if float(score) == best_recall]
    conclusions = [
        f"Highest measured Recall@K: {', '.join(recall_leaders)} ({best_recall:.3f})."
    ]
    hybrid_recall = recalls.get("hybrid")
    reranked_recall = recalls.get("hybrid_reranked")
    if hybrid_recall is not None and reranked_recall is not None:
        change = float(reranked_recall) - float(hybrid_recall)
        conclusions.append(
            f"Hybrid plus reranking changed measured Recall@K by {change:+.3f} versus hybrid."
        )
    if len(recalls) < len(strategies):
        conclusions.append("Some strategies had no relevance labels available for metric calculation.")
    return conclusions
