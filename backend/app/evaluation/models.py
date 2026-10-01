"""Stable evaluation records shared by dataset loading and strategy runners."""

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class EvaluationQuestion:
    question: str
    ground_truth: str | None = None
    relevant_chunk_ids: tuple[str, ...] = ()
    relevant_contexts: tuple[str, ...] = ()
    expected_document: str | None = None
    expected_page: int | None = None
    expected_chunk: str | None = None


@dataclass(frozen=True, slots=True)
class EvaluationDataset:
    name: str
    description: str
    questions: tuple[EvaluationQuestion, ...]


@dataclass(frozen=True, slots=True)
class StrategyResult:
    answer: str
    retrieved_contexts: tuple[str, ...]
    retrieved_chunk_ids: tuple[str, ...]
    metadata: dict[str, Any] = field(default_factory=dict)
