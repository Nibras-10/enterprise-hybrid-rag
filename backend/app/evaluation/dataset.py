"""Load and validate versioned evaluation datasets stored as JSON."""

import json
from pathlib import Path
from typing import Any

from app.evaluation.models import EvaluationDataset, EvaluationQuestion


class DatasetValidationError(ValueError):
    """Raised when an evaluation dataset is malformed or empty."""


def load_dataset(path: Path) -> EvaluationDataset:
    """Read a JSON dataset with question, ground truth, and relevance labels."""
    try:
        raw: Any = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DatasetValidationError("Evaluation dataset could not be read.") from exc
    if not isinstance(raw, dict) or not isinstance(raw.get("questions"), list):
        raise DatasetValidationError("Dataset must contain a questions array.")

    questions: list[EvaluationQuestion] = []
    for index, item in enumerate(raw["questions"]):
        if not isinstance(item, dict) or not isinstance(item.get("question"), str):
            raise DatasetValidationError(f"Question {index} must contain question text.")
        question = item["question"].strip()
        if not question:
            raise DatasetValidationError(f"Question {index} must not be blank.")
        relevant_ids = item.get("relevant_chunk_ids", [])
        relevant_contexts = item.get("relevant_contexts", [])
        if not isinstance(relevant_ids, list) or not all(isinstance(value, str) for value in relevant_ids):
            raise DatasetValidationError(f"Question {index} has invalid relevant_chunk_ids.")
        if not isinstance(relevant_contexts, list) or not all(isinstance(value, str) for value in relevant_contexts):
            raise DatasetValidationError(f"Question {index} has invalid relevant_contexts.")
        ground_truth = item.get("ground_truth")
        if ground_truth is not None and not isinstance(ground_truth, str):
            raise DatasetValidationError(f"Question {index} has invalid ground_truth.")
        expected_document = item.get("expected_document")
        expected_page = item.get("expected_page")
        expected_chunk = item.get("expected_chunk")
        if expected_document is not None and not isinstance(expected_document, str):
            raise DatasetValidationError(f"Question {index} has invalid expected_document.")
        if expected_page is not None and (not isinstance(expected_page, int) or expected_page <= 0):
            raise DatasetValidationError(f"Question {index} has invalid expected_page.")
        if expected_chunk is not None and not isinstance(expected_chunk, str):
            raise DatasetValidationError(f"Question {index} has invalid expected_chunk.")
        questions.append(EvaluationQuestion(
            question=question,
            ground_truth=ground_truth,
            relevant_chunk_ids=tuple(relevant_ids),
            relevant_contexts=tuple(relevant_contexts),
            expected_document=expected_document,
            expected_page=expected_page,
            expected_chunk=expected_chunk,
        ))
    if not questions:
        raise DatasetValidationError("Dataset must contain at least one question.")
    name = raw.get("name", path.stem)
    description = raw.get("description", "")
    if not isinstance(name, str) or not isinstance(description, str):
        raise DatasetValidationError("Dataset name and description must be strings.")
    return EvaluationDataset(name=name, description=description, questions=tuple(questions))
