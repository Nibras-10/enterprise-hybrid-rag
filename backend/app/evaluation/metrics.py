"""Deterministic retrieval metrics plus an optional Ragas scoring adapter."""

from collections.abc import Sequence
import os
from typing import Any


def retrieval_metrics(retrieved_ids: Sequence[str], relevant_ids: Sequence[str], k: int) -> dict[str, float | None]:
    """Compute precision@k and recall@k when relevance labels are available."""
    if k <= 0:
        raise ValueError("k must be greater than zero.")
    if not relevant_ids:
        return {"precision_at_k": None, "recall_at_k": None, "reciprocal_rank": None}
    relevant = set(relevant_ids)
    selected = list(retrieved_ids[:k])
    hits = sum(item in relevant for item in selected)
    first_rank = next((rank for rank, item in enumerate(selected, 1) if item in relevant), None)
    return {
        "precision_at_k": hits / k,
        "recall_at_k": hits / len(relevant),
        "reciprocal_rank": 1 / first_rank if first_rank else 0.0,
    }


def score_with_ragas(rows: list[dict[str, Any]]) -> dict[str, float]:
    """Evaluate response faithfulness and relevance with the installed Ragas API.

    Input rows use Ragas' standard user_input/retrieved_contexts/response/reference columns.
    The evaluator model/provider is configured by Ragas environment variables.
    """
    try:
        from datasets import Dataset
        from ragas import evaluate
        from ragas.embeddings import GoogleEmbeddings
        from ragas.llms import llm_factory
        from ragas.metrics import AnswerRelevancy, ContextPrecision, ContextRecall, Faithfulness
        from google import genai
    except ImportError as exc:
        raise RuntimeError("Install the evaluation dependencies to run Ragas scoring.") from exc
    api_key = os.getenv("GEMINI_API_KEY", "")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is required for Gemini-backed Ragas evaluation.")
    client = genai.Client(api_key=api_key)
    model = os.getenv("RAGAS_EVALUATOR_MODEL") or os.getenv("GEMINI_MODEL") or "gemini-3.8-flash"
    llm = llm_factory(model, provider="google", client=client)
    embeddings = GoogleEmbeddings(
        client=client,
        model=os.getenv("GEMINI_EMBEDDING_MODEL") or "gemini-embedding-2",
    )
    result = evaluate(
        dataset=Dataset.from_list([
            {
                "question": row["user_input"],
                "answer": row["response"],
                "contexts": row["retrieved_contexts"],
                "ground_truth": row["reference"],
            }
            for row in rows
        ]),
        metrics=[
            Faithfulness(llm=llm),
            AnswerRelevancy(llm=llm, embeddings=embeddings),
            ContextPrecision(llm=llm),
            ContextRecall(llm=llm),
        ],
    )
    return {key: float(value) for key, value in result.items() if isinstance(value, (int, float))}
