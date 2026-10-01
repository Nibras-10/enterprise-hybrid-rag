"""Persistent per-tenant BM25 keyword retrieval over stored document chunks."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import re
from threading import RLock
from typing import Any
from uuid import UUID

from dotenv import load_dotenv
from rank_bm25 import BM25Okapi

from app.chunking.models import ChunkDraft
from app.retrieval.models import RetrievalCandidate

load_dotenv()

_TOKEN_PATTERN = re.compile(r"\$\s?\d+(?:[,.]\d+)*|\w+(?:[./-]\w+)*", re.UNICODE)


@dataclass(frozen=True, slots=True)
class BM25Document:
    """Text and citation metadata for one indexed chunk."""

    chunk_id: str
    document_id: str
    document_version_id: str
    text: str
    page_number: int | None
    section_title: str | None
    chunk_type: str
    token_count: int
    source_filename: str
    metadata: dict[str, Any]

    @classmethod
    def from_chunk(cls, chunk: ChunkDraft, source_filename: str) -> "BM25Document":
        return cls(
            chunk_id=str(chunk.id),
            document_id=str(chunk.document_id),
            document_version_id=str(chunk.document_version_id),
            text=chunk.text,
            page_number=chunk.page_number,
            section_title=chunk.section_title,
            chunk_type=chunk.chunk_type,
            token_count=chunk.token_count,
            source_filename=source_filename,
            metadata=chunk.metadata,
        )


class BM25IndexError(RuntimeError):
    """Raised when the persistent keyword index cannot be read or updated."""


class BM25Retriever:
    """Maintain an incremental BM25 corpus isolated to one tenant."""

    def __init__(self, tenant_id: UUID, index_root: Path | None = None) -> None:
        self.tenant_id = tenant_id
        self.index_root = (index_root or Path(os.getenv("BM25_INDEX_PATH", "./data/bm25"))).resolve()
        self.index_path = self.index_root / f"tenant-{tenant_id.hex}.json"
        self._documents: dict[str, BM25Document] | None = None
        self._index: BM25Okapi | None = None
        self._corpus_ids: list[str] = []
        self._corpus_tokens: list[list[str]] = []
        self._lock = RLock()

    def index(self, chunks: Sequence[ChunkDraft], source_filename: str) -> None:
        """Replace one document's previous version and persist its current chunks."""
        if not chunks:
            return
        document_ids = {str(chunk.document_id) for chunk in chunks}
        if len(document_ids) != 1:
            raise ValueError("BM25 index updates must contain chunks from one document.")

        with self._lock:
            documents = self._load_documents()
            document_id = next(iter(document_ids))
            updated = {
                key: value for key, value in documents.items()
                if value.document_id != document_id
            }
            for chunk in chunks:
                updated[str(chunk.id)] = BM25Document.from_chunk(chunk, source_filename)
            self._save_and_rebuild(updated)

    def remove_document(self, document_id: UUID) -> None:
        """Remove one document from the tenant's cached and persistent index."""
        with self._lock:
            documents = self._load_documents()
            updated = {
                key: value for key, value in documents.items()
                if value.document_id != str(document_id)
            }
            self._save_and_rebuild(updated)

    def retrieve(
        self,
        query: str,
        top_k: int = 20,
        document_ids: list[UUID] | None = None,
    ) -> list[RetrievalCandidate]:
        """Return exact-term and lexical matches ordered by BM25 relevance."""
        if top_k <= 0:
            raise ValueError("top_k must be greater than zero.")
        if document_ids == []:
            return []
        query_tokens = self.tokenize(query)
        if not query_tokens:
            return []

        with self._lock:
            documents = self._load_documents()
            if self._index is None or not self._corpus_ids:
                return []
            scores = self._index.get_scores(query_tokens)
            query_terms = set(query_tokens)
            matching_indexes = [
                position
                for position, tokens in enumerate(self._corpus_tokens)
                if query_terms.intersection(tokens)
                and (
                    document_ids is None
                    or documents[self._corpus_ids[position]].document_id
                    in {str(document_id) for document_id in document_ids}
                )
            ]
            ranked_indexes = sorted(
                matching_indexes,
                key=lambda position: scores[position],
                reverse=True,
            )[:top_k]
            results: list[RetrievalCandidate] = []
            for rank_index in ranked_indexes:
                score = float(scores[rank_index])
                document = documents[self._corpus_ids[rank_index]]
                results.append(
                    RetrievalCandidate(
                        chunk_id=UUID(document.chunk_id),
                        document_id=UUID(document.document_id),
                        text=document.text,
                        bm25_score=score,
                        retrieval_methods=("bm25",),
                        metadata={
                            "document_version_id": document.document_version_id,
                            "page_number": document.page_number,
                            "section_title": document.section_title,
                            "chunk_type": document.chunk_type,
                            "token_count": document.token_count,
                            "source_filename": document.source_filename,
                            **document.metadata,
                        },
                    )
                )
            return results

    @staticmethod
    def tokenize(text: str) -> list[str]:
        """Normalize searchable text while retaining IDs, dates, and amounts."""
        return [token.casefold() for token in _TOKEN_PATTERN.findall(text)]

    def _load_documents(self) -> dict[str, BM25Document]:
        if self._documents is not None:
            return self._documents
        if not self.index_path.exists():
            self._documents = {}
            self._index = None
            self._corpus_ids = []
            return self._documents

        try:
            serialized = json.loads(self.index_path.read_text(encoding="utf-8"))
            if serialized.get("tenant_id") != str(self.tenant_id):
                raise BM25IndexError("The persisted BM25 index belongs to another tenant.")
            self._documents = {
                item["chunk_id"]: BM25Document(**item)
                for item in serialized.get("documents", [])
            }
        except (OSError, json.JSONDecodeError, KeyError, TypeError) as exc:
            raise BM25IndexError("The persisted BM25 index could not be loaded.") from exc
        self._rebuild()
        return self._documents

    def _save_and_rebuild(self, documents: dict[str, BM25Document]) -> None:
        self.index_root.mkdir(parents=True, exist_ok=True)
        temporary_path = self.index_path.with_suffix(".tmp")
        serialized = {
            "tenant_id": str(self.tenant_id),
            "documents": [asdict(document) for document in documents.values()],
        }
        try:
            with temporary_path.open("w", encoding="utf-8") as output:
                json.dump(serialized, output, ensure_ascii=False, separators=(",", ":"))
                output.flush()
                os.fsync(output.fileno())
            os.replace(temporary_path, self.index_path)
        except OSError as exc:
            temporary_path.unlink(missing_ok=True)
            raise BM25IndexError("The persistent BM25 index could not be saved.") from exc

        self._documents = documents
        self._rebuild()

    def _rebuild(self) -> None:
        documents = self._documents or {}
        self._corpus_ids = list(documents)
        self._corpus_tokens = [
            self.tokenize(documents[chunk_id].text) for chunk_id in self._corpus_ids
        ]
        self._index = BM25Okapi(self._corpus_tokens) if any(self._corpus_tokens) else None
