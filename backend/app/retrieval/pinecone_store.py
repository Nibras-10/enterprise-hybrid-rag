"""Pinecone dense-vector index setup, storage, retrieval, and deletion."""

from dataclasses import dataclass
import os
import re
import time
from typing import Any
from uuid import UUID

from dotenv import load_dotenv

from app.chunking.models import ChunkDraft
from app.embeddings.models import EmbeddingResult

load_dotenv()

_INDEX_NAME_PATTERN = re.compile(r"^[a-z0-9](?:[a-z0-9-]{0,43}[a-z0-9])?$")


class VectorStoreError(RuntimeError):
    """Raised when Pinecone cannot create, update, or query the configured index."""


@dataclass(frozen=True, slots=True)
class PineconeSettings:
    """Connection and serverless index configuration."""

    api_key: str
    index_name: str
    dimension: int = 1536
    cloud: str = "aws"
    region: str = "us-east-1"
    metric: str = "cosine"

    def __post_init__(self) -> None:
        if not self.api_key:
            raise ValueError("PINECONE_API_KEY must be configured.")
        if not _INDEX_NAME_PATTERN.fullmatch(self.index_name):
            raise ValueError("PINECONE_INDEX_NAME must be a lowercase Pinecone index name.")
        if self.dimension <= 0:
            raise ValueError("Pinecone index dimension must be positive.")
        if self.metric not in {"cosine", "dotproduct", "euclidean"}:
            raise ValueError("Pinecone index metric is not supported.")

    @classmethod
    def from_environment(cls) -> "PineconeSettings":
        """Read vector index settings from environment variables."""
        return cls(
            api_key=os.getenv("PINECONE_API_KEY", ""),
            index_name=os.getenv("PINECONE_INDEX_NAME", ""),
            dimension=int(os.getenv("EMBEDDING_DIMENSION", "1536")),
            cloud=os.getenv("PINECONE_CLOUD", "aws"),
            region=os.getenv("PINECONE_REGION", "us-east-1"),
            metric=os.getenv("PINECONE_METRIC", "cosine"),
        )


@dataclass(frozen=True, slots=True)
class VectorMatch:
    """Dense-search result with its source chunk and Pinecone score."""

    chunk_id: UUID
    document_id: UUID
    score: float
    text: str
    metadata: dict[str, Any]


class PineconeVectorStore:
    """Manage one configured Pinecone serverless index."""

    def __init__(self, settings: PineconeSettings, client: Any | None = None) -> None:
        self.settings = settings
        self._client = client
        self._index: Any | None = None

    @staticmethod
    def namespace_for(tenant_id: UUID, collection_id: UUID | None = None) -> str:
        """Return a deterministic namespace from trusted UUID identifiers."""
        namespace = f"tenant-{tenant_id.hex}"
        if collection_id is not None:
            namespace += f"-collection-{collection_id.hex}"
        return namespace

    def ensure_index(self) -> None:
        """Create the serverless index when absent and reject incompatible indexes."""
        from pinecone import ServerlessSpec

        client = self._get_client()
        try:
            if not client.has_index(self.settings.index_name):
                client.create_index(
                    name=self.settings.index_name,
                    dimension=self.settings.dimension,
                    metric=self.settings.metric,
                    spec=ServerlessSpec(cloud=self.settings.cloud, region=self.settings.region),
                )

            description = self._wait_for_index(client)
            if getattr(description, "dimension", None) != self.settings.dimension:
                raise VectorStoreError("The Pinecone index dimension does not match EMBEDDING_DIMENSION.")
            if getattr(description, "metric", None) != self.settings.metric:
                raise VectorStoreError("The Pinecone index metric does not match PINECONE_METRIC.")

            host = getattr(description, "host", None)
            if not host:
                raise VectorStoreError("Pinecone did not return a host for the configured index.")
            self._index = client.Index(host=host)
        except VectorStoreError:
            raise
        except Exception as exc:
            raise VectorStoreError("The Pinecone index could not be prepared.") from exc

    def upsert_chunks(
        self,
        chunks: list[ChunkDraft],
        embeddings: list[EmbeddingResult],
        *,
        source_filename: str,
        tenant_id: UUID,
        collection_id: UUID | None = None,
    ) -> None:
        """Upsert chunks and explicit metadata into a tenant/collection namespace."""
        if len(chunks) != len(embeddings):
            raise ValueError("Every chunk must have exactly one embedding.")

        records: list[dict[str, Any]] = []
        for chunk, embedding in zip(chunks, embeddings, strict=True):
            if chunk.id != embedding.id:
                raise ValueError("Embedding IDs must match their source chunk IDs.")
            if len(embedding.values) != self.settings.dimension:
                raise ValueError("Embedding dimension does not match the Pinecone index configuration.")
            records.append(
                {
                    "id": str(chunk.id),
                    "values": list(embedding.values),
                    "metadata": self._vector_metadata(
                        chunk,
                        source_filename=source_filename,
                        tenant_id=tenant_id,
                    ),
                }
            )

        if not records:
            return
        index = self._get_index()
        try:
            namespace = self.namespace_for(tenant_id, collection_id)
            for start in range(0, len(records), 100):
                index.upsert(
                    vectors=records[start : start + 100],
                    namespace=namespace,
                )
        except Exception as exc:
            raise VectorStoreError("Pinecone could not save the document vectors.") from exc

    def query(
        self,
        vector: list[float],
        *,
        tenant_id: UUID,
        top_k: int,
        document_ids: list[UUID] | None = None,
        collection_id: UUID | None = None,
    ) -> list[VectorMatch]:
        """Query dense neighbors with trusted tenant and optional document filters."""
        if len(vector) != self.settings.dimension:
            raise ValueError("Query embedding dimension does not match the Pinecone index.")
        if top_k <= 0:
            raise ValueError("top_k must be greater than zero.")
        if document_ids == []:
            return []

        filters: dict[str, Any] = {"tenant_id": {"$eq": str(tenant_id)}}
        if document_ids:
            filters["document_id"] = {"$in": [str(document_id) for document_id in document_ids]}

        try:
            response = self._get_index().query(
                vector=vector,
                top_k=top_k,
                namespace=self.namespace_for(tenant_id, collection_id),
                filter=filters,
                include_metadata=True,
            )
            return [self._to_match(match) for match in response.matches]
        except VectorStoreError:
            raise
        except Exception as exc:
            raise VectorStoreError("Pinecone dense retrieval failed.") from exc

    def delete_document(
        self,
        document_id: UUID,
        *,
        tenant_id: UUID,
        collection_id: UUID | None = None,
        document_version_id: UUID | None = None,
    ) -> None:
        """Remove stale vectors for a document or one superseded version."""
        filters: dict[str, Any] = {
            "tenant_id": {"$eq": str(tenant_id)},
            "document_id": {"$eq": str(document_id)},
        }
        if document_version_id is not None:
            filters["document_version_id"] = {"$eq": str(document_version_id)}
        try:
            self._get_index().delete(
                filter=filters,
                namespace=self.namespace_for(tenant_id, collection_id),
            )
        except Exception as exc:
            raise VectorStoreError("Pinecone could not remove the requested document vectors.") from exc

    def _get_index(self) -> Any:
        if self._index is None:
            self.ensure_index()
        return self._index

    def _get_client(self) -> Any:
        if self._client is None:
            from pinecone import Pinecone

            self._client = Pinecone(api_key=self.settings.api_key)
        return self._client

    def _wait_for_index(self, client: Any) -> Any:
        deadline = time.monotonic() + 120
        while True:
            description = client.describe_index(self.settings.index_name)
            status = getattr(description, "status", None)
            ready = status.get("ready", False) if isinstance(status, dict) else getattr(status, "ready", False)
            if ready:
                return description
            if time.monotonic() >= deadline:
                raise VectorStoreError("The Pinecone index did not become ready before the timeout.")
            time.sleep(1)

    @staticmethod
    def _vector_metadata(
        chunk: ChunkDraft,
        *,
        source_filename: str,
        tenant_id: UUID,
    ) -> dict[str, Any]:
        metadata: dict[str, Any] = {
            "tenant_id": str(tenant_id),
            "document_id": str(chunk.document_id),
            "document_version_id": str(chunk.document_version_id),
            "chunk_id": str(chunk.id),
            "chunk_type": chunk.chunk_type,
            "source_filename": source_filename,
            "text": chunk.text,
            "token_count": chunk.token_count,
        }
        if chunk.page_number is not None:
            metadata["page_number"] = chunk.page_number
        if chunk.section_title:
            metadata["section_title"] = chunk.section_title
        return metadata

    @staticmethod
    def _to_match(match: Any) -> VectorMatch:
        metadata = getattr(match, "metadata", None) or {}
        try:
            return VectorMatch(
                chunk_id=UUID(str(metadata["chunk_id"])),
                document_id=UUID(str(metadata["document_id"])),
                score=float(match.score),
                text=str(metadata["text"]),
                metadata=dict(metadata),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise VectorStoreError("Pinecone returned a vector without required source metadata.") from exc
