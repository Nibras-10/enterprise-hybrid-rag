"""Dense retrieval orchestration for Gemini query embeddings and Pinecone."""

from dataclasses import dataclass
import os
from uuid import UUID

from dotenv import load_dotenv

from app.embeddings.service import GeminiEmbeddingService
from app.observability.langfuse_tracer import LangfuseTracer
from app.retrieval.models import RetrievalCandidate
from app.retrieval.pinecone_store import PineconeVectorStore

load_dotenv()


@dataclass(frozen=True, slots=True)
class DenseRetrievalSettings:
    """Default result count for Pinecone dense retrieval."""

    top_k: int = 20

    def __post_init__(self) -> None:
        if self.top_k <= 0:
            raise ValueError("DENSE_TOP_K must be greater than zero.")

    @classmethod
    def from_environment(cls) -> "DenseRetrievalSettings":
        return cls(top_k=int(os.getenv("DENSE_TOP_K", "20")))


class DenseRetriever:
    """Embed a question and return scoped Pinecone dense-search candidates."""

    def __init__(
        self,
        embedding_service: GeminiEmbeddingService,
        vector_store: PineconeVectorStore,
        settings: DenseRetrievalSettings,
        tracer: LangfuseTracer | None = None,
    ) -> None:
        self.embedding_service = embedding_service
        self.vector_store = vector_store
        self.settings = settings
        self.tracer = tracer or LangfuseTracer()

    def retrieve(
        self,
        question: str,
        *,
        tenant_id: UUID,
        document_ids: list[UUID] | None = None,
        top_k: int | None = None,
        collection_id: UUID | None = None,
    ) -> list[RetrievalCandidate]:
        """Return dense candidates while applying tenant and optional document scope."""
        with self.tracer.span("rag.query_embedding") as stage:
            query_vector = self.embedding_service.embed_query(question)
            if stage is not None:
                stage.update(output={"dimension": len(query_vector)})
        with self.tracer.span(
            "rag.pinecone_dense_search",
            metadata={"top_k": self.settings.top_k if top_k is None else top_k},
        ) as stage:
            matches = self.vector_store.query(
                query_vector,
                tenant_id=tenant_id,
                top_k=self.settings.top_k if top_k is None else top_k,
                document_ids=document_ids,
                collection_id=collection_id,
            )
            if stage is not None:
                stage.update(output={"result_count": len(matches)})
        return [
            RetrievalCandidate(
                chunk_id=match.chunk_id,
                document_id=match.document_id,
                text=match.text,
                dense_score=match.score,
                retrieval_methods=("dense",),
                metadata=match.metadata,
            )
            for match in matches
        ]
