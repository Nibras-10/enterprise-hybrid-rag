"""End-to-end document parsing, chunking, embedding, and indexing orchestration."""

from collections.abc import Callable
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.chunking.models import ChunkDraft
from app.chunking.semantic_chunker import SemanticChunker
from app.embeddings.models import EmbeddingInput
from app.embeddings.service import GeminiEmbeddingService
from app.ingestion.document_ingestion import DocumentIngestionService
from app.ingestion.lifecycle import DocumentLifecycleService, DocumentStatus, IngestionStatus
from app.ingestion.llama_index_bridge import LlamaIndexDocumentBridge
from app.repositories.chunk_repository import ChunkRepository
from app.retrieval.bm25 import BM25Retriever
from app.retrieval.pinecone_store import PineconeVectorStore
from app.models import Chunk, Document, DocumentVersion, IngestionJob


class DocumentIndexingPipeline:
    """Run each ingestion phase and persist status as work becomes durable."""

    def __init__(
        self,
        *,
        session: Session,
        ingestion_service: DocumentIngestionService,
        chunker: SemanticChunker,
        embedding_service: GeminiEmbeddingService,
        vector_store: PineconeVectorStore,
        bm25_retriever_factory: Callable[[UUID], BM25Retriever] | None = None,
    ) -> None:
        self.session = session
        self.ingestion_service = ingestion_service
        self.chunker = chunker
        self.embedding_service = embedding_service
        self.vector_store = vector_store
        self.bm25_retriever_factory = bm25_retriever_factory or BM25Retriever

    def ingest(
        self,
        document_id: UUID,
        *,
        tenant_id: UUID,
        collection_id: UUID | None = None,
    ) -> list[ChunkDraft]:
        """Parse and index a queued document, or persist a clear failed lifecycle state."""
        document_version_id: UUID | None = None
        vectors_upserted = False
        try:
            parsed = self.ingestion_service.parse(document_id)
            document = self.session.get(Document, document_id)
            if document is None:
                raise LookupError("Document was not found after parsing.")
            version = max(document.versions, key=lambda item: item.version)
            document_version_id = version.id
            lifecycle = DocumentLifecycleService(self.session)

            nodes = LlamaIndexDocumentBridge.to_nodes(parsed.elements)
            elements = LlamaIndexDocumentBridge.to_elements(nodes)
            drafts = self.chunker.chunk(elements, version.id)
            ChunkRepository(self.session).replace_version_chunks(version.id, drafts)
            version.status = DocumentStatus.CHUNKED
            lifecycle.advance(document, DocumentStatus.CHUNKED)

            embedding_inputs = [
                EmbeddingInput(id=draft.id, text=draft.text, title=document.title)
                for draft in drafts
            ]
            embeddings = self.embedding_service.embed_documents(embedding_inputs)
            version.status = DocumentStatus.EMBEDDED
            lifecycle.advance(document, DocumentStatus.EMBEDDED)

            vectors_upserted = bool(drafts)
            self.vector_store.upsert_chunks(
                drafts,
                embeddings,
                source_filename=document.filename,
                tenant_id=tenant_id,
                collection_id=collection_id,
            )
            bm25_retriever = self.bm25_retriever_factory(tenant_id)
            if drafts:
                bm25_retriever.index(drafts, document.filename)
            else:
                bm25_retriever.remove_document(document.id)

            chunks = self.session.scalars(
                select(Chunk).where(Chunk.document_version_id == version.id)
            ).all()
            for chunk in chunks:
                chunk.pinecone_vector_id = str(chunk.id)
            version.status = DocumentStatus.INDEXED
            lifecycle.advance(document, DocumentStatus.INDEXED)

            version.status = DocumentStatus.READY
            lifecycle.advance(document, DocumentStatus.READY)
            return drafts
        except Exception as exc:
            self.session.rollback()
            if vectors_upserted and document_version_id is not None:
                try:
                    self.vector_store.delete_document(
                        document_id,
                        tenant_id=tenant_id,
                        collection_id=collection_id,
                        document_version_id=document_version_id,
                    )
                except Exception:
                    pass
                try:
                    self.bm25_retriever_factory(tenant_id).remove_document(document_id)
                except Exception:
                    pass
            self._record_failure(document_id, exc)
            raise

    def _record_failure(self, document_id: UUID, error: Exception) -> None:
        """Store a safe error summary without copying external response bodies."""
        document = self.session.get(Document, document_id)
        if document is None or document.status in {DocumentStatus.FAILED, DocumentStatus.READY}:
            return
        job = self.session.scalar(
            select(IngestionJob)
            .where(IngestionJob.document_id == document_id)
            .order_by(IngestionJob.created_at.desc())
        )
        if job is None:
            return
        summary = f"Document processing failed ({type(error).__name__})."
        try:
            DocumentLifecycleService(self.session).fail(document, job, summary)
        except Exception:
            self.session.rollback()
