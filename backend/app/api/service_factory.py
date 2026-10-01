"""Construct application pipelines from validated environment settings."""

from functools import lru_cache
from uuid import UUID

from sqlalchemy.orm import Session

from app.chunking.config import ChunkingConfig
from app.chunking.semantic_chunker import SemanticChunker
from app.embeddings.service import EmbeddingSettings, GeminiEmbeddingService
from app.generation.gemini_generator import GeminiAnswerGenerator, GenerationSettings
from app.reranking.cohere_reranker import CohereReranker, RerankerSettings
from app.retrieval.dense import DenseRetriever, DenseRetrievalSettings
from app.retrieval.hybrid import HybridRetriever, HybridRetrievalSettings
from app.retrieval.pinecone_store import PineconeSettings, PineconeVectorStore
from app.services.rag_pipeline import RAGPipeline
from app.ingestion.document_ingestion import DocumentIngestionService
from app.ingestion.indexing_pipeline import DocumentIndexingPipeline
from app.ingestion.parser import UnstructuredDocumentParser
from app.core.settings import UploadSettings
from app.embeddings.service import GeminiSimilarityScorer
from app.retrieval.bm25 import BM25Retriever


@lru_cache(maxsize=8)
def _shared_clients(tenant_id: UUID) -> tuple[DenseRetriever, HybridRetriever, CohereReranker, GeminiAnswerGenerator]:
    embedding_service = GeminiEmbeddingService(EmbeddingSettings.from_environment())
    vector_store = PineconeVectorStore(PineconeSettings.from_environment())
    return (
        DenseRetriever(embedding_service, vector_store, DenseRetrievalSettings.from_environment()),
        HybridRetriever(HybridRetrievalSettings.from_environment()),
        CohereReranker(RerankerSettings.from_environment()),
        GeminiAnswerGenerator(GenerationSettings.from_environment()),
    )


def create_rag_pipeline(session: Session, tenant_id: UUID) -> RAGPipeline:
    """Create a request-scoped pipeline and reuse stateless external clients."""
    dense, hybrid, reranker, generator = _shared_clients(tenant_id)
    return RAGPipeline(
        session=session,
        dense_retriever=dense,
        hybrid_retriever=hybrid,
        reranker=reranker,
        generator=generator,
    )


def get_rag_components(
    tenant_id: UUID,
) -> tuple[DenseRetriever, HybridRetriever, CohereReranker, GeminiAnswerGenerator]:
    """Return shared retrieval/generation components for evaluation strategies."""
    return _shared_clients(tenant_id)


def create_indexing_pipeline(session: Session) -> DocumentIndexingPipeline:
    """Construct the document ingestion pipeline from the same embedding settings."""
    embedding_service = GeminiEmbeddingService(EmbeddingSettings.from_environment())
    vector_store = PineconeVectorStore(PineconeSettings.from_environment())
    parser = UnstructuredDocumentParser()
    return DocumentIndexingPipeline(
        session=session,
        ingestion_service=DocumentIngestionService(
            session,
            UploadSettings.from_environment(),
            parser=parser,
        ),
        chunker=SemanticChunker(
            ChunkingConfig.from_environment(),
            GeminiSimilarityScorer(embedding_service),
        ),
        embedding_service=embedding_service,
        vector_store=vector_store,
        bm25_retriever_factory=BM25Retriever,
    )
