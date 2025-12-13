"""
Ingestion API for the RAG Agent Service.

This module exposes endpoints for ingesting text (and optionally files)
into the vector store (Qdrant) used by the RAG pipeline.
"""

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from typing import Any, Dict, Optional

from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

from app.deps import get_qdrant_client, get_embedding_model
from app.rag.vectorstore import VectorStore
from app.rag.ingest import ingest_text  # <- uses your existing ingest_text()

router = APIRouter(tags=["ingest"], prefix="/ingest")


class IngestTextRequest(BaseModel):
    """
    Request payload for ingesting a single text document.
    """
    document_id: str
    text: str
    metadata: Optional[Dict[str, Any]] = None


class IngestTextResponse(BaseModel):
    """
    Response payload for text ingestion, including a chunk count.
    """
    document_id: str
    chunks_written: int
    collection: str = "documents"


class IngestBatchRequest(BaseModel):
    items: list[IngestTextRequest]


class IngestBatchResponse(BaseModel):
    items_ingested: int
    chunks_written: int
    collection: str = "documents"


def get_vector_store(client: QdrantClient = Depends(get_qdrant_client)) -> VectorStore:
    """
    Dependency that provides a VectorStore instance backed by Qdrant.

    Args:
        client: QdrantClient created via `get_qdrant_client`.

    Returns:
        VectorStore: Thin wrapper around Qdrant collections.
    """
    return VectorStore(client)


@router.post(
    "/text",
    summary="Ingest a text document",
    response_model=IngestTextResponse,
)
def ingest_text_endpoint(
        request: IngestTextRequest,
        vector_store: VectorStore = Depends(get_vector_store),
        embedder: SentenceTransformer = Depends(get_embedding_model),
) -> IngestTextResponse:
    """
    Ingest a single text document into the vector store.

    Steps:
    - Chunk the text into passages.
    - Embed each chunk with the shared embedding model.
    - Upsert all chunks into the configured Qdrant collection.
    - Return the number of chunks written for this document.
    """
    try:
        # Ensure document_id is present in metadata so it is stored in Qdrant
        base_meta: Dict[str, Any] = request.metadata or {}
        base_meta.setdefault("document_id", request.document_id)

        chunks_written = ingest_text(
            text=request.text,
            metadata=base_meta,
            vector_store=vector_store,
            embedder=embedder,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {exc}") from exc

    return IngestTextResponse(
        document_id=request.document_id,
        chunks_written=chunks_written,
    )


@router.post(
    "/text/batch",
    summary="Ingest multiple text documents in one request",
    response_model=IngestBatchResponse,
)
def ingest_text_batch_endpoint(
        request: IngestBatchRequest,
        vector_store: VectorStore = Depends(get_vector_store),
        embedder: SentenceTransformer = Depends(get_embedding_model),
) -> IngestBatchResponse:
    """
    Ingest multiple text documents into the vector store.

    Each item is processed using the same pipeline as single-text ingestion.
    This endpoint is intended for demos, bulk imports, and testing workflows.
    """
    total_chunks = 0

    try:
        for item in request.items:
            base_meta: Dict[str, Any] = item.metadata or {}
            base_meta.setdefault("document_id", item.document_id)

            chunks_written = ingest_text(
                text=item.text,
                metadata=base_meta,
                vector_store=vector_store,
                embedder=embedder,
            )

            total_chunks += chunks_written

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Batch ingestion failed: {exc}",
        ) from exc

    return IngestBatchResponse(
        items_ingested=len(request.items),
        chunks_written=total_chunks,
    )
