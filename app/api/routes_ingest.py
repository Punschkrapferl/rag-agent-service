"""
Ingestion API for the RAG Agent Service.

This module exposes endpoints to:
- Ingest raw text into the vector store (`POST /api/ingest/text`)
- Ingest multiple text documents in one request (`POST /api/ingest/text/batch`)
- Ingest files (PDF, HTML, TXT) into the vector store (`POST /api/ingest/file`)

Both text and file ingestion:
- Chunk the input
- Embed the chunks
- Upsert the vectors + payload into Qdrant via `VectorStore`
"""

from __future__ import annotations

import tempfile
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional

import structlog
from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

from app.deps import get_qdrant_client
from app.rag.ingest import ingest_file as ingest_file_flow
from app.rag.ingest import ingest_text as ingest_text_flow
from app.rag.vectorstore import VectorStore

# Router is mounted under /api in app.main
router = APIRouter(tags=["ingest"])
logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class IngestTextRequest(BaseModel):
    """
    Request body for ingesting raw text.

    Attributes:
        document_id: Logical identifier for the document (e.g. database id).
        text: Raw text content to be chunked and embedded.
        metadata: Optional metadata to attach to each stored chunk payload.
    """

    document_id: str
    text: str
    metadata: Optional[Dict[str, Any]] = None


class IngestTextResponse(BaseModel):
    """
    Response body for text ingestion.
    """

    document_id: str
    chunks_indexed: int


class IngestTextBatchItemResult(BaseModel):
    """
    Per-document result entry for batch text ingestion.
    """

    document_id: str
    chunks_indexed: int


class IngestTextBatchRequest(BaseModel):
    """
    Request body for batch text ingestion.

    Attributes:
        items: List of individual text ingestion requests.
    """

    items: List[IngestTextRequest]


class IngestTextBatchResponse(BaseModel):
    """
    Response body for batch text ingestion.

    Attributes:
        results: Per-document ingestion results.
        total_chunks_indexed: Sum of all chunks indexed across the batch.
    """

    results: List[IngestTextBatchItemResult]
    total_chunks_indexed: int


class IngestFileResponse(BaseModel):
    """
    Response body for file ingestion.
    """

    filename: str
    document_id: Optional[str]
    chunks_indexed: int
    detected_type: str


# ---------------------------------------------------------------------------
# Dependencies
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def get_embedder() -> SentenceTransformer:
    """
    Return a singleton sentence-transformers model instance.

    The cache ensures that the embedding model is loaded only once per process,
    which is important for both performance and memory usage.
    """
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


def get_vector_store() -> VectorStore:
    """
    Construct a VectorStore instance backed by Qdrant.

    The underlying QdrantClient is provided by `get_qdrant_client`, which
    reads host/port configuration from environment variables via `Settings`.
    """
    client = get_qdrant_client()
    return VectorStore(client)


# ---------------------------------------------------------------------------
# Single-text ingestion
# ---------------------------------------------------------------------------


@router.post(
    "/ingest/text",
    summary="Ingest raw text",
    response_model=IngestTextResponse,
)
def ingest_text_endpoint(
    payload: IngestTextRequest,
    vector_store: VectorStore = Depends(get_vector_store),
    embedder: SentenceTransformer = Depends(get_embedder),
) -> IngestTextResponse:
    """
    Ingest a single block of raw text into the vector store.

    The text is:
    - Chunked into smaller segments
    - Embedded with the configured sentence-transformers model
    - Upserted into Qdrant together with the provided metadata

    Args:
        payload: Text + metadata to ingest.
        vector_store: Qdrant-backed vector store instance (dependency).
        embedder: Sentence-transformers model used to compute embeddings.

    Returns:
        IngestTextResponse: The document id and number of chunks stored.
    """
    meta: Dict[str, Any] = {"document_id": payload.document_id}
    if payload.metadata:
        meta.update(payload.metadata)

    chunks = ingest_text_flow(
        text=payload.text,
        metadata=meta,
        vector_store=vector_store,
        embedder=embedder,
    )

    return IngestTextResponse(
        document_id=payload.document_id,
        chunks_indexed=chunks,
    )


# ---------------------------------------------------------------------------
# Batch text ingestion
# ---------------------------------------------------------------------------


@router.post(
    "/ingest/text/batch",
    summary="Ingest multiple text documents",
    response_model=IngestTextBatchResponse,
)
def ingest_text_batch_endpoint(
    payload: IngestTextBatchRequest,
    vector_store: VectorStore = Depends(get_vector_store),
    embedder: SentenceTransformer = Depends(get_embedder),
) -> IngestTextBatchResponse:
    """
    Ingest multiple text documents in a single request.

    Each item in `payload.items` is processed like the single `/ingest/text`
    endpoint:

    - Chunk text
    - Embed chunks
    - Upsert into Qdrant via `VectorStore`

    Args:
        payload: Batch of text ingestion requests.
        vector_store: Shared vector store instance.
        embedder: Shared embedding model instance.

    Returns:
        IngestTextBatchResponse: Per-document results and total chunk count.
    """
    results: List[IngestTextBatchItemResult] = []
    total_chunks = 0

    for item in payload.items:
        meta: Dict[str, Any] = {"document_id": item.document_id}
        if item.metadata:
            meta.update(item.metadata)

        chunks = ingest_text_flow(
            text=item.text,
            metadata=meta,
            vector_store=vector_store,
            embedder=embedder,
        )

        results.append(
            IngestTextBatchItemResult(
                document_id=item.document_id,
                chunks_indexed=chunks,
            )
        )
        total_chunks += chunks

    return IngestTextBatchResponse(
        results=results,
        total_chunks_indexed=total_chunks,
    )


# ---------------------------------------------------------------------------
# File ingestion
# ---------------------------------------------------------------------------


@router.post(
    "/ingest/file",
    summary="Ingest a document file",
    response_model=IngestFileResponse,
)
async def ingest_file_endpoint(
    file: UploadFile = File(..., description="PDF, HTML/HTM, or plain text file"),
    document_id: Optional[str] = None,
    vector_store: VectorStore = Depends(get_vector_store),
    embedder: SentenceTransformer = Depends(get_embedder),
) -> IngestFileResponse:
    """
    Ingest a document file from an upload.

    Supported file types:
    - `.pdf`
    - `.html` / `.htm`
    - `.txt`

    The file is stored as a temporary file on disk, parsed into text, chunked,
    embedded, and then upserted into Qdrant.

    Args:
        file: Uploaded file to ingest.
        document_id: Optional logical id to associate with all chunks.
        vector_store: Qdrant-backed vector store.
        embedder: Sentence-transformers model used for embeddings.

    Raises:
        HTTPException: If the file type is not supported.

    Returns:
        IngestFileResponse: Basic information about the ingested file and
        the number of chunks stored.
    """
    suffix = Path(file.filename).suffix.lower()
    if suffix not in {".pdf", ".html", ".htm", ".txt"}:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {suffix}",
        )

    # Store temporarily on disk so pypdf / bs4 can process it
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        raw = await file.read()
        tmp.write(raw)
        tmp_path = Path(tmp.name)

    chunks: int = 0

    try:
        meta: Dict[str, Any] = {"filename": file.filename}
        if document_id:
            meta["document_id"] = document_id

        chunks = ingest_file_flow(
            path=tmp_path,
            metadata=meta,
            vector_store=vector_store,
            embedder=embedder,
        )
    finally:
        try:
            tmp_path.unlink(missing_ok=True)
        except FileNotFoundError:
            # File already removed or never created — safe to ignore.
            pass
        except OSError:
            # Log but do not fail the request if cleanup fails.
            logger.warning(
                "Failed to delete temporary file",
                tmp_path=str(tmp_path),
            )

    return IngestFileResponse(
        filename=file.filename,
        document_id=document_id,
        chunks_indexed=chunks,
        detected_type=suffix,
    )
