from __future__ import annotations

import tempfile
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import structlog

from app.deps import get_qdrant_client
from app.rag.vectorstore import VectorStore
from app.rag.ingest import ingest_text as ingest_text_flow
from app.rag.ingest import ingest_file as ingest_file_flow

# No prefix here – prefix will be added in main.py via include_router(...)
router = APIRouter(tags=["ingest"])
logger = structlog.get_logger(__name__)


class IngestTextRequest(BaseModel):
    document_id: str
    text: str
    metadata: Optional[Dict[str, Any]] = None


@lru_cache(maxsize=1)
def get_embedder() -> SentenceTransformer:
    # Small, production-friendly model already in requirements
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


def get_vector_store() -> VectorStore:
    client = get_qdrant_client()
    return VectorStore(client)


@router.post("/ingest/text")
def ingest_text_endpoint(
    payload: IngestTextRequest,
    vector_store: VectorStore = Depends(get_vector_store),
    embedder: SentenceTransformer = Depends(get_embedder),
):
    meta: Dict[str, Any] = {"document_id": payload.document_id}
    if payload.metadata:
        meta.update(payload.metadata)

    chunks = ingest_text_flow(
        text=payload.text,
        metadata=meta,
        vector_store=vector_store,
        embedder=embedder,
    )

    return {
        "document_id": payload.document_id,
        "chunks_indexed": chunks,
    }


@router.post("/ingest/file")
async def ingest_file_endpoint(
    file: UploadFile = File(...),
    document_id: Optional[str] = None,
    vector_store: VectorStore = Depends(get_vector_store),
    embedder: SentenceTransformer = Depends(get_embedder),
):
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
            pass
        except OSError:
            logger.warning(
                "Failed to delete temporary file",
                tmp_path=str(tmp_path),
            )

    return {
        "filename": file.filename,
        "document_id": document_id,
        "chunks_indexed": chunks,
        "detected_type": suffix,
    }
