"""
Shared dependency providers for the RAG Agent Service.

This module exposes factory functions for:

- `QdrantClient`: low-level client used by `VectorStore`.
- `SentenceTransformer`: embedding model used for both ingestion and querying.

Both factories are wrapped with `lru_cache(maxsize=1)` so that:
- The Qdrant connection is created only once per process.
- The embedding model is loaded into memory only once per process.

These functions are used both directly (e.g. in `VectorStore`) and as
FastAPI dependencies (via `Depends(...)`) in the API layer.
"""

from functools import lru_cache

from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

from app.config import get_settings


@lru_cache(maxsize=1)
def get_qdrant_client() -> QdrantClient:
    """
    Create and cache a QdrantClient instance.

    The host and port are read from application settings, which in turn can
    be configured via environment variables or a `.env` file.

    Returns:
        QdrantClient: Singleton client used to talk to the Qdrant server.
    """
    settings = get_settings()
    return QdrantClient(
        host=settings.qdrant_host,
        port=settings.qdrant_port,
    )


@lru_cache(maxsize=1)
def get_embedding_model() -> SentenceTransformer:
    """
    Create and cache the sentence-transformers embedding model.

    The model name is taken from application settings (`embedding_model_name`),
    allowing it to be adjusted per environment without code changes.

    Returns:
        SentenceTransformer: Singleton embedding model instance.
    """
    settings = get_settings()
    return SentenceTransformer(settings.embedding_model_name)
