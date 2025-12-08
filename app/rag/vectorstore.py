"""
Qdrant-backed vector store abstraction for the RAG Agent Service.

This module defines a thin wrapper around `QdrantClient` that is responsible for:

- Ensuring the target collection exists with the correct vector size and distance metric.
- Upserting embeddings together with arbitrary payload metadata.
- Running vector similarity search with an optional filter.

By centralizing this logic, the rest of the application (ingestion, query API,
pipeline) can work with a simple `VectorStore` interface instead of directly
handling Qdrant client details.
"""

from __future__ import annotations

from typing import Any, List, Optional

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, Filter, PointStruct, VectorParams

from app.config import get_settings


class VectorStore:
    """
    Wrapper around `QdrantClient` for a single collection.

    Responsibilities:
    - Ensure the collection exists and is configured with the expected dimension
      and distance metric.
    - Upsert points (embedding vectors + payload dictionaries).
    - Perform vector similarity search with an optional Qdrant `Filter`.
    """

    def __init__(
        self,
        client: QdrantClient,
        collection_name: Optional[str] = None,
        dim: int = 384,
    ) -> None:
        """
        Initialize a VectorStore.

        Args:
            client: Initialized `QdrantClient` instance.
            collection_name: Name of the Qdrant collection to use. If not
                provided, the value from application settings is used.
            dim: Dimensionality of the embedding vectors. Must match the
                embedding model output size.
        """
        settings = get_settings()
        self.client = client
        self.collection_name = collection_name or settings.qdrant_collection
        self.dim = dim
        self._ensure_collection()

    def _ensure_collection(self) -> None:
        """
        Create the collection in Qdrant if it does not already exist.

        The collection is created with:
        - Vector size equal to `self.dim`
        - Cosine distance as the similarity metric
        """
        if not self.client.collection_exists(self.collection_name):
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self.dim,
                    distance=Distance.COSINE,
                ),
            )

    # -----------------------------------------------------------------------
    # Ingestion API
    # -----------------------------------------------------------------------

    def upsert(
        self,
        embeddings: List[List[float]],
        documents: List[dict],
    ) -> None:
        """
        Upsert embeddings and their payloads into the Qdrant collection.

        For each document, a corresponding vector is upserted as a `PointStruct`.
        The payload is stored as-is, enabling flexible metadata filtering later.

        Each document dict may contain:
        - "id" (optional): custom point id; if missing, the index in the list is used.
        - "text": chunk text.
        - Any additional metadata fields (e.g. document_id, filename, etc.).

        Args:
            embeddings: List of embedding vectors, one per document.
            documents: List of payload dictionaries, one per embedding.

        Raises:
            ValueError: If the number of embeddings does not match the number
                of documents.
        """
        if len(embeddings) != len(documents):
            raise ValueError("embeddings and documents must have same length")

        points: List[PointStruct] = []
        for idx, (embedding, doc) in enumerate(zip(embeddings, documents)):
            point_id: Any = doc.get("id", idx)
            points.append(
                PointStruct(
                    id=point_id,
                    vector=embedding,
                    payload=doc,
                )
            )

        self.client.upsert(
            collection_name=self.collection_name,
            points=points,
            wait=True,
        )

    # -----------------------------------------------------------------------
    # Query API
    # -----------------------------------------------------------------------

    def search(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        qfilter: Optional[Filter] = None,
    ):
        """
        Perform a vector similarity search against the collection.

        Args:
            query_embedding: Embedding vector representing the query.
            top_k: Maximum number of nearest neighbors to return.
            qfilter: Optional Qdrant `Filter` to restrict the search to a subset
                of points based on payload conditions.

        Returns:
            A list of `ScoredPoint` objects as returned by `QdrantClient.search`.
        """
        return self.client.search(
            collection_name=self.collection_name,
            query_vector=query_embedding,
            limit=top_k,
            query_filter=qfilter,
        )
