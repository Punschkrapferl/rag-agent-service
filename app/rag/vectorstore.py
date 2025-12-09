"""
Qdrant-backed vector store abstraction for the RAG Agent Service.

This module defines a thin wrapper around `QdrantClient` that is responsible for:

- Ensuring the target collection exists with the correct vector size and distance metric.
- Upserting embeddings together with arbitrary payload metadata.
- Running vector similarity search with an optional filter.
"""

from __future__ import annotations

from typing import List, Optional

from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import UnexpectedResponse
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
        settings = get_settings()
        self.client = client
        self.collection_name = collection_name or settings.qdrant_collection
        self.dim = dim
        self._ensure_collection()

        # in-memory counter for sequential IDs
        self._id_counter: Optional[int] = None

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
    # ID handling
    # -----------------------------------------------------------------------

    def _ensure_id_counter(self) -> None:
        """
        Initialize the ID counter from the current point count in Qdrant.

        If the collection is empty, we start at 0.
        Otherwise we start at `count`, so new IDs are appended after existing ones.
        """
        if self._id_counter is not None:
            return

        try:
            result = self.client.count(self.collection_name, exact=True)
        except UnexpectedResponse:
            # If count fails (e.g. collection just created but not ready),
            # fall back to 0; subsequent inserts will move the counter forward.
            self._id_counter = 0
        else:
            self._id_counter = result.count

    def _next_id(self) -> int:
        """
        Return the next sequential integer ID for a new point.
        """
        self._ensure_id_counter()
        assert self._id_counter is not None
        point_id = self._id_counter
        self._id_counter += 1
        return point_id

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

        IDs are assigned as sequential integers starting at 0 and increasing
        across the lifetime of the collection. This avoids overwriting points
        while keeping IDs human-readable in the dashboard.
        """
        if len(embeddings) != len(documents):
            raise ValueError("embeddings and documents must have same length")

        points: List[PointStruct] = []

        for embedding, doc in zip(embeddings, documents):
            point_id = self._next_id()

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
        """
        return self.client.search(
            collection_name=self.collection_name,
            query_vector=query_embedding,
            limit=top_k,
            query_filter=qfilter,
        )
