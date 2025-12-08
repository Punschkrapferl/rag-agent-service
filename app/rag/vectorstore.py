from __future__ import annotations

from typing import Any, List, Optional

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter

from app.config import get_settings


class VectorStore:
    """
    Thin wrapper around QdrantClient.

    Responsibilities:
    - Ensure collection exists with correct vector size and distance metric.
    - Upsert points (embeddings + payload).
    - Search by vector with optional filter.
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

    def _ensure_collection(self) -> None:
        if not self.client.collection_exists(self.collection_name):
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self.dim,
                    distance=Distance.COSINE,
                ),
            )

    # -------- Ingestion API --------

    def upsert(
        self,
        embeddings: List[List[float]],
        documents: List[dict],
    ) -> None:
        """
        Upsert embeddings + payload into Qdrant.

        Each document dict may contain:
        - "id" (optional): custom id, otherwise index is used
        - "text": chunk text
        - any additional metadata fields
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

    # -------- Query API --------

    def search(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        qfilter: Optional[Filter] = None,
    ):
        """
        Vector search wrapper used by the /query endpoint.
        """
        return self.client.search(
            collection_name=self.collection_name,
            query_vector=query_embedding,
            limit=top_k,
            query_filter=qfilter,
        )
