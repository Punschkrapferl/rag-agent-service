"""
Debug and inspection endpoints for the RAG Agent Service.

Currently provides:

- GET /api/debug/collections
  List the collections available in Qdrant. Useful to quickly verify
  that ingestion is working and data is present.
"""

from fastapi import APIRouter, Depends
from qdrant_client import QdrantClient

from app.deps import get_qdrant_client

router = APIRouter(tags=["debug"], prefix="/debug")


@router.get(
    "/collections",
    summary="List Qdrant collections",
)
def list_collections(client: QdrantClient = Depends(get_qdrant_client)) -> dict:
    """
    Return the list of collection names currently available in Qdrant.

    Args:
        client: QdrantClient obtained via dependency injection.

    Returns:
        dict: A dictionary with a single key `collections` containing
              the list of collection names.
    """
    response = client.get_collections()
    names = [c.name for c in response.collections]
    return {"collections": names}
