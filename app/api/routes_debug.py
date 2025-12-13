"""
Debug and inspection endpoints for the RAG Agent Service.

Currently provides:

- GET /api/debug/collections
  List the collections available in Qdrant. Useful to quickly verify
  that ingestion is working and data is present.

- DELETE /api/debug/qdrant/demo-vectors
  Remove demo/test vectors from Qdrant based on payload metadata.
"""

from fastapi import APIRouter, Depends
from qdrant_client import QdrantClient
from qdrant_client.http.models import Filter, FieldCondition, MatchValue, FilterSelector

from app.deps import get_qdrant_client
from app.config import get_settings

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


def _demo_filter() -> FilterSelector:
    """
    Common filter used to delete demo vectors.

    By default this matches points with payload field `source = "demo"`.
    Extend this if you also want to remove other demo tags such as
    "demo-single" / "demo-batch" / "batch-demo".
    """
    return FilterSelector(
        filter=Filter(
            must=[
                FieldCondition(
                    key="source",
                    match=MatchValue(value="demo"),
                ),
                # Example of broader matching if you want to delete all demo variants:
                # FieldCondition(
                #     key="source",
                #     match=MatchAny(
                #         any=["demo", "demo-single", "demo-batch", "batch-demo"]
                #     ),
                # ),
            ]
        )
    )


@router.delete(
    "/qdrant/demo-vectors",
    summary="Delete demo vectors from Qdrant",
)
def delete_demo_vectors(
    client: QdrantClient = Depends(get_qdrant_client),
) -> dict:
    """
    Delete all vectors tagged as demo/test data from Qdrant.

    This endpoint removes points whose payload contains `source = "demo"`,
    allowing cleanup of development data without resetting the collection.
    """
    settings = get_settings()
    client.delete(
        collection_name=settings.qdrant_collection,
        points_selector=_demo_filter(),
    )

    return {
        "status": "ok",
        "deleted": "vectors with source=demo",
        "collection": settings.qdrant_collection,
    }
