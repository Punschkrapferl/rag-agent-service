from fastapi import APIRouter, Depends
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

from app.deps import get_qdrant_client, get_embedding_model
from app.rag.models import QueryRequest, QueryResponse
from app.rag.pipeline import run_rag_pipeline
from app.rag.vectorstore import VectorStore

# No prefix here – prefix added in main.py
router = APIRouter(tags=["query"])


def get_vector_store(client: QdrantClient = Depends(get_qdrant_client)) -> VectorStore:
    return VectorStore(client)


@router.post("/query", response_model=QueryResponse)
async def query_rag(
    request: QueryRequest,
    vector_store: VectorStore = Depends(get_vector_store),
    embedder: SentenceTransformer = Depends(get_embedding_model),
) -> QueryResponse:
    return run_rag_pipeline(request, vector_store, embedder)
