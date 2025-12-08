from typing import List

import structlog
from sentence_transformers import SentenceTransformer

from app.rag.models import QueryRequest, QueryResponse, RetrievedDocument
from app.rag.vectorstore import VectorStore

logger = structlog.get_logger()


def run_rag_pipeline(
    request: QueryRequest,
    vector_store: VectorStore,
    embedder: SentenceTransformer,
) -> QueryResponse:
    """
    Simple RAG pipeline:

    - Embed the incoming query
    - Search similar documents in Qdrant
    - Build a placeholder answer from the top hits
    """
    logger.info("rag_pipeline_start", query=request.query, top_k=request.top_k)

    # Encode query
    query_emb = embedder.encode(request.query).tolist()

    # Qdrant returns a list of ScoredPoint objects
    hits = vector_store.search(query_emb, top_k=request.top_k)

    docs: List[RetrievedDocument] = []
    for hit in hits:
        _id = str(hit.id)
        score = float(hit.score)
        payload = hit.payload or {}

        text = payload.get("text", "")
        meta = {k: v for k, v in payload.items() if k != "text"}

        docs.append(
            RetrievedDocument(
                id=_id,
                score=score,
                text=text,
                metadata=meta,
            )
        )

    # Build a trivial answer from the retrieved context
    context_snippet = "\n\n".join(d.text for d in docs[:3])

    if not context_snippet:
        answer = "No relevant documents found for your query."
    else:
        answer = (
            "This is a placeholder answer generated from retrieved context. "
            "In a real system, an LLM would summarize and reason over:\n\n"
            f"{context_snippet}"
        )

    logger.info("rag_pipeline_end", docs_returned=len(docs))

    return QueryResponse(
        query=request.query,
        answer=answer,
        documents=docs,
    )
