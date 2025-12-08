"""
Core Retrieval-Augmented Generation (RAG) pipeline.

This module defines the `run_rag_pipeline` function, which is responsible for:
- Encoding the incoming query into an embedding.
- Retrieving similar chunks from the vector store.
- Converting raw search hits into structured `RetrievedDocument` objects.
- Producing an answer string based on the retrieved context.

In its current form, the "generation" step is a simple placeholder that
concatenates the top retrieved chunks. In a production system, this is the
place where an LLM call would be integrated to generate a grounded answer.
"""

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
    Execute a simple RAG (Retrieval-Augmented Generation) pipeline.

    High-level steps:
    1. Embed the incoming query using the shared sentence-transformers model.
    2. Perform a vector search in the Qdrant-backed `VectorStore`.
    3. Map the raw search hits into `RetrievedDocument` objects.
    4. Construct a placeholder answer string from the top retrieved documents.

    Args:
        request: QueryRequest containing the user query and `top_k` value.
        vector_store: VectorStore used to perform similarity search over
            pre-ingested document chunks.
        embedder: SentenceTransformer instance used to encode the query into
            an embedding vector.

    Returns:
        QueryResponse: Includes:
            - the original query,
            - an answer string (currently a placeholder),
            - the list of retrieved documents that were used as context.
    """
    logger.info("rag_pipeline_start", query=request.query, top_k=request.top_k)

    # 1) Encode query into an embedding vector
    query_emb = embedder.encode(request.query).tolist()

    # 2) Vector search in Qdrant via VectorStore
    #    Qdrant returns a list of ScoredPoint objects
    hits = vector_store.search(query_emb, top_k=request.top_k)

    # 3) Map hits into RetrievedDocument models
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

    # 4) Construct a trivial answer from the retrieved context
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
