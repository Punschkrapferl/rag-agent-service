"""
Pydantic models used by the RAG query API.

These models define the request and response schema for `/api/query`:

- `QueryRequest`: input to the RAG pipeline (user query + retrieval depth).
- `RetrievedDocument`: a single retrieved chunk with score and metadata.
- `QueryResponse`: final answer plus the list of retrieved documents.
"""

from typing import Any, List

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    """
    Request body for a RAG query.

    Attributes:
        query: Natural-language query string provided by the client.
        top_k: Maximum number of documents to retrieve from the vector store.
    """

    query: str
    top_k: int = 5


class RetrievedDocument(BaseModel):
    """
    Representation of a single retrieved document/chunk.

    Attributes:
        id: Identifier of the stored point in the vector store (e.g. Qdrant id).
        score: Similarity score returned by the vector search (e.g. cosine distance).
        text: Raw text content of the retrieved chunk.
        metadata: Arbitrary metadata attached to the chunk at ingestion time
                  (document id, filename, source, etc.).
    """

    id: str
    score: float
    text: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class QueryResponse(BaseModel):
    """
    Response body for a RAG query.

    Attributes:
        query: Echo of the original query for traceability.
        answer: Generated answer text based on the retrieved documents.
        documents: List of retrieved documents/chunks used as context.
    """

    query: str
    answer: str
    documents: List[RetrievedDocument]
