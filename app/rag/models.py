from pydantic import BaseModel
from typing import List, Any


class QueryRequest(BaseModel):
    query: str
    top_k: int = 5


class RetrievedDocument(BaseModel):
    id: str
    score: float
    text: str
    metadata: dict[str, Any] = {}


class QueryResponse(BaseModel):
    query: str
    answer: str
    documents: List[RetrievedDocument]
