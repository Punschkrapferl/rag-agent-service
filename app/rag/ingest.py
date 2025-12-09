"""
Low-level ingestion utilities for the RAG Agent Service.

This module provides three main building blocks:

1. Text extraction
   - `extract_text_from_pdf`: read text from PDF files using pypdf.
   - `extract_text_from_html`: strip markup and extract visible text from HTML.
   - `extract_text_from_file`: dispatch helper for PDF / HTML / plain text.

2. Chunking
   - `chunk_text`: simple word-based chunker with configurable size and overlap.

3. Ingestion flows
   - `ingest_text`: end-to-end ingestion of already extracted text
     (chunk → embed → upsert to Qdrant).
   - `ingest_file`: convenience wrapper to ingest a file from disk.

These utilities are used by the ingestion API layer (`routes_ingest.py`) to
turn uploaded documents into vectorized chunks stored in the vector store.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import uuid4  # used for unique chunk IDs

from bs4 import BeautifulSoup
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

from app.rag.vectorstore import VectorStore


# ---------------------------------------------------------------------------
# Text extraction
# ---------------------------------------------------------------------------


def extract_text_from_pdf(path: Path) -> str:
    """
    Extract textual content from a PDF file.

    Each page is processed with pypdf's `extract_text`, and page texts are
    joined with newline separators.
    """
    reader = PdfReader(str(path))
    texts: List[str] = []
    for page in reader.pages:
        page_text = page.extract_text() or ""
        texts.append(page_text)
    return "\n".join(texts)


def extract_text_from_html(path: Path) -> str:
    """
    Extract visible text from an HTML file.

    - Reads the file as UTF-8 text.
    - Parses it with BeautifulSoup.
    - Removes script/style/noscript tags.
    - Collapses whitespace.
    """
    html = path.read_text(encoding="utf-8")
    soup = BeautifulSoup(html, "html.parser")

    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()

    text = soup.get_text(separator=" ")
    return " ".join(text.split())


def extract_text_from_file(path: Path) -> str:
    """
    Extract text from a file, dispatching based on file extension.
    """
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return extract_text_from_pdf(path)
    if suffix in {".html", ".htm"}:
        return extract_text_from_html(path)
    # Fallback: treat as UTF-8 plain text
    return path.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------


def chunk_text(
    text: str,
    max_tokens: int = 500,
    overlap: int = 50,
) -> List[str]:
    """
    Split text into overlapping word-based chunks.
    """
    words = text.split()
    if not words:
        return []

    chunks: List[str] = []
    start = 0
    n = len(words)

    while start < n:
        end = min(start + max_tokens, n)
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        if end >= n:
            break
        start = end - overlap

    return chunks


# ---------------------------------------------------------------------------
# Ingestion flow
# ---------------------------------------------------------------------------


def ingest_text(
    text: str,
    metadata: Optional[Dict[str, Any]],
    vector_store: VectorStore,
    embedder: SentenceTransformer,
) -> int:
    """
    End-to-end ingestion pipeline for already extracted text.

    Steps:
    1. Chunk the input text.
    2. Encode each chunk using the sentence-transformers model.
    3. Upsert embeddings + payload into the vector store.

    Each stored document payload contains:
    - `text`: the chunk text
    - `chunk_index`: position of the chunk within the document
    - `chunk_id`: globally unique ID for the chunk (for debugging/tracing)
    - all fields from `metadata` (if provided)
    """
    chunks = chunk_text(text)
    if not chunks:
        return 0

    embeddings = embedder.encode(chunks, convert_to_numpy=True).tolist()

    docs: List[Dict[str, Any]] = []
    base_meta = metadata or {}
    document_id = base_meta.get("document_id", "doc")

    for idx, chunk in enumerate(chunks):
        doc: Dict[str, Any] = {
            "text": chunk,
            "chunk_index": idx,
            "chunk_id": f"{document_id}-{idx}-{uuid4()}",
        }
        doc.update(base_meta)
        docs.append(doc)

    vector_store.upsert(embeddings, docs)
    return len(chunks)


def ingest_file(
    path: Path,
    metadata: Optional[Dict[str, Any]],
    vector_store: VectorStore,
    embedder: SentenceTransformer,
) -> int:
    """
    Convenience wrapper for ingesting a document file from disk.
    """
    text = extract_text_from_file(path)
    file_meta = {"source_path": str(path)}
    if metadata:
        file_meta.update(metadata)
    return ingest_text(text, file_meta, vector_store, embedder)
