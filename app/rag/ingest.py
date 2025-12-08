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

    Args:
        path: Filesystem path to the PDF file.

    Returns:
        A single string containing the concatenated text of all pages.
        If a page has no extractable text, an empty string is used for that page.
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

    This function:
    - Reads the file as UTF-8 text.
    - Parses it with BeautifulSoup.
    - Removes script/style/noscript tags.
    - Collapses whitespace.

    Args:
        path: Filesystem path to the HTML/HTM file.

    Returns:
        A whitespace-normalized string containing visible text.
    """
    html = path.read_text(encoding="utf-8")
    soup = BeautifulSoup(html, "html.parser")

    # Remove non-content tags
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()

    text = soup.get_text(separator=" ")
    # Normalize whitespace
    return " ".join(text.split())


def extract_text_from_file(path: Path) -> str:
    """
    Extract text from a file, dispatching based on file extension.

    Supported:
    - `.pdf`  → `extract_text_from_pdf`
    - `.html` / `.htm` → `extract_text_from_html`
    - anything else → treated as UTF-8 plain text

    Args:
        path: Filesystem path to the document.

    Returns:
        Extracted text as a string.
    """
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return extract_text_from_pdf(path)
    if suffix in {".html", ".htm"}:
        return extract_text_from_html(path)
    # Fallback: treat as plain text
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

    This intentionally avoids a heavy tokenizer dependency and approximates
    "tokens" by simple whitespace-separated words.

    Behavior:
    - Split the input string on whitespace to obtain a list of "tokens".
    - Create chunks of length `max_tokens` words.
    - Between chunks, keep `overlap` words of overlap to improve recall.

    Args:
        text: Raw text to be chunked.
        max_tokens: Maximum number of words per chunk.
        overlap: Number of words to overlap between consecutive chunks.

    Returns:
        A list of chunk strings. Returns an empty list if the input is empty.
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
    1. Chunk the input text with `chunk_text`.
    2. Encode each chunk using the provided sentence-transformers model.
    3. Upsert embeddings + payload into the vector store.

    Each stored document payload contains:
    - `text`: the chunk text
    - `chunk_index`: position of the chunk within the document
    - all fields from `metadata` (if provided)

    Args:
        text: Raw text to ingest.
        metadata: Optional base metadata attached to every chunk payload.
        vector_store: VectorStore used to upsert embeddings into Qdrant.
        embedder: Sentence-transformers model used to encode the chunks.

    Returns:
        The number of chunks that were ingested (0 if there was no text).
    """
    chunks = chunk_text(text)
    if not chunks:
        return 0

    embeddings = embedder.encode(chunks, convert_to_numpy=True).tolist()

    docs: List[Dict[str, Any]] = []
    base_meta = metadata or {}
    for idx, chunk in enumerate(chunks):
        doc: Dict[str, Any] = {
            "text": chunk,
            "chunk_index": idx,
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

    The file is:
    - Parsed into text via `extract_text_from_file`.
    - Passed to `ingest_text` together with metadata.

    An additional metadata field `source_path` is always added to the payload
    to record the origin of the ingested content.

    Args:
        path: Filesystem path to the document file (PDF/HTML/TXT).
        metadata: Optional base metadata attached to every chunk.
        vector_store: VectorStore used for upserts.
        embedder: Sentence-transformers model.

    Returns:
        The number of chunks ingested for this file.
    """
    text = extract_text_from_file(path)
    file_meta = {"source_path": str(path)}
    if metadata:
        file_meta.update(metadata)
    return ingest_text(text, file_meta, vector_store, embedder)
