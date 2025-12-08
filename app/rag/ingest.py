from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from bs4 import BeautifulSoup
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

from app.rag.vectorstore import VectorStore


# -------- Text extraction --------


def extract_text_from_pdf(path: Path) -> str:
    reader = PdfReader(str(path))
    texts: List[str] = []
    for page in reader.pages:
        page_text = page.extract_text() or ""
        texts.append(page_text)
    return "\n".join(texts)


def extract_text_from_html(path: Path) -> str:
    html = path.read_text(encoding="utf-8")
    soup = BeautifulSoup(html, "html.parser")

    # Remove non-content tags
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()

    text = soup.get_text(separator=" ")
    # Normalize whitespace
    return " ".join(text.split())


def extract_text_from_file(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return extract_text_from_pdf(path)
    if suffix in {".html", ".htm"}:
        return extract_text_from_html(path)
    # Fallback: treat as plain text
    return path.read_text(encoding="utf-8")


# -------- Chunking --------


def chunk_text(
    text: str,
    max_tokens: int = 500,
    overlap: int = 50,
) -> List[str]:
    """
    Very simple word-based chunking to avoid extra tokenizer dependencies.

    - Splits on whitespace.
    - Ensures some overlap between chunks for better RAG recall.
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


# -------- Ingestion flow --------


def ingest_text(
    text: str,
    metadata: Optional[Dict[str, Any]],
    vector_store: VectorStore,
    embedder: SentenceTransformer,
) -> int:
    """
    End-to-end ingestion for already extracted text.

    - Chunk text
    - Embed chunks
    - Upsert into Qdrant via VectorStore
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
    Convenience for ingesting PDF/HTML/plain-text files from disk.
    """
    text = extract_text_from_file(path)
    file_meta = {"source_path": str(path)}
    if metadata:
        file_meta.update(metadata)
    return ingest_text(text, file_meta, vector_store, embedder)
