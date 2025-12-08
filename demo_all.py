"""
End-to-end demo script for the RAG Agent Service.

This script exercises all main endpoints:

1. /api/health
2. /api/ingest/text
3. /api/ingest/text/batch
4. /api/ingest/file
5. /api/query

Run with:
    python demo_all.py
"""

import os
from pathlib import Path

import requests

API_BASE = "http://localhost:8000/api"


def demo_health() -> None:
    print("=== 1) Health check ===")
    resp = requests.get(f"{API_BASE}/health")
    print("Status:", resp.status_code)
    print("Body:", resp.json())


def demo_ingest_text_single() -> None:
    print("\n=== 2) Ingest single text documents ===")
    docs = [
        {
            "document_id": "demo-doc-1",
            "text": "The capital of France is Paris.",
            "metadata": {"source": "demo-single", "topic": "geography"},
        },
        {
            "document_id": "demo-doc-2",
            "text": "Berlin is the capital of Germany.",
            "metadata": {"source": "demo-single", "topic": "geography"},
        },
    ]

    for doc in docs:
        resp = requests.post(f"{API_BASE}/ingest/text", json=doc)
        print(doc["document_id"], "->", resp.status_code)
        print("Raw body:", resp.text)  # instead of resp.json()


def demo_ingest_text_batch() -> None:
    print("\n=== 3) Ingest multiple texts (batch) ===")
    batch_payload = {
        "items": [
            {
                "document_id": "demo-doc-3",
                "text": "Tokyo is the capital of Japan.",
                "metadata": {"source": "demo-batch", "topic": "geography"},
            },
            {
                "document_id": "demo-doc-4",
                "text": "Madrid is the capital of Spain.",
                "metadata": {"source": "demo-batch", "topic": "geography"},
            },
        ]
    }

    resp = requests.post(f"{API_BASE}/ingest/text/batch", json=batch_payload)
    print("Status:", resp.status_code)
    print("Body:", resp.json())


def demo_ingest_file() -> None:
    print("\n=== 4) Ingest file ===")

    # Adjust this path to any existing file you want to test with.
    # A simple option: create a `sample.txt` in the project root.
    sample_path = Path("sample.txt")

    if not sample_path.exists():
        print("Skipping file ingest: sample.txt does not exist in current directory.")
        return

    with sample_path.open("rb") as f:
        files = {"file": (sample_path.name, f, "text/plain")}
        data = {"document_id": "demo-file-1"}

        resp = requests.post(f"{API_BASE}/ingest/file", files=files, data=data)

    print("Status:", resp.status_code)
    print("Body:", resp.json())


def demo_query() -> None:
    print("\n=== 5) Query RAG endpoint ===")

    queries = [
        "What is the capital of France?",
        "What is the capital of Germany?",
        "What is the capital of Spain?",
        "What is the capital of Japan?",
    ]

    for q in queries:
        payload = {"query": q, "top_k": 3}
        resp = requests.post(f"{API_BASE}/query", json=payload)
        print(f"\n--- Query: {q} ---")
        print("Status:", resp.status_code)
        if resp.status_code != 200:
            print("Error body:", resp.text)
            continue

        data = resp.json()
        print("Answer:", data.get("answer"))
        print("Documents:")
        for doc in data.get("documents", []):
            doc_id = doc.get("id")
            score = doc.get("score")
            text_preview = (doc.get("text") or "").strip()
            if len(text_preview) > 60:
                text_preview = text_preview[:57] + "..."
            print(f" - id={doc_id} | score={score:.4f} | text='{text_preview}'")


def main() -> None:
    # Optional: allow overriding API base via env var
    global API_BASE
    API_BASE = os.getenv("RAG_API_BASE", API_BASE)

    demo_health()
    demo_ingest_text_single()
    demo_ingest_text_batch()
    demo_ingest_file()
    demo_query()


if __name__ == "__main__":
    main()
