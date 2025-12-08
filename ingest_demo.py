import json
import requests
from requests import RequestException

API_BASE = "http://localhost:8000/api"
INGEST_URL = f"{API_BASE}/ingest/text"


def main() -> None:
    docs = [
        {
            "document_id": "doc-1",
            "text": "The capital of France is Paris.",
            "metadata": {"source": "demo", "topic": "geography"},
        },
        {
            "document_id": "doc-2",
            "text": "Berlin is the capital of Germany.",
            "metadata": {"source": "demo", "topic": "geography"},
        },
        {
            "document_id": "doc-3",
            "text": "Tokyo is the capital of Japan.",
            "metadata": {"source": "demo", "topic": "geography"},
        },
    ]

    print("API_BASE:", API_BASE)

    for doc in docs:
        try:
            resp = requests.post(INGEST_URL, json=doc)
        except RequestException as exc:
            print(f"{doc['document_id']} -> request failed:", exc)
            continue

        print(f"{doc['document_id']} ->", resp.status_code)
        try:
            print("  Body:", resp.json())
        except json.JSONDecodeError:
            print("  Raw body:", resp.text)


if __name__ == "__main__":
    main()
