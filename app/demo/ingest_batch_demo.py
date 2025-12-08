import json
import requests
from requests import RequestException

API_BASE = "http://localhost:8000/api"
BATCH_URL = f"{API_BASE}/ingest/text/batch"


def main() -> None:
    docs = [
        {
            "document_id": "doc-10",
            "text": "Madrid is the capital of Spain.",
            "metadata": {"source": "batch-demo", "topic": "geography"},
        },
        {
            "document_id": "doc-11",
            "text": "Rome is the capital of Italy.",
            "metadata": {"source": "batch-demo", "topic": "geography"},
        },
    ]

    payload = {"items": docs}
    print("API_BASE:", API_BASE)

    try:
        resp = requests.post(BATCH_URL, json=payload)
    except RequestException as exc:
        print("Request failed:", exc)
        return

    print("Status:", resp.status_code)
    try:
        print("Body:", resp.json())
    except json.JSONDecodeError:
        print("Raw body:", resp.text)


if __name__ == "__main__":
    main()
