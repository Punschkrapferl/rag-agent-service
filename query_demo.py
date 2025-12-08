import json
import requests
from requests import RequestException

API_BASE = "http://localhost:8000/api"
QUERY_URL = f"{API_BASE}/query"


def main() -> None:
    queries = [
        "What is the capital of France?",
        "What is the capital of Germany?",
        "What is the capital of Japan?",
    ]

    print("API_BASE:", API_BASE)

    for q in queries:
        payload = {"query": q, "top_k": 3}
        try:
            resp = requests.post(QUERY_URL, json=payload)
        except RequestException as exc:
            print(f"\n=== Query: {q} ===")
            print("Request failed:", exc)
            continue

        print(f"\n=== Query: {q} ===")
        print("Status:", resp.status_code)

        try:
            data = resp.json()
        except json.JSONDecodeError:
            print("Raw body:", resp.text)
            continue

        print("Answer:", data.get("answer"))
        print("Documents:")
        for doc in data.get("documents", []):
            print(" -", doc.get("id"), "| score:", doc.get("score"))


if __name__ == "__main__":
    main()
