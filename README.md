# RAG Agent Service
RAG Agent Service
A production-style Retrieval-Augmented Generation (RAG) microservice built with FastAPI, Qdrant, and Sentence-Transformers.
The service supports text ingestion, vector embedding, semantic search, and query answering.
---
# Run Using Prebuilt Docker Images (Recommended for Reviewers)
You do **not** need to clone or build anything.
Start the full system directly using Docker Hub images:
---
# 1. Install Docker Desktop
https://www.docker.com/products/docker-desktop/
# 2. Start the entire system
```
docker compose up
```
Docker will automatically pull the prebuilt images from Docker Hub:
punschkrapferl23/rag-agent-service-api:latest
qdrant/qdrant:v1.11.0

Components:
- **rag-api** → FastAPI backend (http://localhost:8000)
- **qdrant** → Vector DB + dashboard (http://localhost:6333/dashboard)
---
# API Usage Examples - Test after docker compose up
## Health Check
```
curl http://localhost:8000/api/health
```
## Ingest Text
```
curl -X POST "http://localhost:8000/api/ingest/text" -H "Content-Type: application/json" -d '{
"document_id": "doc-1",
"text": "The capital of France is Paris.",
"metadata": {"source": "demo"}
}'
```
## Query
```
curl -X POST "http://localhost:8000/api/query" -H "Content-Type: application/json" -d '{
"query": "What is the capital of France?",
"top_k": 3
}'
```

# Local Development (Build from Source)
```
git clone https://github.com/your/repo.git
cd rag-agent-service
docker compose -f docker-compose.dev.yml up --build
```
---
# Note on Image Size
The main API image is ~2 GB because it includes:
- Python + FastAPI
- Qdrant client libraries
- Sentence-transformers model weights
- PDF/HTML extraction dependencies

In a real production environment:
- The model would be mounted as a volume
- Base image would be changed to python:slim
- Layers would be aggressively minimized
---

---
# Demo Scripts
## Run full workflow
```
python demo/demo_all.py
```
## Ingest single document
```
python demo/ingest_demo.py
```
## Batch ingest
```
python demo/ingest_batch_demo.py
```
## Query tests
```
python demo/query_demo.py
```
---
# Architecture Overview
```
Client → FastAPI (rag-api) → Embedding Model → Qdrant Vector DB
→ Ranked Results → JSON Response
```
Main Components:
- FastAPI backend
- Sentence-Transformers embedding model
- Qdrant vector store
- Ingestion + chunking pipeline
- Semantic search
---
# Project Structure
```
app/
api/
routes_health.py
routes_ingest.py
routes_query.py
rag/
ingest.py
pipeline.py
vectorstore.py
models.py
observability/
metrics.py
tracing.py
deps.py
config.py
main.py
docker/
Dockerfile.api
tests/
test_health.py
test_query.py
docker-compose.yml
docker-compose.dev.yml
requirements.txt
README.md
```
---
# MIT License

Copyright (c) 2025 Punschkrapferl

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```