# RAG Agent Service
A production-style Retrieval-Augmented Generation (RAG) microservice built with FastAPI, Qdrant,
and Sentence-Transformers.
---
# Run Using Prebuilt Docker Images (Recommended for Reviewers)
You do **not** need to clone or build anything.
Start the full system directly using Docker Hub images:
```
docker compose up
```
Components:
- **rag-api** → FastAPI backend (http://localhost:8000)
- **qdrant** → Vector DB + dashboard (http://localhost:6333/dashboard)
---
# Local Development (Build from Source)
```
git clone https://github.com/your/repo.git
cd rag-agent-service
docker compose -f docker-compose.dev.yml up --build
```
---
# Note on Image Size
The `rag-agent-service-api` image is ~2 GB due to embedded ML dependencies.
In production, the model would be a mounted volume and the base image optimized (`python:slim`).
---
# API Usage Examples
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
curl -X POST "http://localhost:8000/api/query" -H "Content-Type: application/json" -d '{"query":
"What is the capital of France?", "top_k": 3}'
```
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