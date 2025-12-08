# RAG Agent Service – Quick Start (for Recruiters)
This section gives a fast overview of how to run and test the project immediately using Docker.
## Quick Start
### 1. Clone the repository
```bash
git clone
cd rag-agent-service
```
### 2. Start everything with Docker Compose
This launches:
- FastAPI service (rag-api)
- Qdrant vector database
```bash
docker compose up --build
```
API is now running at:
```
http://localhost:8000
```
Qdrant Dashboard:
```
http://localhost:6333/dashboard
```
---
### 3. Test basic functionality
#### Health Check
```bash
curl http://localhost:8000/api/health
```
#### Ingest Text
```bash
curl -X POST "http://localhost:8000/api/ingest/text" -H "Content-Type: application/json" -d '{
"document_id": "doc-1",
"text": "The capital of France is Paris.",
"metadata": {"source": "demo"}
}'
```
#### Query
```bash
curl -X POST "http://localhost:8000/api/query" -H "Content-Type: application/json" -d '{"query": "What is
the capital of France?", "top_k": 3}'
```
You will see a retrieved document and a placeholder answer.
---
---

# RAG Agent Service
A production-style **Retrieval-Augmented Generation (RAG) microservice** built with **FastAPI**,
**Qdrant**, and **Sentence-Transformers**.
This project demonstrates backend engineering, vector search pipelines, observability, testing, Docker
orchestration, and CI/CD—aligned with modern AI system architecture expectations.
---
## ■ Features
### **API Endpoints**
| Endpoint | Description |
|---------|-------------|
| `GET /api/health` | Health check |
| `POST /api/ingest/text` | Ingest raw text into vector DB |
| `POST /api/ingest/file` | Ingest PDF/HTML/TXT files |
| `POST /api/query` | Vector search + placeholder answer |
| `GET /metrics` | Prometheus metrics endpoint |
---
## ■■ Core Components
### **FastAPI Backend**
- Modular routing structure (`/api/health`, `/api/ingest`, `/api/query`)
- Clean dependency injection for Qdrant + embedding model
- JSON responses with Pydantic models
- Structured, maintainable microservice layout
---
## ■ Document Ingestion Pipeline
### Supports:
- **PDF**
- **HTML / HTM**
- **Plain text**
### Pipeline Steps:
1. Extract text from file
2. Normalize / clean content
3. Word-based chunking (500 tokens, 50 overlap)
4. Embed chunks with:
```
sentence-transformers/all-MiniLM-L6-v2
```
5. Store embeddings + metadata in **Qdrant**
Metadata stored includes:
- `document_id`
- `chunk_index`
- filename / source
- any custom metadata fields
---
## ■ Query Pipeline
1. Encode the user query using MiniLM
2. Perform cosine similarity search in Qdrant
3. Collect top-K relevant chunks
4. Return:
- Retrieved documents
- Scores
- Metadata
- **A placeholder answer** (no LLM required)
---
## ■ Observability
### **Logging**
- JSON logging using `structlog`
- Request tracing identifiers included
### **Metrics**
- Prometheus endpoint at `/metrics`
### **Tracing**
- OpenTelemetry auto-instrumentation
---
## ■ Testing
### Unit Tests (`pytest`)
- Health endpoint
- Query endpoint (mocked VectorStore)
### Continuous Integration
- GitHub Actions running tests on each push/PR
---
## ■ Running with Docker
```bash
docker compose up --build
```
API Docs: `http://localhost:8000/docs`
---
## ■ Example: Ingest Text
```bash
curl -X POST "http://localhost:8000/api/ingest/text" \
-H "Content-Type: application/json" \
-d '{
"document_id": "doc-1",
"text": "The capital of France is Paris.",
"metadata": {"source": "demo"}
}'
```
---
## ■ Example: Query
```bash
curl -X POST "http://localhost:8000/api/query" \
-H "Content-Type: application/json" \
-d '{"query": "What is the capital of France?", "top_k": 3}'
```
---
## ■ Project Structure
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
## ■ Summary
A realistic **RAG backend microservice** with:
- End-to-end ingestion → embedding → vector search
- FastAPI architecture
- Qdrant integration
- Testing + CI
- Docker orchestration
- Metrics + tracing`