"""
Main application factory for the RAG Agent Service.

This module is responsible for:

- Creating and configuring the FastAPI application instance.
- Wiring up:
  - Routers (health, ingest, query, debug) under a common API prefix.
  - Structured logging.
  - Prometheus metrics (`/metrics` endpoint).
  - OpenTelemetry tracing.
- Defining the application lifespan (startup/shutdown hooks).

The global `app` object at the bottom is the ASGI entrypoint used by uvicorn
and in tests.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
import asyncio

import structlog
from fastapi import FastAPI
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

from app.api import routes_health, routes_ingest, routes_query, routes_debug
from app.config import get_settings
from app.deps import get_embedding_model, get_qdrant_client  # warm-up + Qdrant
from app.logging_config import configure_logging
from app.observability.metrics import setup_metrics
from app.observability.tracing import setup_tracing

logger = structlog.get_logger()
settings = get_settings()


async def _wait_for_qdrant(client: QdrantClient, retries: int = 10, delay: float = 1.0) -> None:
    """
    Wait for Qdrant to become reachable.

    This avoids failing cleanup when Qdrant is still starting up.
    """
    for attempt in range(1, retries + 1):
        try:
            client.get_collections()
            logger.info("qdrant_ready", attempt=attempt)
            return
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "qdrant_not_ready",
                attempt=attempt,
                reason=str(exc),
            )
            await asyncio.sleep(delay)
    logger.warning("qdrant_still_unreachable_after_retries", retries=retries)


def _reset_collection(client: QdrantClient) -> None:
    """
    Recreate the Qdrant collection using the non-deprecated
    collection_exists / create_collection API.

    Steps:
    - Determine embedding dimension from the configured model.
    - If the collection exists, drop it via delete_collection.
    - Create a fresh collection with the correct vector size and distance.

    Result: every startup in matching environments begins with an empty
    collection, without requiring `docker compose ... down -v`.
    """
    embedder = get_embedding_model()
    try:
        dim = embedder.get_sentence_embedding_dimension()
    except AttributeError:
        sample_vec = embedder.encode("demo")
        dim = len(sample_vec)

    collection_name = settings.qdrant_collection

    if client.collection_exists(collection_name=collection_name):
        logger.info(
            "reset_collection_drop_existing",
            collection=collection_name,
        )
        client.delete_collection(collection_name=collection_name)

    logger.info(
        "reset_collection_create",
        collection=collection_name,
        dimension=dim,
    )

    client.create_collection(
        collection_name=collection_name,
        vectors_config=qmodels.VectorParams(
            size=dim,
            distance=qmodels.Distance.COSINE,
        ),
    )

    count = client.count(collection_name=collection_name, exact=True).count
    logger.info(
        "reset_collection_complete",
        collection=collection_name,
        dimension=dim,
        count_after_reset=count,
    )


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    logger.info("startup", environment=settings.environment)

    # Warm up embedding model
    logger.info("model_warmup_start", model_name=settings.embedding_model_name)
    _ = get_embedding_model()
    logger.info("model_warmup_complete", model_name=settings.embedding_model_name)

    # Hard-reset Qdrant collection at startup in demo/dev
    if settings.environment in {"demo", "dev"}:
        logger.info("startup_reset_enabled")
        client = get_qdrant_client()

        await _wait_for_qdrant(client)

        try:
            _reset_collection(client)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "startup_reset_failed",
                reason=str(exc),
            )

    try:
        yield
    finally:
        logger.info("shutdown")


def create_app() -> FastAPI:
    """
    Application factory.

    This function:
    - Configures structured logging.
    - Creates the FastAPI app with a lifespan context manager.
    - Registers all API routers under the configured `api_prefix`.
    - Enables observability via Prometheus metrics and OpenTelemetry tracing.

    Returns:
        FastAPI: Fully configured FastAPI application instance.
    """
    configure_logging()

    application = FastAPI(
        title=settings.app_name,
        lifespan=lifespan,
    )

    application.include_router(routes_health.router, prefix=settings.api_prefix)
    application.include_router(routes_query.router, prefix=settings.api_prefix)
    application.include_router(routes_ingest.router, prefix=settings.api_prefix)
    application.include_router(routes_debug.router, prefix=settings.api_prefix)

    setup_metrics(application)
    setup_tracing(application, service_name=settings.service_name)

    return application


app = create_app()
