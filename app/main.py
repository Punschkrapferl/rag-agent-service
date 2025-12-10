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

import structlog
from fastapi import FastAPI

from app.api import routes_health, routes_ingest, routes_query, routes_debug
from app.config import get_settings
from app.deps import get_embedding_model  # warm-up: load SentenceTransformer once
from app.logging_config import configure_logging
from app.observability.metrics import setup_metrics
from app.observability.tracing import setup_tracing

logger = structlog.get_logger()
settings = get_settings()


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    """
    Application lifespan context manager.

    Startup:
        - Logs a "startup" event including the current environment.
        - Warms up the embedding model so the first request does not block
          on model download/initialization.

    Shutdown:
        - Logs a "shutdown" event.
    """
    # Startup logic
    logger.info("startup", environment=settings.environment)

    # Warm up the embedding model once at startup so that subsequent requests
    # can reuse the cached instance without incurring a long delay.
    logger.info("model_warmup_start", model_name=settings.embedding_model_name)
    _ = get_embedding_model()
    logger.info("model_warmup_complete", model_name=settings.embedding_model_name)

    try:
        yield
    finally:
        # Shutdown logic
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

    # API routers mounted under the common prefix (e.g. "/api")
    application.include_router(routes_health.router, prefix=settings.api_prefix)
    application.include_router(routes_query.router, prefix=settings.api_prefix)
    application.include_router(routes_ingest.router, prefix=settings.api_prefix)
    application.include_router(routes_debug.router, prefix=settings.api_prefix)

    # Observability hooks
    setup_metrics(application)
    setup_tracing(application, service_name=settings.service_name)

    return application


# ASGI application entrypoint (used by uvicorn and tests)
app = create_app()
