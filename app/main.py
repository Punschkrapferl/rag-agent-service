from __future__ import annotations

from contextlib import asynccontextmanager
from collections.abc import AsyncIterator

import structlog
from fastapi import FastAPI

from app.config import get_settings
from app.logging_config import configure_logging
from app.api import routes_health, routes_query, routes_ingest
from app.observability.metrics import setup_metrics
from app.observability.tracing import setup_tracing

logger = structlog.get_logger()
settings = get_settings()


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    # Startup logic
    logger.info("startup", environment=settings.environment)
    try:
        yield
    finally:
        # Shutdown logic
        logger.info("shutdown")


def create_app() -> FastAPI:
    configure_logging()

    application = FastAPI(
        title=settings.app_name,
        lifespan=lifespan,
    )

    # Routers under /api
    application.include_router(routes_health.router, prefix=settings.api_prefix)
    application.include_router(routes_query.router, prefix=settings.api_prefix)
    application.include_router(routes_ingest.router, prefix=settings.api_prefix)

    # Observability
    setup_metrics(application)
    setup_tracing(application, service_name=settings.service_name)

    return application


app = create_app()
