"""
Structured logging configuration for the RAG Agent Service.

This module configures:

- `structlog` as the primary logging library, emitting JSON-formatted logs.
- Python's built-in `logging` to forward log records through structlog.
- A timestamp, log level, stack info, and exception information for each log.

The resulting logs are:
- Machine-readable (JSON), suitable for log aggregation systems.
- Enriched with context via `structlog.contextvars` if used elsewhere.
"""

import logging

import structlog


def configure_logging() -> None:
    """
    Configure structlog and the standard logging module.

    This function should be called once during application startup (e.g. in
    `app.main.create_app`). It sets up:

    - A timestamper that adds ISO-formatted timestamps.
    - Processors that inject log level, stack info, and exception info.
    - A JSON renderer so logs are emitted as JSON strings.
    - A basic logging configuration that routes standard logging through
      structlog, with INFO as the default log level.
    """
    timestamper = structlog.processors.TimeStamper(fmt="iso")

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,  # merge per-request context
            timestamper,                              # add ISO timestamp
            structlog.processors.add_log_level,       # include log level
            structlog.processors.StackInfoRenderer(), # optional stack info
            structlog.processors.format_exc_info,     # render exceptions
            structlog.processors.JSONRenderer(),      # output as JSON
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,  # type: ignore[arg-type]
        cache_logger_on_first_use=True,
    )

    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",  # full record is provided by structlog JSONRenderer
    )
