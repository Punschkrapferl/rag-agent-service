"""
Prometheus-metrics integration for the RAG Agent Service.

This module wires Prometheus-compatible metrics into the FastAPI app using
`prometheus-fastapi-instrumentator`. It exposes a `/metrics` endpoint that
can be scraped by Prometheus or similar monitoring systems.
"""

from prometheus_fastapi_instrumentator import Instrumentator
from fastapi import FastAPI


def setup_metrics(app: FastAPI) -> None:
    """
    Attach Prometheus instrumentation and expose the `/metrics` endpoint.

    This function:
    - Instruments all FastAPI routes to record default HTTP metrics
      (latency, status codes, etc.).
    - Registers a `/metrics` endpoint that serves metrics in Prometheus format.

    Args:
        app: The FastAPI application instance to instrument.
    """
    Instrumentator().instrument(app).expose(app, endpoint="/metrics")
