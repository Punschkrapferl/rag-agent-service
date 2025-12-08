"""
OpenTelemetry tracing setup for the RAG Agent Service.

This module configures distributed tracing for the FastAPI application using
OpenTelemetry. It supports two export modes:

- OTLP over HTTP (when `OTEL_EXPORTER_OTLP_ENDPOINT` is set), suitable for
  sending traces to backends like Jaeger, Tempo, Grafana Cloud, etc.
- Console export (default), which prints spans to stdout for local development.

Tracing is:
- Initialized only once per process (guarded by `_tracing_initialized`).
- Disabled automatically when running under pytest to avoid interfering with
  test output and capture streams.
"""

from __future__ import annotations

import os
import sys
from typing import Optional

from fastapi import FastAPI
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.logging import LoggingInstrumentor
from opentelemetry.sdk.resources import SERVICE_NAME, Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import (
    BatchSpanProcessor,
    ConsoleSpanExporter,
)

_tracing_initialized = False


def _build_tracer_provider(service_name: str) -> TracerProvider:
    """
    Create a TracerProvider with a service name resource.

    Args:
        service_name: Logical name of the service, shown in your tracing backend.

    Returns:
        TracerProvider: Configured OpenTelemetry tracer provider.
    """
    resource = Resource.create({SERVICE_NAME: service_name})
    provider = TracerProvider(resource=resource)
    return provider


def _build_exporter() -> OTLPSpanExporter | ConsoleSpanExporter:
    """
    Build a span exporter based on environment configuration.

    Behavior:
    - If `OTEL_EXPORTER_OTLP_ENDPOINT` is defined, create an OTLP HTTP exporter
      that sends spans to that endpoint.
    - Otherwise, fall back to a ConsoleSpanExporter that prints spans to stdout.

    Returns:
        OTLPSpanExporter or ConsoleSpanExporter: Exporter used by the span processor.
    """
    otlp_endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT")
    if otlp_endpoint:
        return OTLPSpanExporter(endpoint=otlp_endpoint)
    return ConsoleSpanExporter()


def setup_tracing(app: FastAPI, service_name: Optional[str] = None) -> None:
    """
    Initialize OpenTelemetry tracing for the FastAPI application.

    This function:
    - Skips initialization when running under pytest (test process),
      to avoid ConsoleSpanExporter writing to a closed capture stream.
    - Creates a TracerProvider with the configured service name.
    - Attaches a BatchSpanProcessor with either an OTLP or console exporter.
    - Instruments FastAPI routes so each request is traced.
    - Instruments logging so logs can be correlated with trace context.

    Args:
        app: FastAPI application instance to instrument.
        service_name: Optional explicit service name. If not provided,
            falls back to `OTEL_SERVICE_NAME` env var or
            `"rag-agent-service"` as a default.

    Note:
        Tracing is initialized only once per process. Subsequent calls are no-ops.
    """
    global _tracing_initialized
    if _tracing_initialized:
        return

    # Do not enable tracing inside pytest to avoid noisy test output.
    if "pytest" in sys.modules:
        return

    if service_name is None:
        service_name = os.getenv("OTEL_SERVICE_NAME", "rag-agent-service")

    provider = _build_tracer_provider(service_name)
    exporter = _build_exporter()
    span_processor = BatchSpanProcessor(exporter)
    provider.add_span_processor(span_processor)

    trace.set_tracer_provider(provider)

    # Instrument FastAPI
    FastAPIInstrumentor.instrument_app(app)

    # Optional: correlate logs with traces
    LoggingInstrumentor().instrument(set_logging_format=True)

    _tracing_initialized = True
