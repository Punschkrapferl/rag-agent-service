from __future__ import annotations

import os
import sys
from typing import Optional

from fastapi import FastAPI

from opentelemetry import trace
from opentelemetry.sdk.resources import SERVICE_NAME, Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import (
    BatchSpanProcessor,
    ConsoleSpanExporter,
)
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.logging import LoggingInstrumentor


_tracing_initialized = False


def _build_tracer_provider(service_name: str) -> TracerProvider:
    resource = Resource.create({SERVICE_NAME: service_name})
    provider = TracerProvider(resource=resource)
    return provider


def _build_exporter() -> OTLPSpanExporter | ConsoleSpanExporter:
    """
    If OTEL_EXPORTER_OTLP_ENDPOINT is set, export via OTLP HTTP.
    Otherwise, export spans to console (useful for local dev).
    """
    otlp_endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT")
    if otlp_endpoint:
        return OTLPSpanExporter(endpoint=otlp_endpoint)
    return ConsoleSpanExporter()


def setup_tracing(app: FastAPI, service_name: Optional[str] = None) -> None:
    """
    Initialize OpenTelemetry tracing for FastAPI.

    Skips initialization when running under pytest (test process),
    to avoid ConsoleSpanExporter writing to a closed capture stream.
    """
    global _tracing_initialized
    if _tracing_initialized:
        return

    # Do not enable tracing inside pytest
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
