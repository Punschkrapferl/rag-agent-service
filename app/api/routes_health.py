"""
Health check endpoint for the RAG Agent Service.

This router exposes a single liveness probe at `/api/health` that can be used by:
- Kubernetes / Docker health checks
- Load balancers
- External monitoring systems

The endpoint returns a simple JSON object with a `status` field.
"""

from typing_extensions import TypedDict  # <-- important: typing_extensions, not typing

from fastapi import APIRouter


class HealthResponse(TypedDict):
    """Schema for the health check response."""
    status: str


# Router that groups all health-related endpoints in the OpenAPI docs
router = APIRouter(
    tags=["health"],
    responses={
        200: {"description": "Service is running and accepting requests."},
    },
)


@router.get(
    "/health",
    summary="Service health check",
    response_model=HealthResponse,
)
async def health() -> HealthResponse:
    """
    Liveness endpoint for the service.

    Returns:
        HealthResponse: A JSON object indicating that the service is up.

    Example response:
    ```json
    {
      "status": "ok"
    }
    ```
    """
    return {"status": "ok"}
