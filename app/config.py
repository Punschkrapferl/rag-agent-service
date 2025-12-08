"""
Application configuration for the RAG Agent Service.

This module centralizes configuration using Pydantic's `BaseSettings`, which
allows values to be provided via:

- Environment variables
- A `.env` file in the project root
- Default values defined in the `Settings` class

Typical usage:

    from app.config import get_settings

    settings = get_settings()
    print(settings.qdrant_host)

Using `get_settings()` with `lru_cache` ensures that the settings object is
created only once per process, which is both efficient and convenient to use
throughout the codebase.
"""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Strongly-typed application configuration.

    Fields:
        app_name: Human-readable application name used in FastAPI docs, etc.
        service_name: Logical service identifier used for tracing.

        environment: Deployment environment indicator (e.g. "local", "docker",
            "staging", "production").
        debug: Whether to enable debug-related behavior.
        api_prefix: Base path under which all API routes are mounted.

        qdrant_host: Hostname of the Qdrant instance.
        qdrant_port: Port of the Qdrant instance.
        qdrant_collection: Default collection name for storing embeddings.

        embedding_model_name: Name of the sentence-transformers model used
            for both ingestion and query embeddings.
    """

    # Pydantic v2 configuration: load from `.env` with UTF-8 encoding
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
    )

    # Application
    app_name: str = "RAG Agent Service"
    service_name: str = "rag-agent-service"

    environment: str = "local"
    debug: bool = True
    api_prefix: str = "/api"

    # Qdrant
    qdrant_host: str = "qdrant"
    qdrant_port: int = 6333
    qdrant_collection: str = "documents"

    # Model
    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """
    Return a cached Settings instance.

    The first call reads configuration from environment variables and `.env`.
    Subsequent calls return the same instance, avoiding repeated parsing.

    Returns:
        Settings: Singleton settings object for the current process.
    """
    return Settings()
