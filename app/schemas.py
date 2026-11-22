"""Pydantic request and response models.

These models define the API contract and feed FastAPI's OpenAPI docs at
``/docs``. Each model carries an example so Swagger shows realistic payloads.
"""

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class ErrorBody(BaseModel):
    """The inner object of every error response."""

    code: str = Field(description="Stable, machine-readable error code.")
    message: str = Field(description="Human-readable explanation.")
    request_id: str = Field(description="ID to quote when reporting a problem; matches the X-Request-ID header.")
    details: dict[str, Any] = Field(default_factory=dict, description="Extra context, such as an existing document ID.")


class ErrorResponse(BaseModel):
    """Shape shared by all error responses (4xx and 5xx)."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "error": {
                    "code": "duplicate_document",
                    "message": "This file was already uploaded.",
                    "request_id": "3f2b9c1e8a7d4e6f9b0c1d2e3f4a5b6c",
                    "details": {"document_id": "d7c1a2b3"},
                }
            }
        }
    )

    error: ErrorBody


class HealthResponse(BaseModel):
    """Service status. Reports whether keys are configured, never their values."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "status": "ok",
                "app_name": "Production RAG Service",
                "version": "0.1.0",
                "environment": "development",
                "llm_configured": True,
                "embeddings_configured": True,
            }
        }
    )

    status: str = Field(description="'ok' when the API process is running.")
    app_name: str
    version: str
    environment: str
    llm_configured: bool = Field(description="Groq key and answer model are set.")
    embeddings_configured: bool = Field(description="Gemini key and embedding model are set.")
