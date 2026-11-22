"""JSON API routes.

Route handlers stay thin: they read the request, call a service, and return a
response model. Business logic lives in the service and pipeline modules.
"""

from typing import Annotated

from fastapi import APIRouter, Depends

from app.config import Settings, get_settings
from app.schemas import ErrorResponse, HealthResponse

router = APIRouter()


@router.get(
    "/health",
    response_model=HealthResponse,
    tags=["system"],
    summary="Check that the service is running",
    description="Returns service status and whether the LLM and embedding providers are configured. "
    "Never returns key values.",
    responses={500: {"model": ErrorResponse, "description": "Unexpected server error"}},
)
async def health(settings: Annotated[Settings, Depends(get_settings)]) -> HealthResponse:
    """Report service status."""
    return HealthResponse(
        status="ok",
        app_name=settings.app_name,
        version=settings.app_version,
        environment=settings.environment,
        llm_configured=settings.groq_configured,
        embeddings_configured=settings.gemini_configured,
    )
