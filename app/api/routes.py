"""JSON API routes.

Route handlers stay thin: they read the request, call a service, and return a
response model. Business logic lives in the service and pipeline modules.
"""

from typing import Annotated

from fastapi import APIRouter, Depends, File, Request, Response, UploadFile, status

from app.config import Settings, get_settings
from app.schemas import DocumentListResponse, DocumentResponse, ErrorResponse, HealthResponse
from app.ingestion.validation import read_upload_limited
from app.services.document_service import DocumentService

router = APIRouter()


def get_document_service(request: Request) -> DocumentService:
    return request.app.state.document_service


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


@router.post(
    "/api/v1/documents", response_model=DocumentResponse, status_code=status.HTTP_201_CREATED,
    tags=["documents"], summary="Upload and index a PDF or DOCX document",
    responses={400: {"model": ErrorResponse}, 409: {"model": ErrorResponse}, 413: {"model": ErrorResponse},
               415: {"model": ErrorResponse}, 503: {"model": ErrorResponse}},
)
async def upload_document(
    file: Annotated[UploadFile, File(description="PDF or DOCX document")],
    settings: Annotated[Settings, Depends(get_settings)],
    service: Annotated[DocumentService, Depends(get_document_service)],
) -> DocumentResponse:
    data = await read_upload_limited(file, settings.max_upload_bytes)
    document = await service.upload(file.filename, data)
    return DocumentResponse(**document)


@router.get("/api/v1/documents", response_model=DocumentListResponse, tags=["documents"],
            summary="List indexed documents")
async def list_documents(
    service: Annotated[DocumentService, Depends(get_document_service)],
) -> DocumentListResponse:
    documents = service.list_documents()
    return DocumentListResponse(documents=documents, total=len(documents))


@router.delete("/api/v1/documents/{document_id}", status_code=status.HTTP_204_NO_CONTENT,
               tags=["documents"], summary="Delete a document and its vectors",
               responses={404: {"model": ErrorResponse}})
async def delete_document(
    document_id: str,
    service: Annotated[DocumentService, Depends(get_document_service)],
) -> Response:
    await service.delete(document_id)
    return Response(status_code=status.HTTP_204_NO_CONTENT)
