"""JSON API routes: health and documents.

Route handlers stay thin: they read the request, call a service, and return a
response model. Business logic lives in the service and pipeline modules.
"""

from typing import Annotated

from fastapi import APIRouter, Depends, File, Path, Response, UploadFile, status

from app.api.deps import get_document_service, limit_uploads
from app.config import Settings, get_settings
from app.ingestion.validation import read_upload_limited
from app.schemas import DocumentListResponse, DocumentResponse, ErrorResponse, HealthResponse
from app.services.document_service import DocumentService

router = APIRouter()

DocumentIdPath = Annotated[str, Path(pattern=r"^[0-9a-f]{16}$", description="16 hex characters")]


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
    "/api/v1/documents",
    response_model=DocumentResponse,
    status_code=status.HTTP_201_CREATED,
    tags=["documents"],
    summary="Upload and index a PDF or DOCX document",
    description="Validates, extracts, chunks, embeds and stores the file. Identical bytes return 409.",
    responses={
        400: {"model": ErrorResponse, "description": "Empty, corrupt, password-protected or no extractable text"},
        409: {"model": ErrorResponse, "description": "The same file was already uploaded"},
        413: {"model": ErrorResponse, "description": "File or page count too large"},
        415: {"model": ErrorResponse, "description": "Not a PDF or DOCX file"},
        422: {"model": ErrorResponse, "description": "No file in the request"},
        429: {"model": ErrorResponse, "description": "Rate limit exceeded"},
        503: {"model": ErrorResponse, "description": "Embedding API or vector database unavailable"},
    },
    dependencies=[Depends(limit_uploads)],
)
async def upload_document(
    file: Annotated[UploadFile, File(description="PDF or DOCX document")],
    settings: Annotated[Settings, Depends(get_settings)],
    service: Annotated[DocumentService, Depends(get_document_service)],
) -> DocumentResponse:
    data = await read_upload_limited(file, settings.max_upload_bytes)
    document = await service.upload(file.filename, data)
    return DocumentResponse(**document)


@router.get(
    "/api/v1/documents",
    response_model=DocumentListResponse,
    tags=["documents"],
    summary="List indexed documents",
)
async def list_documents(
    service: Annotated[DocumentService, Depends(get_document_service)],
) -> DocumentListResponse:
    documents = service.list_documents()
    return DocumentListResponse(documents=documents, total=len(documents))


@router.get(
    "/api/v1/documents/{document_id}",
    response_model=DocumentResponse,
    tags=["documents"],
    summary="Get one document's metadata",
    responses={404: {"model": ErrorResponse}, 422: {"model": ErrorResponse}},
)
async def get_document(
    document_id: DocumentIdPath,
    service: Annotated[DocumentService, Depends(get_document_service)],
) -> DocumentResponse:
    return DocumentResponse(**service.get_document(document_id))


@router.delete(
    "/api/v1/documents/{document_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    tags=["documents"],
    summary="Delete a document and its vectors",
    description="Removes every vector of the document, so its text can never be retrieved or cited again.",
    responses={404: {"model": ErrorResponse}, 422: {"model": ErrorResponse}, 503: {"model": ErrorResponse}},
)
async def delete_document(
    document_id: DocumentIdPath,
    service: Annotated[DocumentService, Depends(get_document_service)],
) -> Response:
    await service.delete(document_id)
    return Response(status_code=status.HTTP_204_NO_CONTENT)
