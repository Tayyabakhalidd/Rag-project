"""Application entry point: builds the FastAPI app. No business logic here.

Run locally with::

    uvicorn app.main:app --reload

``create_app()`` wires together settings, logging, middleware, error handlers
and routers. Tests call it with their own ``Settings`` so they never read the
real ``.env`` file.
"""

import logging

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.api.qa import router as qa_router
from app.api.routes import router as api_router
from app.api.ui import router as ui_router
from app.config import Settings, get_settings
from app.errors import AppError
from app.generation import GroqLLM, LLMClient
from app.guardrails import RateLimiter
from app.ingestion.embed import Embedder, build_embedder
from app.logging_setup import configure_logging
from app.middleware import RequestContextMiddleware
from app.retrieval import Retriever
from app.services.document_service import DocumentService
from app.services.qa_service import QAService
from app.storage.registry import DocumentRegistry
from app.storage.vector_store import ChromaVectorStore

logger = logging.getLogger("app")


def _error_response(
    request: Request, status_code: int, code: str, message: str, details: dict, headers: dict | None = None
) -> JSONResponse:
    """Build the single error shape used by every error response."""
    request.state.error_code = code  # picked up by the access log
    return JSONResponse(
        status_code=status_code,
        headers=headers or None,
        content={
            "error": {
                "code": code,
                "message": message,
                "request_id": getattr(request.state, "request_id", "-"),
                "details": details,
            }
        },
    )


def register_exception_handlers(app: FastAPI) -> None:
    """Map every kind of error to the same JSON error shape."""

    @app.exception_handler(AppError)
    async def handle_app_error(request: Request, exc: AppError) -> JSONResponse:
        if exc.status_code >= 500:
            logger.warning("Service error: %s", exc.code)
        return _error_response(request, exc.status_code, exc.code, exc.message, exc.details, exc.headers)

    @app.exception_handler(RequestValidationError)
    async def handle_validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
        # Report where and why validation failed, but not the submitted value,
        # so questions or file contents are never echoed back or logged.
        problems = [
            {"location": ".".join(str(part) for part in err.get("loc", ())), "problem": err.get("msg", "")}
            for err in exc.errors()
        ]
        return _error_response(
            request, 422, "validation_error", "The request data is invalid.", {"problems": problems}
        )

    @app.exception_handler(StarletteHTTPException)
    async def handle_http_error(request: Request, exc: StarletteHTTPException) -> JSONResponse:
        codes = {404: "not_found", 405: "method_not_allowed"}
        code = codes.get(exc.status_code, "http_error")
        message = exc.detail if isinstance(exc.detail, str) else "HTTP error."
        return _error_response(request, exc.status_code, code, message, {})


def create_app(
    settings: Settings | None = None,
    *,
    embedder: Embedder | None = None,
    llm: LLMClient | None = None,
) -> FastAPI:
    """Build and configure the FastAPI application.

    ``embedder`` and ``llm`` default to the real Gemini and Groq clients.
    Tests pass deterministic fakes, so the test suite never calls a paid or
    live API.
    """
    settings = settings or get_settings()
    configure_logging(settings.log_level, settings.secret_values())

    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        description=(
            "Upload PDF and DOCX documents, ask questions in plain language, and get answers "
            "with citations to the retrieved document chunks. Errors share one JSON shape: "
            '`{"error": {"code", "message", "request_id", "details"}}`.'
        ),
    )
    # Make route dependencies use the same settings object the app was built with.
    app.dependency_overrides[get_settings] = lambda: settings
    app.state.settings = settings

    # Shared components: one registry, one vector collection, one cached embedder.
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    registry = DocumentRegistry(settings.data_dir / "documents.sqlite3")
    vectors = ChromaVectorStore(settings.data_dir / "chroma")
    cached_embedder = build_embedder(settings, registry, embedder)
    retriever = Retriever(vectors, cached_embedder)
    app.state.document_service = DocumentService(settings, registry, vectors, cached_embedder)
    app.state.qa_service = QAService(settings, registry, retriever, llm or GroqLLM(settings))
    app.state.question_limiter = RateLimiter(settings.questions_per_minute)
    app.state.upload_limiter = RateLimiter(settings.uploads_per_minute)

    app.add_middleware(RequestContextMiddleware)
    register_exception_handlers(app)
    app.include_router(api_router)
    app.include_router(qa_router)
    app.include_router(ui_router)
    return app


app = create_app()
