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

from app.api.routes import router as api_router
from app.config import Settings, get_settings
from app.errors import AppError
from app.logging_setup import configure_logging
from app.middleware import RequestContextMiddleware

logger = logging.getLogger("app")


def _error_response(request: Request, status_code: int, code: str, message: str, details: dict) -> JSONResponse:
    """Build the single error shape used by every error response."""
    request.state.error_code = code  # picked up by the access log
    return JSONResponse(
        status_code=status_code,
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
        return _error_response(request, exc.status_code, exc.code, exc.message, exc.details)

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


def create_app(settings: Settings | None = None) -> FastAPI:
    """Build and configure the FastAPI application."""
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

    app.add_middleware(RequestContextMiddleware)
    register_exception_handlers(app)
    app.include_router(api_router)
    return app


app = create_app()
