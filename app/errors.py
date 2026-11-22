"""Application errors.

Services raise these exceptions; ``main.py`` turns every one of them into the
same JSON shape::

    {"error": {"code": "...", "message": "...", "request_id": "...", "details": {...}}}

Keeping status codes here (not in route handlers) means business logic never
has to know about HTTP, and every error looks the same to API clients.
"""

from typing import Any


class AppError(Exception):
    """Base class for errors that should reach the client with a clear message."""

    status_code: int = 500
    code: str = "internal_error"
    message: str = "An unexpected error occurred."

    def __init__(self, message: str | None = None, *, details: dict[str, Any] | None = None) -> None:
        self.message = message or self.message
        self.details = details or {}
        super().__init__(self.message)


class EmptyDocumentError(AppError):
    status_code = 400
    code = "empty_document"
    message = "The file is empty or contains no extractable text."


class GuardrailBlockedError(AppError):
    status_code = 400
    code = "guardrail_blocked"
    message = "The request was blocked by a safety check."


class DocumentNotFoundError(AppError):
    status_code = 404
    code = "document_not_found"
    message = "No document exists with this ID."


class DuplicateDocumentError(AppError):
    status_code = 409
    code = "duplicate_document"
    message = "This file was already uploaded."


class FileTooLargeError(AppError):
    status_code = 413
    code = "file_too_large"
    message = "The file is larger than the allowed upload size."


class UnsupportedFileTypeError(AppError):
    status_code = 415
    code = "unsupported_file_type"
    message = "This file type is not supported. Upload a PDF or DOCX file."


class RateLimitExceededError(AppError):
    status_code = 429
    code = "rate_limit_exceeded"
    message = "Too many requests. Please wait and try again."


class UpstreamUnavailableError(AppError):
    """An external service (LLM, embeddings, vector store) failed or timed out."""

    status_code = 503
    code = "upstream_unavailable"
    message = "A required external service is unavailable. Please try again shortly."


class UpstreamRateLimitedError(UpstreamUnavailableError):
    """A provider's own quota (e.g. a free tier) ran out. Not our 429."""

    code = "upstream_rate_limited"
    message = "An external provider's rate limit was reached. Please try again shortly."
