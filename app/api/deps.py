"""FastAPI dependencies shared by the JSON API and the HTML pages.

Services are created once in ``create_app`` and stored on ``app.state``;
these helpers hand them to route handlers. Rate limits are dependencies, so a
route opts in by listing them.
"""

from fastapi import Request

from app.errors import RateLimitExceededError
from app.guardrails import RateLimiter
from app.services.document_service import DocumentService
from app.services.qa_service import QAService


def get_document_service(request: Request) -> DocumentService:
    return request.app.state.document_service


def get_qa_service(request: Request) -> QAService:
    return request.app.state.qa_service


def _client_key(request: Request) -> str:
    return request.client.host if request.client else "unknown"


def _enforce(limiter: RateLimiter, request: Request) -> None:
    retry_after = limiter.acquire(_client_key(request))
    if retry_after is not None:
        raise RateLimitExceededError(
            details={"retry_after_seconds": retry_after}, headers={"Retry-After": str(retry_after)}
        )


def limit_questions(request: Request) -> None:
    """Per-IP limit for /ask, /search and the question form (QUESTIONS_PER_MINUTE)."""
    _enforce(request.app.state.question_limiter, request)


def limit_uploads(request: Request) -> None:
    """Per-IP limit for uploads (UPLOADS_PER_MINUTE)."""
    _enforce(request.app.state.upload_limiter, request)
