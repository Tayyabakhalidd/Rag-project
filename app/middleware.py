"""HTTP middleware: request IDs and one access-log line per request.

For every request this middleware:

1. Reuses a safe incoming ``X-Request-ID`` header or creates a new ID.
2. Stores the ID on ``request.state`` and in the logging context.
3. Times the request and logs method, path, status, latency and error code.
4. Returns the ID in the ``X-Request-ID`` response header.

It never logs request bodies, query strings or headers, so questions,
document text and API keys cannot leak into logs from here.
"""

import logging
import re
import time
import uuid

from fastapi import Request, Response
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint

from app.logging_setup import request_id_var

logger = logging.getLogger("app.access")

REQUEST_ID_HEADER = "X-Request-ID"
# Accept client-supplied IDs only if they are short and harmless.
SAFE_REQUEST_ID = re.compile(r"^[A-Za-z0-9\-]{8,64}$")


def new_request_id() -> str:
    """Create a random request ID."""
    return uuid.uuid4().hex


class RequestContextMiddleware(BaseHTTPMiddleware):
    """Assign a request ID, log the request, and catch unexpected crashes."""

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        incoming = request.headers.get(REQUEST_ID_HEADER, "")
        request_id = incoming if SAFE_REQUEST_ID.match(incoming) else new_request_id()
        request.state.request_id = request_id
        token = request_id_var.set(request_id)
        started = time.perf_counter()

        try:
            response = await call_next(request)
        except Exception:
            # Anything not handled by the exception handlers lands here.
            # Log the details server-side; send the client a generic message.
            logger.exception("Unhandled error")
            response = JSONResponse(
                status_code=500,
                content={
                    "error": {
                        "code": "internal_error",
                        "message": "An unexpected error occurred.",
                        "request_id": request_id,
                        "details": {},
                    }
                },
            )

        latency_ms = round((time.perf_counter() - started) * 1000, 1)
        response.headers[REQUEST_ID_HEADER] = request_id
        logger.info(
            "request completed",
            extra={
                "method": request.method,
                "path": request.url.path,  # path only: no query string
                "status": response.status_code,
                "latency_ms": latency_ms,
                "error_code": getattr(request.state, "error_code", None),
            },
        )
        request_id_var.reset(token)
        return response
