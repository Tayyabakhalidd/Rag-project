"""Structured logging with request IDs and secret redaction.

Every log line is one JSON object, which makes logs easy to search.

* ``RequestIdFilter`` adds the current request ID (set by the middleware).
* ``JsonFormatter`` redacts configured API keys and key-like strings from the
  finished line (message *and* any exception traceback), so a secret can never
  be written to the log by mistake.
"""

import contextvars
import json
import logging
import re
from datetime import UTC, datetime

# Holds the ID of the request being handled by the current task.
request_id_var: contextvars.ContextVar[str] = contextvars.ContextVar("request_id", default="-")

# Patterns that look like API keys (OpenAI/Groq "sk-"/"gsk_", Google "AIza").
KEY_LIKE_PATTERN = re.compile(r"\b(?:sk-[A-Za-z0-9_\-]{16,}|gsk_[A-Za-z0-9]{16,}|AIza[0-9A-Za-z_\-]{30,})")
REDACTED = "[REDACTED]"


def redact(text: str, secrets: list[str]) -> str:
    """Remove known secret values and key-like strings from ``text``."""
    for secret in secrets:
        if secret:
            text = text.replace(secret, REDACTED)
    return KEY_LIKE_PATTERN.sub(REDACTED, text)


class RequestIdFilter(logging.Filter):
    """Attach the current request ID to each log record."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = request_id_var.get()
        return True


class JsonFormatter(logging.Formatter):
    """Format a log record as a single, redacted JSON line."""

    # Extra fields our code may attach with ``logger.info(..., extra={...})``.
    EXTRA_FIELDS = ("method", "path", "status", "latency_ms", "error_code")

    def __init__(self, secrets: list[str] | None = None) -> None:
        super().__init__()
        self._secrets = secrets or []

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "time": datetime.fromtimestamp(record.created, UTC).isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "logger": record.name,
            "request_id": getattr(record, "request_id", "-"),
            "message": record.getMessage(),
        }
        for field in self.EXTRA_FIELDS:
            value = getattr(record, field, None)
            if value is not None:
                payload[field] = value
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return redact(json.dumps(payload, ensure_ascii=False), self._secrets)


def configure_logging(level: str, secrets: list[str]) -> None:
    """Configure the ``app`` logger once at startup."""
    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter(secrets))
    handler.addFilter(RequestIdFilter())

    logger = logging.getLogger("app")
    logger.handlers.clear()
    logger.addHandler(handler)
    logger.setLevel(level.upper())
    logger.propagate = False
