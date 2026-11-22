"""API tests through HTTP: success cases and expected failures."""

import json
import logging

from pydantic import BaseModel

from app.config import Settings
from app.errors import DocumentNotFoundError, UpstreamUnavailableError
from app.logging_setup import JsonFormatter, redact
from app.main import create_app
from fastapi.testclient import TestClient
from tests.conftest import FAKE_GEMINI_KEY, FAKE_GROQ_KEY
from tests.factories import make_pdf


def assert_error_shape(body: dict, code: str) -> None:
    """Every error must use the shared shape."""
    assert set(body) == {"error"}
    assert set(body["error"]) == {"code", "message", "request_id", "details"}
    assert body["error"]["code"] == code


# --- /health ---------------------------------------------------------------

def test_health_ok(client):
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["llm_configured"] is True
    assert body["embeddings_configured"] is True


def test_health_reports_missing_configuration(tmp_path):
    bare = Settings(_env_file=None, environment="test", data_dir=tmp_path)
    response = TestClient(create_app(bare)).get("/health")
    assert response.status_code == 200
    assert response.json()["llm_configured"] is False
    assert response.json()["embeddings_configured"] is False


def test_health_never_returns_key_values(client):
    text = client.get("/health").text
    assert FAKE_GROQ_KEY not in text
    assert FAKE_GEMINI_KEY not in text


def test_openapi_docs_build_and_list_health(client):
    response = client.get("/openapi.json")
    assert response.status_code == 200
    assert "/health" in response.json()["paths"]
    assert client.get("/docs").status_code == 200


# --- Document ingestion API -----------------------------------------------

def test_upload_list_duplicate_and_delete_document(client, app):
    payload = make_pdf([["Production RAG service security guidance with enough text to index."]])
    upload = client.post("/api/v1/documents", files={"file": ("guide.pdf", payload, "application/pdf")})
    assert upload.status_code == 201, upload.text
    document = upload.json()
    assert document["filename"] == "guide.pdf"
    assert document["chunk_count"] >= 1
    document_id = document["document_id"]
    service = app.state.document_service
    assert service.vectors.count_for_document(document_id) == document["chunk_count"]

    duplicate = client.post("/api/v1/documents", files={"file": ("renamed.pdf", payload, "application/pdf")})
    assert duplicate.status_code == 409
    assert_error_shape(duplicate.json(), "duplicate_document")
    assert duplicate.json()["error"]["details"]["document_id"] == document_id

    listing = client.get("/api/v1/documents")
    assert listing.status_code == 200
    assert listing.json()["total"] == 1
    assert listing.json()["documents"][0]["document_id"] == document_id

    deleted = client.delete(f"/api/v1/documents/{document_id}")
    assert deleted.status_code == 204
    assert service.vectors.count_for_document(document_id) == 0
    assert client.get("/api/v1/documents").json() == {"documents": [], "total": 0}
    missing = client.delete(f"/api/v1/documents/{document_id}")
    assert missing.status_code == 404
    assert_error_shape(missing.json(), "document_not_found")


def test_document_upload_returns_validation_error(client):
    response = client.post("/api/v1/documents", files={"file": ("notes.txt", b"hello", "text/plain")})
    assert response.status_code == 415
    assert_error_shape(response.json(), "unsupported_file_type")


def test_openapi_lists_document_endpoints(client):
    paths = client.get("/openapi.json").json()["paths"]
    assert "/api/v1/documents" in paths
    assert {"post", "get"} <= set(paths["/api/v1/documents"])
    assert "delete" in paths["/api/v1/documents/{document_id}"]


# --- Request IDs -----------------------------------------------------------

def test_response_has_generated_request_id(client):
    response = client.get("/health")
    assert len(response.headers["X-Request-ID"]) == 32


def test_safe_incoming_request_id_is_reused(client):
    response = client.get("/health", headers={"X-Request-ID": "demo-request-0001"})
    assert response.headers["X-Request-ID"] == "demo-request-0001"


def test_unsafe_incoming_request_id_is_replaced(client):
    response = client.get("/health", headers={"X-Request-ID": "bad id <script>"})
    assert response.headers["X-Request-ID"] != "bad id <script>"


# --- Error shape -----------------------------------------------------------

def test_unknown_route_returns_404_error_shape(client):
    response = client.get("/does-not-exist")
    assert response.status_code == 404
    body = response.json()
    assert_error_shape(body, "not_found")
    assert body["error"]["request_id"] == response.headers["X-Request-ID"]


def test_wrong_method_returns_405_error_shape(client):
    response = client.post("/health")
    assert response.status_code == 405
    assert_error_shape(response.json(), "method_not_allowed")


def test_app_errors_map_to_status_and_shape(app):
    @app.get("/_test/missing")
    async def missing():
        raise DocumentNotFoundError(details={"document_id": "abc"})

    @app.get("/_test/upstream")
    async def upstream():
        raise UpstreamUnavailableError()

    client = TestClient(app)
    response = client.get("/_test/missing")
    assert response.status_code == 404
    assert_error_shape(response.json(), "document_not_found")
    assert response.json()["error"]["details"] == {"document_id": "abc"}

    response = client.get("/_test/upstream")
    assert response.status_code == 503
    assert_error_shape(response.json(), "upstream_unavailable")


def test_validation_error_uses_shape_and_does_not_echo_input(app):
    class Body(BaseModel):
        question: str
        top_k: int

    @app.post("/_test/validate")
    async def validate(body: Body):
        return {"ok": True}

    response = TestClient(app).post("/_test/validate", json={"question": "secret question text", "top_k": "x"})
    assert response.status_code == 422
    body = response.json()
    assert_error_shape(body, "validation_error")
    assert body["error"]["details"]["problems"][0]["location"] == "body.top_k"
    assert "secret question text" not in response.text


def test_unexpected_exception_returns_generic_500(app):
    @app.get("/_test/crash")
    async def crash():
        raise RuntimeError("internal detail that must not leak")

    response = TestClient(app, raise_server_exceptions=False).get("/_test/crash")
    assert response.status_code == 500
    assert_error_shape(response.json(), "internal_error")
    assert "internal detail" not in response.text


# --- Logging safety --------------------------------------------------------

def test_redact_removes_configured_and_key_like_secrets():
    text = f"key={FAKE_GROQ_KEY} other=sk-abcdefghijklmnopqrstuvwx custom=mysecretvalue"
    cleaned = redact(text, ["mysecretvalue"])
    assert FAKE_GROQ_KEY not in cleaned
    assert "sk-abcdefghijklmnopqrstuvwx" not in cleaned
    assert "mysecretvalue" not in cleaned
    assert cleaned.count("[REDACTED]") == 3


def test_json_formatter_redacts_secrets_in_messages():
    formatter = JsonFormatter(secrets=[FAKE_GEMINI_KEY])
    record = logging.LogRecord("app", logging.INFO, __file__, 1, "using key %s", (FAKE_GEMINI_KEY,), None)
    line = formatter.format(record)
    assert FAKE_GEMINI_KEY not in line
    assert json.loads(line)["message"] == "using key [REDACTED]"
