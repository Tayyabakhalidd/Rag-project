"""Shared pytest fixtures.

Tests build the app with their own ``Settings`` and ``_env_file=None`` so the
real ``.env`` file (and any real API key) is never read during testing.
"""

import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app
from app.storage.registry import DocumentRegistry
from app.storage.vector_store import ChromaVectorStore
from app.services.document_service import DocumentService


class FakeEmbedder:
    """Deterministic test provider; never makes a network request."""

    def __init__(self, dimensions: int = 768):
        self.dimensions = dimensions
        self.calls = []

    async def embed(self, texts):
        self.calls.append(list(texts))
        vectors = []
        for text in texts:
            # Stable, non-zero vector based on the text, padded to test dimensions.
            value = (sum(text.encode("utf-8")) % 997 + 1) / 997
            vectors.append([value] + [0.0] * (self.dimensions - 1))
        return vectors

FAKE_GROQ_KEY = "gsk_TESTONLYfakekey1234567890abcdef"
FAKE_GEMINI_KEY = "AIzaTESTONLYfakekey1234567890abcdefgh"


@pytest.fixture
def settings(tmp_path) -> Settings:
    """Settings for tests: fake keys, temporary data folder, no .env file."""
    return Settings(
        _env_file=None,
        environment="test",
        data_dir=tmp_path / "data",
        groq_api_key=FAKE_GROQ_KEY,
        llm_model="test-answer-model",
        judge_model="test-judge-model",
        gemini_api_key=FAKE_GEMINI_KEY,
        embedding_model="test-embedding-model",
    )


@pytest.fixture
def app(settings):
    """A fresh FastAPI app per test."""
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    registry = DocumentRegistry(settings.data_dir / "documents.sqlite3")
    vectors = ChromaVectorStore(settings.data_dir / "chroma")
    service = DocumentService(settings, registry, vectors, FakeEmbedder(settings.embedding_dimensions))
    return create_app(settings, service)


@pytest.fixture
def client(app) -> TestClient:
    """HTTP client that calls the app in-process (no server needed)."""
    return TestClient(app)
