"""Shared pytest fixtures.

Tests build the app with their own ``Settings`` and ``_env_file=None`` so the
real ``.env`` file (and any real API key) is never read during testing.
"""

import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app

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
    return create_app(settings)


@pytest.fixture
def client(app) -> TestClient:
    """HTTP client that calls the app in-process (no server needed)."""
    return TestClient(app)
