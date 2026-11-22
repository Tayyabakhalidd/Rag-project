"""Shared pytest fixtures.

Tests build the app with their own ``Settings`` and ``_env_file=None`` so the
real ``.env`` file (and any real API key) is never read during testing. The
embedding API and the LLM are replaced by deterministic fakes, so the suite
makes no network calls and costs nothing.
"""

import hashlib
import json
import math
import re

import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app

FAKE_GROQ_KEY = "gsk_TESTONLYfakekey1234567890abcdef"
FAKE_GEMINI_KEY = "AIzaTESTONLYfakekey1234567890abcdefgh"
TEST_DIMENSIONS = 256  # large enough that unrelated words rarely share a slot


class FakeEmbedder:
    """Bag-of-words hashing embedder: texts that share words get similar vectors.

    Deterministic and offline, but still meaningful for retrieval tests: a
    question shares words with the chunk that answers it.
    """

    def __init__(self, dimensions: int = TEST_DIMENSIONS):
        self.dimensions = dimensions
        self.calls: list[list[str]] = []
        self.error: Exception | None = None  # set to make every call fail

    async def embed(self, texts):
        self.calls.append(list(texts))
        if self.error:
            raise self.error
        return [self.vector(text) for text in texts]

    def vector(self, text: str) -> list[float]:
        values = [0.0] * self.dimensions
        for word in re.findall(r"[a-z0-9]{3,}", text.lower()):
            values[int(hashlib.md5(word.encode()).hexdigest(), 16) % self.dimensions] += 1.0
        norm = math.sqrt(sum(v * v for v in values)) or 1.0
        return [v / norm for v in values] if any(values) else [1.0] + [0.0] * (self.dimensions - 1)


class FakeLLM:
    """Scripted LLM. By default it answers from chunk C1 and cites it."""

    def __init__(self):
        self.calls: list[list[dict]] = []
        self.replies: list[str | Exception] = []  # consumed first, one per call

    async def complete(self, messages):
        self.calls.append(messages)
        if self.replies:
            reply = self.replies.pop(0)
            if isinstance(reply, Exception):
                raise reply
            return reply
        return json.dumps(
            {
                "off_topic": False,
                "answerable": True,
                "answer": "The documents describe this directly [C1].",
                "citations": ["C1"],
                "general_knowledge": None,
            }
        )

    def reply_with(self, **fields) -> None:
        base = {"off_topic": False, "answerable": False, "answer": "", "citations": [], "general_knowledge": None}
        base.update(fields)
        self.replies.append(json.dumps(base))


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
        embedding_dimensions=TEST_DIMENSIONS,
        # Thresholds suited to the bag-of-words fake embedder.
        off_topic_floor=0.10,
        answer_threshold=0.20,
    )


@pytest.fixture
def fake_embedder() -> FakeEmbedder:
    return FakeEmbedder()


@pytest.fixture
def fake_llm() -> FakeLLM:
    return FakeLLM()


@pytest.fixture
def app(settings, fake_embedder, fake_llm):
    """A fresh FastAPI app per test, wired to the fakes."""
    return create_app(settings, embedder=fake_embedder, llm=fake_llm)


@pytest.fixture
def client(app) -> TestClient:
    """HTTP client that calls the app in-process (no server needed)."""
    return TestClient(app)
