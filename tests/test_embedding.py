"""Embedding tests use only deterministic fake providers (no network)."""

import asyncio

import pytest

from app.config import Settings
from app.errors import UpstreamRateLimitedError, UpstreamUnavailableError
from app.ingestion.embed import CachedEmbedder, GeminiEmbedder, check_vectors
from app.storage.registry import DocumentRegistry


class CountingEmbedder:
    def __init__(self, fail_on_call: int | None = None):
        self.calls = []
        self.fail_on_call = fail_on_call

    async def embed(self, texts):
        self.calls.append(list(texts))
        if self.fail_on_call is not None and len(self.calls) == self.fail_on_call:
            raise UpstreamRateLimitedError()
        return [[float(len(text)), 1.0] for text in texts]


def test_embedding_cache_deduplicates_batch_and_persists(tmp_path):
    registry = DocumentRegistry(tmp_path / "registry.sqlite3")
    provider = CountingEmbedder()
    cached = CachedEmbedder(provider, registry, "fake-model", 2)

    first = asyncio.run(cached.embed(["alpha", "beta", "alpha"]))
    assert first == [[5.0, 1.0], [4.0, 1.0], [5.0, 1.0]]
    assert provider.calls == [["alpha", "beta"]]

    second = asyncio.run(cached.embed(["beta", "alpha"]))
    assert second == [[4.0, 1.0], [5.0, 1.0]]
    assert provider.calls == [["alpha", "beta"]]


def test_texts_are_sent_in_batches(tmp_path):
    provider = CountingEmbedder()
    cached = CachedEmbedder(provider, DocumentRegistry(tmp_path / "r.sqlite3"), "m", 2, batch_size=2)
    asyncio.run(cached.embed(["a1", "b22", "c333", "d4444", "e55555"]))
    assert [len(call) for call in provider.calls] == [2, 2, 1]


def test_finished_batches_stay_cached_after_a_failure(tmp_path):
    registry = DocumentRegistry(tmp_path / "r.sqlite3")
    texts = ["one", "two", "three", "four"]
    failing = CachedEmbedder(CountingEmbedder(fail_on_call=2), registry, "m", 2, batch_size=2)
    with pytest.raises(UpstreamRateLimitedError):
        asyncio.run(failing.embed(texts))

    retry_provider = CountingEmbedder()
    retry = CachedEmbedder(retry_provider, registry, "m", 2, batch_size=2)
    asyncio.run(retry.embed(texts))
    assert retry_provider.calls == [["three", "four"]]  # the first batch came from the cache


def test_cache_key_includes_model(tmp_path):
    registry = DocumentRegistry(tmp_path / "r.sqlite3")
    provider = CountingEmbedder()
    asyncio.run(CachedEmbedder(provider, registry, "model-1", 2).embed(["text"]))
    asyncio.run(CachedEmbedder(provider, registry, "model-2", 2).embed(["text"]))
    assert len(provider.calls) == 2


def test_unconfigured_gemini_fails_with_503_without_network():
    embedder = GeminiEmbedder(Settings(_env_file=None, gemini_api_key=None, embedding_model=None))
    with pytest.raises(UpstreamUnavailableError):
        asyncio.run(embedder.embed(["hello"]))


class _StubEmbeddings:
    """Mimics the OpenAI SDK response of Gemini's compatible endpoint (index is null)."""

    def __init__(self, index_values):
        self.index_values = index_values
        self.kwargs = None

    async def create(self, **kwargs):
        from types import SimpleNamespace

        self.kwargs = kwargs
        data = [SimpleNamespace(index=i, embedding=[float(n), 1.0]) for n, i in enumerate(self.index_values)]
        return SimpleNamespace(data=data)


@pytest.mark.parametrize("index_values, expected_first", [([None, None], 0.0), ([1, 0], 1.0)])
def test_gemini_response_order_with_and_without_indexes(index_values, expected_first):
    settings = Settings(_env_file=None, gemini_api_key="AIzaTESTONLYfakekey1234567890abcdefgh",
                        embedding_model="test-model", embedding_dimensions=2)
    embedder = GeminiEmbedder(settings)
    stub = _StubEmbeddings(index_values)
    embedder.client = type("Client", (), {"embeddings": stub})()
    vectors = asyncio.run(embedder.embed(["first", "second"]))
    assert vectors[0][0] == expected_first  # null indexes keep input order; real indexes are sorted
    assert stub.kwargs["dimensions"] == 2 and stub.kwargs["model"] == "test-model"


@pytest.mark.parametrize(
    "vectors, count",
    [([[1.0, 2.0]], 2), ([[1.0]], 1), ([[float("nan"), 1.0]], 1)],
)
def test_bad_provider_vectors_are_rejected(vectors, count):
    with pytest.raises(UpstreamUnavailableError):
        check_vectors(vectors, count, dimensions=2)
