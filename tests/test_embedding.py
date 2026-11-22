"""Embedding cache tests use only a deterministic fake provider."""

import asyncio

from app.ingestion.embed import CachedEmbedder
from app.storage.registry import DocumentRegistry


class CountingEmbedder:
    def __init__(self):
        self.calls = []

    async def embed(self, texts):
        self.calls.append(list(texts))
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
