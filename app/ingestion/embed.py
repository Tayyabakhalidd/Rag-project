"""Gemini embeddings through Google's OpenAI-compatible endpoint.

Two layers:

* ``GeminiEmbedder`` makes the network call (timeout and bounded retries are
  enforced by the OpenAI SDK) and checks the returned vectors.
* ``CachedEmbedder`` wraps any embedder with the SQLite embedding cache and
  sends texts in batches. Each finished batch is cached immediately, so if a
  free-tier limit stops a large document halfway, the next attempt only embeds
  what is still missing.

Both document chunks and questions go through the same cached embedder. Tests
inject a deterministic fake provider and never touch the network.
"""

import hashlib
import math
from typing import Protocol, Sequence

from openai import APIConnectionError, APITimeoutError, AsyncOpenAI, RateLimitError

from app.config import Settings
from app.errors import UpstreamRateLimitedError, UpstreamUnavailableError


class Embedder(Protocol):
    """Anything that turns a list of texts into a list of vectors."""

    async def embed(self, texts: Sequence[str]) -> list[list[float]]: ...


class EmbeddingCacheStore(Protocol):
    """Storage used by ``CachedEmbedder`` (implemented by ``DocumentRegistry``)."""

    def get_embedding(self, cache_key: str) -> list[float] | None: ...

    def put_embedding(self, cache_key: str, vector: list[float]) -> None: ...


class GeminiEmbedder:
    """Gemini embedding API client, configured with bounded timeout and retries."""

    def __init__(self, settings: Settings) -> None:
        self.model = settings.embedding_model or ""
        self.dimensions = settings.embedding_dimensions
        self.client: AsyncOpenAI | None = None
        # The app can start without keys; a call then fails with a clear 503.
        if settings.gemini_configured:
            self.client = AsyncOpenAI(
                api_key=settings.gemini_api_key.get_secret_value(),
                base_url=settings.gemini_base_url,
                timeout=settings.embedding_timeout_seconds,
                max_retries=settings.external_max_retries,
            )

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        if self.client is None:
            raise UpstreamUnavailableError("Gemini embeddings are not configured (GEMINI_API_KEY, EMBEDDING_MODEL).")
        try:
            result = await self.client.embeddings.create(
                model=self.model,
                input=list(texts),
                dimensions=self.dimensions,
            )
        except RateLimitError as exc:
            raise UpstreamRateLimitedError() from exc
        except (APIConnectionError, APITimeoutError) as exc:
            raise UpstreamUnavailableError() from exc
        except Exception as exc:
            # Avoid returning provider exception text, which can contain request data.
            raise UpstreamUnavailableError("The embedding provider rejected the request.") from exc

        items = list(result.data or [])
        # Gemini's OpenAI-compatible endpoint returns items in input order with
        # ``index`` set to null, so only sort when every item carries an index.
        if items and all(isinstance(item.index, int) for item in items):
            items.sort(key=lambda item: item.index)
        vectors = [list(item.embedding or []) for item in items]
        check_vectors(vectors, len(texts), self.dimensions)
        return vectors


def check_vectors(vectors: list[list[float]], expected_count: int, dimensions: int) -> None:
    """Reject a provider response with the wrong count, size, or non-finite values."""
    if len(vectors) != expected_count or any(
        len(vector) != dimensions or not all(math.isfinite(value) for value in vector) for vector in vectors
    ):
        raise UpstreamUnavailableError("The embedding provider returned vectors with an unexpected shape.")


def embedding_cache_key(model: str, dimensions: int, text: str) -> str:
    """Stable, content-derived cache key; source text is never stored as a key."""
    payload = f"{model}\0{dimensions}\0{text}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


class CachedEmbedder:
    """Cache-first embedder that sends missing texts to the provider in batches."""

    def __init__(
        self,
        provider: Embedder,
        cache: EmbeddingCacheStore,
        model: str,
        dimensions: int,
        batch_size: int = 50,
    ) -> None:
        self.provider = provider
        self.cache = cache
        self.model = model
        self.dimensions = dimensions
        self.batch_size = batch_size
        self.provider_calls = 0  # how many requests reached the provider (for the notebook/eval)

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        keys = [embedding_cache_key(self.model, self.dimensions, text) for text in texts]
        found: dict[str, list[float]] = {}
        missing: dict[str, str] = {}  # cache key -> text, de-duplicated, in first-seen order
        for key, text in zip(keys, texts):
            if key in found or key in missing:
                continue
            vector = self.cache.get_embedding(key)
            if vector is None:
                missing[key] = text
            else:
                found[key] = vector

        pending = list(missing.items())
        for start in range(0, len(pending), self.batch_size):
            batch = pending[start : start + self.batch_size]
            computed = await self.provider.embed([text for _, text in batch])
            self.provider_calls += 1
            if len(computed) != len(batch):
                raise UpstreamUnavailableError("The embedding provider returned an unexpected number of vectors.")
            for (key, _), vector in zip(batch, computed):
                self.cache.put_embedding(key, vector)
                found[key] = vector
        return [found[key] for key in keys]


def build_embedder(settings: Settings, cache: EmbeddingCacheStore, provider: Embedder | None = None) -> CachedEmbedder:
    """The embedder the services use: Gemini (or an injected fake) behind the cache."""
    model = settings.embedding_model or "unconfigured"
    return CachedEmbedder(
        provider or GeminiEmbedder(settings),
        cache,
        model,
        settings.embedding_dimensions,
        settings.embedding_batch_size,
    )
