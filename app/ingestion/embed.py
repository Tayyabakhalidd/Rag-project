"""Gemini embeddings through Google's OpenAI-compatible endpoint.

The adapter is injectable, so tests can use a deterministic fake and never need
network access. Successful vectors are cached in the document registry.
"""

import hashlib
import math
from typing import Protocol, Sequence

from openai import APIConnectionError, APITimeoutError, AsyncOpenAI, RateLimitError

from app.config import Settings
from app.errors import UpstreamRateLimitedError, UpstreamUnavailableError


class Embedder(Protocol):
    async def embed(self, texts: Sequence[str]) -> list[list[float]]: ...


class GeminiEmbedder:
    """Gemini embedding API client, configured with bounded timeout/retries."""

    def __init__(self, settings: Settings) -> None:
        if not settings.gemini_configured:
            raise UpstreamUnavailableError("Gemini embeddings are not configured.")
        self.model = settings.embedding_model or ""
        self.dimensions = settings.embedding_dimensions
        self.client = AsyncOpenAI(
            api_key=settings.gemini_api_key.get_secret_value(),
            base_url=settings.gemini_base_url,
            timeout=settings.embedding_timeout_seconds,
            max_retries=settings.external_max_retries,
        )

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
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

        vectors = [item.embedding for item in sorted(result.data, key=lambda item: item.index)]
        if len(vectors) != len(texts) or any(
            len(vector) != self.dimensions or not all(math.isfinite(value) for value in vector)
            for vector in vectors
        ):
            raise UpstreamUnavailableError("The embedding provider returned vectors with an unexpected shape.")
        return vectors


def embedding_cache_key(model: str, dimensions: int, text: str) -> str:
    """Stable, content-derived cache key; source text is never stored as a key."""
    payload = f"{model}\0{dimensions}\0{text}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


class CachedEmbedder:
    """Cache successful embeddings using the registry's SQLite cache methods."""

    def __init__(self, provider: Embedder, registry, model: str, dimensions: int) -> None:
        self.provider = provider
        self.registry = registry
        self.model = model
        self.dimensions = dimensions

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        keys = [embedding_cache_key(self.model, self.dimensions, text) for text in texts]
        vectors: list[list[float] | None] = [self.registry.get_embedding(key) for key in keys]
        missing_positions = [i for i, vector in enumerate(vectors) if vector is None]
        if missing_positions:
            # Avoid duplicate provider calls for repeated chunk text in one batch.
            unique_texts: list[str] = []
            unique_keys: list[str] = []
            positions_by_key: dict[str, list[int]] = {}
            for i in missing_positions:
                if keys[i] not in positions_by_key:
                    positions_by_key[keys[i]] = []
                    unique_keys.append(keys[i])
                    unique_texts.append(texts[i])
                positions_by_key[keys[i]].append(i)
            computed = await self.provider.embed(unique_texts)
            if len(computed) != len(unique_texts):
                raise UpstreamUnavailableError("The embedding provider returned an unexpected number of vectors.")
            for key, vector in zip(unique_keys, computed):
                self.registry.put_embedding(key, vector)
                for position in positions_by_key[key]:
                    vectors[position] = vector
        return [vector for vector in vectors if vector is not None]
