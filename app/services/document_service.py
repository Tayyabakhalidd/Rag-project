"""Orchestrates document ingestion, listing, and deletion."""

import asyncio

from app.config import Settings
from app.errors import DocumentNotFoundError, DuplicateDocumentError, UpstreamUnavailableError
from app.ingestion.embed import CachedEmbedder, Embedder, GeminiEmbedder
from app.ingestion.pipeline import prepare_document
from app.storage.registry import DocumentRegistry, document_record
from app.storage.vector_store import ChromaVectorStore


class DocumentService:
    def __init__(self, settings: Settings, registry: DocumentRegistry, vectors: ChromaVectorStore,
                 embedder: Embedder | None = None) -> None:
        self.settings = settings
        self.registry = registry
        self.vectors = vectors
        self._embedder = embedder

    def _cached_embedder(self):
        if self._embedder is None:
            provider = GeminiEmbedder(self.settings)
            return CachedEmbedder(provider, self.registry, provider.model, provider.dimensions)
        return CachedEmbedder(self._embedder, self.registry, self.settings.embedding_model or "test", self.settings.embedding_dimensions)

    async def upload(self, filename: str | None, data: bytes) -> dict:
        prepared = await asyncio.to_thread(prepare_document, filename, data, self.settings)
        if not self.registry.reserve(document_record(prepared)):
            raise DuplicateDocumentError(details={"document_id": prepared.document_id})
        try:
            vectors = await self._cached_embedder().embed([chunk.text for chunk in prepared.chunks])
            await asyncio.to_thread(self.vectors.upsert, prepared.chunks, vectors)
            await asyncio.to_thread(self.registry.mark_ready, prepared.document_id)
        except Exception:
            # Keep registry and Chroma consistent if embedding or persistence fails.
            try:
                await asyncio.to_thread(self.vectors.delete_document, prepared.document_id)
            finally:
                await asyncio.to_thread(self.registry.remove, prepared.document_id)
            raise
        return self.registry.get(prepared.document_id) or {}

    def list_documents(self) -> list[dict]:
        return self.registry.list_ready()

    async def delete(self, document_id: str) -> None:
        if self.registry.get(document_id) is None:
            raise DocumentNotFoundError(details={"document_id": document_id})
        await asyncio.to_thread(self.vectors.delete_document, document_id)
        await asyncio.to_thread(self.registry.remove, document_id)
