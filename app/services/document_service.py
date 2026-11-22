"""Orchestrates document ingestion, listing and deletion.

Upload is all-or-nothing:

1. ``prepare_document`` validates, extracts and chunks (CPU work, in a thread).
2. The content hash is reserved in SQLite; a second copy gets 409.
3. Chunks are embedded (cache first) and written to Chroma.
4. The registry row is marked ``ready``.

If step 3 or 4 fails, the vectors written so far and the reserved row are
removed, so no half-indexed document is ever listed or searched.
"""

import asyncio

from app.config import Settings
from app.errors import DocumentNotFoundError, DuplicateDocumentError
from app.ingestion.embed import Embedder
from app.ingestion.pipeline import prepare_document
from app.storage.registry import DocumentRegistry, document_record
from app.storage.vector_store import ChromaVectorStore


class DocumentService:
    """Business logic behind the document endpoints and the upload form."""

    def __init__(
        self, settings: Settings, registry: DocumentRegistry, vectors: ChromaVectorStore, embedder: Embedder
    ) -> None:
        self.settings = settings
        self.registry = registry
        self.vectors = vectors
        self.embedder = embedder  # normally a CachedEmbedder (see app.ingestion.embed.build_embedder)

    async def upload(self, filename: str | None, data: bytes) -> dict:
        """Ingest one file and return its registry record."""
        prepared = await asyncio.to_thread(prepare_document, filename, data, self.settings)
        if not await asyncio.to_thread(self.registry.reserve, document_record(prepared)):
            existing = self.registry.get(prepared.document_id)
            details = {"document_id": prepared.document_id}
            if existing:
                details["filename"] = existing["filename"]
            raise DuplicateDocumentError(details=details)
        try:
            vectors = await self.embedder.embed([chunk.text for chunk in prepared.chunks])
            await asyncio.to_thread(self.vectors.upsert, prepared.chunks, vectors)
            await asyncio.to_thread(self.registry.mark_ready, prepared.document_id)
        except BaseException:
            # Keep registry and Chroma consistent if embedding or persistence fails
            # (BaseException also covers a cancelled request).
            try:
                await asyncio.to_thread(self.vectors.delete_document, prepared.document_id)
            finally:
                await asyncio.to_thread(self.registry.remove, prepared.document_id)
            raise
        return self.registry.get(prepared.document_id) or {}

    def list_documents(self) -> list[dict]:
        return self.registry.list_ready()

    def get_document(self, document_id: str) -> dict:
        document = self.registry.get(document_id)
        if document is None:
            raise DocumentNotFoundError(details={"document_id": document_id})
        return document

    async def delete(self, document_id: str) -> dict:
        """Delete vectors first (and confirm), then the registry row.

        If the vector delete fails, a 503 is raised and the row is kept, so the
        user can simply retry.
        """
        document = self.get_document(document_id)
        await asyncio.to_thread(self.vectors.delete_document, document_id)
        await asyncio.to_thread(self.registry.remove, document_id)
        return document
