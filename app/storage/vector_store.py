"""Chroma persistent vector collection adapter.

Chroma stores, for every chunk, its embedding, its text and its citation
metadata. The collection uses cosine distance, so a search result's similarity
score is ``1 - distance`` (1.0 = same direction, 0.0 = unrelated).

Every Chroma failure is turned into ``UpstreamUnavailableError`` (HTTP 503),
because for the API a broken vector database is an unavailable dependency.
"""

import logging
from pathlib import Path
from typing import Any, Sequence

import chromadb
import chromadb.config

from app.errors import UpstreamUnavailableError

logger = logging.getLogger("app.vector_store")

VECTOR_STORE_DOWN = "The vector database is unavailable. Please try again shortly."


class ChromaVectorStore:
    """Thin wrapper around one persistent Chroma collection."""

    def __init__(self, path: Path, collection_name: str = "rag_chunks") -> None:
        path.mkdir(parents=True, exist_ok=True)
        self.collection_name = collection_name
        # Telemetry off: no usage data leaves the machine from the vector store.
        self.client = chromadb.PersistentClient(
            path=str(path), settings=chromadb.config.Settings(anonymized_telemetry=False)
        )
        self.collection = self._open_collection()

    def _open_collection(self):
        return self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def reset(self) -> None:
        """Delete and recreate the collection (used by the evaluation for a fresh index)."""
        try:
            self.client.delete_collection(self.collection_name)
        except Exception:  # the collection may not exist yet
            pass
        self.collection = self._open_collection()

    def upsert(self, chunks, vectors: list[list[float]]) -> None:
        """Store chunks with their vectors; chunk IDs are stable, so re-running is safe."""
        if len(chunks) != len(vectors):
            raise ValueError("Chunk and embedding counts differ.")
        if not chunks:
            return
        # Chroma metadata values cannot be None, so missing page/section keys are left out.
        metadatas: list[dict[str, Any]] = [
            {key: value for key, value in chunk.metadata().items() if value is not None} for chunk in chunks
        ]
        try:
            self.collection.upsert(
                ids=[chunk.chunk_id for chunk in chunks],
                embeddings=vectors,
                documents=[chunk.text for chunk in chunks],
                metadatas=metadatas,
            )
        except Exception as exc:
            logger.warning("Chroma upsert failed: %s", type(exc).__name__)
            raise UpstreamUnavailableError(VECTOR_STORE_DOWN) from exc

    def delete_document(self, document_id: str) -> None:
        """Delete every vector of one document, then confirm none remain."""
        try:
            self.collection.delete(where={"document_id": document_id})
            remaining = self.count_for_document(document_id)
        except UpstreamUnavailableError:
            raise
        except Exception as exc:
            logger.warning("Chroma delete failed: %s", type(exc).__name__)
            raise UpstreamUnavailableError(VECTOR_STORE_DOWN) from exc
        if remaining:
            raise UpstreamUnavailableError("Some vectors could not be deleted. Please retry.")

    def count_for_document(self, document_id: str) -> int:
        try:
            return len(self.collection.get(where={"document_id": document_id}, include=[])["ids"])
        except Exception as exc:
            raise UpstreamUnavailableError(VECTOR_STORE_DOWN) from exc

    def count(self) -> int:
        try:
            return self.collection.count()
        except Exception as exc:
            raise UpstreamUnavailableError(VECTOR_STORE_DOWN) from exc

    def query(
        self, vector: list[float], top_k: int, document_ids: Sequence[str] | None = None
    ) -> list[dict[str, Any]]:
        """Return the ``top_k`` most similar chunks, best first.

        ``document_ids`` restricts the search to those documents (a metadata
        filter applied inside Chroma, before ranking).
        """
        where = {"document_id": {"$in": list(document_ids)}} if document_ids else None
        try:
            if self.collection.count() == 0:
                return []
            result = self.collection.query(
                query_embeddings=[vector],
                n_results=top_k,
                where=where,
                include=["documents", "metadatas", "distances"],
            )
        except Exception as exc:
            logger.warning("Chroma query failed: %s", type(exc).__name__)
            raise UpstreamUnavailableError(VECTOR_STORE_DOWN) from exc

        hits = []
        for chunk_id, text, metadata, distance in zip(
            result["ids"][0], result["documents"][0], result["metadatas"][0], result["distances"][0]
        ):
            hits.append({"chunk_id": chunk_id, "text": text, "metadata": dict(metadata), "score": 1.0 - distance})
        return hits
