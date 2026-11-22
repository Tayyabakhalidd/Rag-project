"""Chroma persistent vector collection adapter."""

from pathlib import Path
from typing import Any

import chromadb


class ChromaVectorStore:
    def __init__(self, path: Path, collection_name: str = "rag_chunks") -> None:
        path.mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(path=str(path))
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def upsert(self, chunks, vectors: list[list[float]]) -> None:
        if len(chunks) != len(vectors):
            raise ValueError("Chunk and embedding counts differ.")
        if not chunks:
            return
        metadatas: list[dict[str, Any]] = []
        for chunk in chunks:
            metadatas.append({key: value for key, value in chunk.metadata().items() if value is not None})
        self.collection.upsert(
            ids=[chunk.chunk_id for chunk in chunks],
            embeddings=vectors,
            documents=[chunk.text for chunk in chunks],
            metadatas=metadatas,
        )

    def delete_document(self, document_id: str) -> None:
        self.collection.delete(where={"document_id": document_id})

    def count_for_document(self, document_id: str) -> int:
        return len(self.collection.get(where={"document_id": document_id}, include=[])["ids"])
