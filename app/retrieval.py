"""Retrieval: embed the question, search Chroma, return scored chunks.

The question is embedded with the same cached Gemini embedder as the
documents (a question asked twice costs one API call). Chroma ranks chunks by
cosine similarity; an optional ``document_ids`` filter limits the search to
chosen documents. Thresholds are applied later by the QA service, so the
evaluation can measure raw retrieval quality.
"""

import asyncio
from dataclasses import dataclass
from typing import Any, Sequence

from app.ingestion.embed import Embedder
from app.storage.vector_store import ChromaVectorStore


@dataclass(frozen=True)
class RetrievedChunk:
    """One search result with everything needed to show and cite it."""

    rank: int  # 1 = most similar
    chunk_id: str
    document_id: str
    filename: str
    file_type: str
    page: int | None
    section: str | None
    chunk_index: int
    text: str
    score: float  # cosine similarity, higher is more similar

    @property
    def citation_id(self) -> str:
        """Label used in the prompt and in answers: C1, C2, ..."""
        return f"C{self.rank}"

    @property
    def source_label(self) -> str:
        """Human-readable location, e.g. 'NIST.AI.100-1.pdf, page 12'."""
        parts = [self.filename]
        if self.page is not None:
            parts.append(f"page {self.page}")
        if self.section:
            parts.append(f"section {self.section}")
        return ", ".join(parts)

    @classmethod
    def from_hit(cls, rank: int, hit: dict[str, Any]) -> "RetrievedChunk":
        meta = hit["metadata"]
        return cls(
            rank=rank,
            chunk_id=hit["chunk_id"],
            document_id=meta.get("document_id", ""),
            filename=meta.get("filename", ""),
            file_type=meta.get("file_type", ""),
            page=meta.get("page"),
            section=meta.get("section"),
            chunk_index=int(meta.get("chunk_index", 0)),
            text=hit["text"],
            score=round(float(hit["score"]), 4),
        )


class Retriever:
    """Semantic search over the vector store."""

    def __init__(self, vectors: ChromaVectorStore, embedder: Embedder) -> None:
        self.vectors = vectors
        self.embedder = embedder

    async def search(
        self, query: str, top_k: int, document_ids: Sequence[str] | None = None
    ) -> list[RetrievedChunk]:
        """Return up to ``top_k`` chunks, most similar first."""
        [vector] = await self.embedder.embed([query])
        hits = await asyncio.to_thread(self.vectors.query, vector, top_k, document_ids)
        return [RetrievedChunk.from_hit(rank, hit) for rank, hit in enumerate(hits, start=1)]
