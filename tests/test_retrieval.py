"""Retrieval tests: a real temporary Chroma collection with the fake embedder."""

import asyncio

import pytest

from app.ingestion.embed import CachedEmbedder
from app.ingestion.models import Chunk
from app.retrieval import Retriever
from app.storage.registry import DocumentRegistry
from app.storage.vector_store import ChromaVectorStore
from tests.conftest import FakeEmbedder

TEXTS = {
    "doc_a": [
        "The GOVERN function cultivates a culture of risk management.",
        "The MEASURE function uses quantitative and qualitative tools to analyze risk.",
    ],
    "doc_b": ["Prompt injection manipulates a large language model through crafted inputs."],
}


def make_chunk(document_id: str, index: int, text: str, page: int) -> Chunk:
    return Chunk(
        chunk_id=f"{document_id}:{index}", document_id=document_id, filename=f"{document_id}.pdf", file_type="pdf",
        chunk_index=index, text=text, page=page, section=None, section_order=index, char_start=0,
        char_end=len(text), sha256="0" * 64, chunk_config="A",
    )


@pytest.fixture
def retriever(tmp_path):
    vectors = ChromaVectorStore(tmp_path / "chroma")
    embedder = FakeEmbedder()
    for document_id, texts in TEXTS.items():
        chunks = [make_chunk(document_id, i, text, page=i + 1) for i, text in enumerate(texts)]
        vectors.upsert(chunks, asyncio.run(embedder.embed([c.text for c in chunks])))
    return Retriever(vectors, embedder)


def test_search_ranks_the_matching_chunk_first(retriever):
    results = asyncio.run(retriever.search("What does the MEASURE function analyze?", top_k=3))
    assert [r.rank for r in results] == [1, 2, 3]
    assert results[0].chunk_id == "doc_a:1"
    assert results[0].page == 2 and results[0].filename == "doc_a.pdf"
    assert results[0].citation_id == "C1"
    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True)
    assert all(-1.0 <= s <= 1.0 for s in scores)


def test_document_filter_limits_results(retriever):
    results = asyncio.run(retriever.search("MEASURE function risk", top_k=5, document_ids=["doc_b"]))
    assert {r.document_id for r in results} == {"doc_b"}


def test_top_k_larger_than_collection_returns_everything(retriever):
    assert len(asyncio.run(retriever.search("risk", top_k=10))) == 3


def test_deleted_document_is_never_returned(retriever):
    retriever.vectors.delete_document("doc_b")
    assert retriever.vectors.count_for_document("doc_b") == 0
    results = asyncio.run(retriever.search("prompt injection crafted inputs", top_k=5))
    assert all(r.document_id != "doc_b" for r in results)


def test_empty_collection_returns_no_results(tmp_path):
    retriever = Retriever(ChromaVectorStore(tmp_path / "empty"), FakeEmbedder())
    assert asyncio.run(retriever.search("anything", top_k=5)) == []


def test_query_embeddings_use_the_cache(tmp_path):
    provider = FakeEmbedder()
    cached = CachedEmbedder(provider, DocumentRegistry(tmp_path / "r.sqlite3"), "m", provider.dimensions)
    retriever = Retriever(ChromaVectorStore(tmp_path / "chroma"), cached)
    asyncio.run(retriever.search("same question", top_k=1))
    asyncio.run(retriever.search("same question", top_k=1))
    assert provider.calls == [["same question"]]
