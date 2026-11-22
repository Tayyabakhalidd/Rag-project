"""Retrieval and answer metrics (pure functions, unit-tested in tests/test_metrics.py).

Relevance is decided by source, not by chunk ID: a retrieved chunk is
relevant to an expected source if it comes from the same file and from one of
the listed pages. That keeps one golden set valid for both chunking
configurations, whose chunk IDs differ.

Definitions, for one question and a cut-off ``k``:

* hit@k: 1 if any of the top-k chunks matches any expected source, else 0.
* recall@k: share of the expected sources matched by at least one top-k chunk
  (a multi-part question with two sources scores 0.5 if only one is found).
* reciprocal rank (RR): 1 / rank of the first relevant chunk, 0 if none in the
  retrieved list. MRR is the mean RR over questions.
"""

from statistics import mean
from typing import Any, Iterable, Sequence


def matches(chunk: dict[str, Any], source: dict[str, Any]) -> bool:
    """True if ``chunk`` (filename, page) satisfies one expected source."""
    return chunk.get("filename") == source["filename"] and chunk.get("page") in source["pages"]


def hit_at_k(retrieved: Sequence[dict[str, Any]], sources: Sequence[dict[str, Any]], k: int) -> float:
    top = retrieved[:k]
    return 1.0 if any(matches(chunk, source) for chunk in top for source in sources) else 0.0


def recall_at_k(retrieved: Sequence[dict[str, Any]], sources: Sequence[dict[str, Any]], k: int) -> float:
    if not sources:
        return 0.0
    top = retrieved[:k]
    found = sum(1 for source in sources if any(matches(chunk, source) for chunk in top))
    return found / len(sources)


def reciprocal_rank(retrieved: Sequence[dict[str, Any]], sources: Sequence[dict[str, Any]]) -> float:
    for rank, chunk in enumerate(retrieved, start=1):
        if any(matches(chunk, source) for source in sources):
            return 1.0 / rank
    return 0.0


def retrieval_summary(rows: Iterable[dict[str, Any]], ks: Sequence[int] = (1, 3, 5)) -> dict[str, float]:
    """Average the per-question metrics in ``rows`` (each has hit@k, recall@k and rr keys)."""
    rows = list(rows)
    if not rows:
        return {}
    summary = {f"hit@{k}": mean(r[f"hit@{k}"] for r in rows) for k in ks}
    summary.update({f"recall@{k}": mean(r[f"recall@{k}"] for r in rows) for k in ks})
    summary["mrr"] = mean(r["rr"] for r in rows)
    summary["questions"] = len(rows)
    return summary


def per_question_retrieval(retrieved: Sequence[dict[str, Any]], sources: Sequence[dict[str, Any]],
                           ks: Sequence[int] = (1, 3, 5)) -> dict[str, float]:
    row = {f"hit@{k}": hit_at_k(retrieved, sources, k) for k in ks}
    row.update({f"recall@{k}": recall_at_k(retrieved, sources, k) for k in ks})
    row["rr"] = reciprocal_rank(retrieved, sources)
    return row


def rate(values: Iterable[bool]) -> float | None:
    """Share of True values, or None when there is nothing to measure."""
    values = list(values)
    return sum(values) / len(values) if values else None
