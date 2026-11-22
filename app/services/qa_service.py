"""Question answering: guardrails -> retrieval -> gate -> LLM -> checks.

``QAService.ask`` turns one question into one of these outcomes:

=====================  ==========================================  ==========================
Outcome                How it is detected                          Response
=====================  ==========================================  ==========================
answered               best score >= answer threshold, the LLM     200, cited answer
                       says answerable, >= 1 valid citation
not_in_documents       the LLM says not answerable, the best       200, "documents don't
                       score is below the answer threshold, or     contain this" + labelled
                       no valid citation survives                  general knowledge
off_topic              best score < off-topic floor (no LLM        200, polite refusal only
                       call), or the LLM flags the topic
no_documents           nothing has been uploaded yet               200, refusal
output_blocked         the output check found a secret or a        200, answer withheld
                       leaked system prompt
injection/jailbreak    input guardrail patterns                    400 guardrail_blocked
=====================  ==========================================  ==========================
"""

import hashlib
import logging
import time
from dataclasses import dataclass, field
from typing import Sequence

from app.config import Settings
from app.errors import DocumentNotFoundError, GuardrailBlockedError
from app.generation import (
    PROMPT_FINGERPRINTS,
    LLMClient,
    build_messages,
    new_canary,
    parse_llm_output,
    validate_citations,
)
from app.guardrails import check_output, check_question
from app.retrieval import RetrievedChunk, Retriever
from app.storage.registry import DocumentRegistry

logger = logging.getLogger("app.qa")

NO_DOCUMENTS_MESSAGE = "No documents have been uploaded yet. Upload a PDF or DOCX file, then ask again."
NOT_IN_DOCUMENTS_MESSAGE = "The uploaded documents don't contain the answer to this question."
OFF_TOPIC_MESSAGE = (
    "Sorry, I can only answer questions about the uploaded documents, which cover {topic}. "
    "This question is outside that subject."
)
OUTPUT_BLOCKED_MESSAGE = "The answer was withheld because it failed a safety check. Please rephrase the question."


@dataclass
class AskResult:
    """Everything the API and the UI need to show an answer."""

    answerable: bool
    answer: str
    general_knowledge: str | None
    refusal_reason: str | None  # not_in_documents | off_topic | no_documents | output_blocked
    citations: list[RetrievedChunk]
    retrieved: list[RetrievedChunk]
    cited_ids: set[str] = field(default_factory=set)
    latency_ms: float = 0.0
    # Diagnostics used by the evaluation (not all are returned by the API).
    llm_called: bool = False
    raw_citation_count: int = 0
    invalid_citation_count: int = 0

    @property
    def best_score(self) -> float | None:
        return self.retrieved[0].score if self.retrieved else None


class QAService:
    """Answers questions over the indexed documents."""

    def __init__(self, settings: Settings, registry: DocumentRegistry, retriever: Retriever, llm: LLMClient) -> None:
        self.settings = settings
        self.registry = registry
        self.retriever = retriever
        self.llm = llm
        self.canary = new_canary()

    def _check_filter(self, document_ids: Sequence[str] | None) -> None:
        if document_ids:
            missing = self.registry.missing_ids(list(document_ids))
            if missing:
                raise DocumentNotFoundError(details={"document_ids": missing})

    async def search(
        self, query: str, top_k: int | None = None, document_ids: Sequence[str] | None = None
    ) -> list[RetrievedChunk]:
        """Retrieval only (no LLM): used by POST /api/v1/search and the evaluation."""
        self._check_filter(document_ids)
        if self.registry.count_ready() == 0:
            return []
        return await self.retriever.search(query, top_k or self.settings.top_k, document_ids)

    async def ask(
        self, question: str, top_k: int | None = None, document_ids: Sequence[str] | None = None
    ) -> AskResult:
        started = time.perf_counter()
        # Log the question by length and hash only, never its text.
        logger.info(
            "question received chars=%d hash=%s",
            len(question),
            hashlib.sha256(question.encode("utf-8")).hexdigest()[:12],
        )

        verdict = check_question(question)
        if verdict.blocked:
            logger.warning("question blocked reason=%s", verdict.reason)
            raise GuardrailBlockedError(
                "The question was blocked because it looks like an attempt to override the assistant's "
                "instructions or extract hidden information.",
                details={"reason": verdict.reason},
            )

        self._check_filter(document_ids)
        if self.registry.count_ready() == 0:
            return self._finish(started, _refusal("no_documents", NO_DOCUMENTS_MESSAGE, []))

        chunks = await self.retriever.search(question, top_k or self.settings.top_k, document_ids)
        if not chunks:
            return self._finish(started, _refusal("no_documents", NO_DOCUMENTS_MESSAGE, []))

        off_topic = OFF_TOPIC_MESSAGE.format(topic=self.settings.corpus_topic)
        if chunks[0].score < self.settings.off_topic_floor:
            # Nothing is even loosely related: refuse without spending an LLM call.
            return self._finish(started, _refusal("off_topic", off_topic, chunks))

        raw = await self.llm.complete(build_messages(question, chunks, self.settings.corpus_topic, self.canary))
        output = parse_llm_output(raw)
        if output.off_topic:
            return self._finish(started, _refusal("off_topic", off_topic, chunks, llm_called=True))

        check = validate_citations(output, chunks)
        blocked = check_output(
            [check.answer, check.general_knowledge], self.settings.secret_values(), self.canary, PROMPT_FINGERPRINTS
        )
        if blocked:
            logger.warning("answer withheld reason=%s", blocked)
            return self._finish(started, _refusal("output_blocked", OUTPUT_BLOCKED_MESSAGE, chunks, llm_called=True))

        grounded = (
            output.answerable
            and chunks[0].score >= self.settings.answer_threshold
            and bool(check.cited_ids)
            and bool(check.answer)
        )
        if grounded:
            by_id = {chunk.citation_id: chunk for chunk in chunks}
            result = AskResult(
                answerable=True,
                answer=check.answer,
                general_knowledge=check.general_knowledge,
                refusal_reason=None,
                citations=[by_id[cid] for cid in check.cited_ids],
                retrieved=chunks,
                cited_ids=set(check.cited_ids),
            )
        else:
            result = _refusal("not_in_documents", NOT_IN_DOCUMENTS_MESSAGE, chunks)
            result.general_knowledge = check.general_knowledge
        result.llm_called = True
        result.raw_citation_count = check.raw_cited
        result.invalid_citation_count = len(check.invalid_ids)
        return self._finish(started, result)

    @staticmethod
    def _finish(started: float, result: AskResult) -> AskResult:
        result.latency_ms = round((time.perf_counter() - started) * 1000, 1)
        logger.info("question answered outcome=%s", result.refusal_reason or "answered")
        return result


def _refusal(reason: str, message: str, chunks: list[RetrievedChunk], llm_called: bool = False) -> AskResult:
    return AskResult(
        answerable=False,
        answer=message,
        general_knowledge=None,
        refusal_reason=reason,
        citations=[],
        retrieved=chunks,
        llm_called=llm_called,
    )
