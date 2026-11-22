"""JSON API: ask a question, or search without the LLM. Handlers stay thin."""

import time
from typing import Annotated

from fastapi import APIRouter, Depends, Request

from app.api.deps import get_qa_service, limit_questions
from app.retrieval import RetrievedChunk
from app.schemas import AskRequest, AskResponse, CitationOut, ErrorResponse, RetrievedChunkOut, SearchRequest, SearchResponse
from app.services.qa_service import AskResult, QAService

router = APIRouter(prefix="/api/v1", tags=["questions"])


def citation_out(chunk: RetrievedChunk) -> CitationOut:
    return CitationOut(
        citation_id=chunk.citation_id,
        chunk_id=chunk.chunk_id,
        document_id=chunk.document_id,
        filename=chunk.filename,
        page=chunk.page,
        section=chunk.section,
        chunk_index=chunk.chunk_index,
    )


def chunk_out(chunk: RetrievedChunk, cited: bool) -> RetrievedChunkOut:
    return RetrievedChunkOut(**citation_out(chunk).model_dump(), score=chunk.score, text=chunk.text, cited=cited)


def ask_response(result: AskResult, request_id: str) -> AskResponse:
    return AskResponse(
        answerable=result.answerable,
        answer=result.answer,
        general_knowledge=result.general_knowledge,
        refusal_reason=result.refusal_reason,
        citations=[citation_out(chunk) for chunk in result.citations],
        retrieved_count=len(result.retrieved),
        retrieved_chunks=[chunk_out(chunk, chunk.citation_id in result.cited_ids) for chunk in result.retrieved],
        latency_ms=result.latency_ms,
        request_id=request_id,
    )


@router.post(
    "/ask",
    response_model=AskResponse,
    summary="Ask a question about the uploaded documents",
    description=(
        "Returns a grounded answer with citations, or a refusal (still 200) with `refusal_reason` "
        "`not_in_documents`, `off_topic`, `no_documents` or `output_blocked`. Prompt-injection and "
        "jailbreak attempts return 400 `guardrail_blocked`."
    ),
    responses={
        400: {"model": ErrorResponse, "description": "Blocked by an input guardrail"},
        404: {"model": ErrorResponse, "description": "A document in `document_ids` does not exist"},
        422: {"model": ErrorResponse, "description": "Empty or too-long question, invalid top_k"},
        429: {"model": ErrorResponse, "description": "Rate limit exceeded"},
        503: {"model": ErrorResponse, "description": "Embedding API, LLM or vector database unavailable"},
    },
    dependencies=[Depends(limit_questions)],
)
async def ask(body: AskRequest, request: Request, qa: Annotated[QAService, Depends(get_qa_service)]) -> AskResponse:
    result = await qa.ask(body.question, body.top_k, body.document_ids)
    return ask_response(result, request.state.request_id)


@router.post(
    "/search",
    response_model=SearchResponse,
    summary="Retrieve the most similar chunks (no LLM)",
    description="Embeds the query and returns the top-k chunks with cosine similarity scores.",
    responses={
        404: {"model": ErrorResponse},
        422: {"model": ErrorResponse},
        429: {"model": ErrorResponse},
        503: {"model": ErrorResponse},
    },
    dependencies=[Depends(limit_questions)],
)
async def search(body: SearchRequest, qa: Annotated[QAService, Depends(get_qa_service)]) -> SearchResponse:
    started = time.perf_counter()
    chunks = await qa.search(body.query, body.top_k, body.document_ids)
    return SearchResponse(
        results=[chunk_out(chunk, cited=False) for chunk in chunks],
        retrieved_count=len(chunks),
        latency_ms=round((time.perf_counter() - started) * 1000, 1),
    )
