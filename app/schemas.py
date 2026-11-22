"""Pydantic request and response models.

These models define the API contract and feed FastAPI's OpenAPI docs at
``/docs``. Each model carries an example so Swagger shows realistic payloads.
"""

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints

from app.config import MAX_DOCUMENT_FILTER, MAX_QUESTION_CHARS, MAX_TOP_K


class ErrorBody(BaseModel):
    """The inner object of every error response."""

    code: str = Field(description="Stable, machine-readable error code.")
    message: str = Field(description="Human-readable explanation.")
    request_id: str = Field(description="ID to quote when reporting a problem; matches the X-Request-ID header.")
    details: dict[str, Any] = Field(default_factory=dict, description="Extra context, such as an existing document ID.")


class ErrorResponse(BaseModel):
    """Shape shared by all error responses (4xx and 5xx)."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "error": {
                    "code": "duplicate_document",
                    "message": "This file was already uploaded.",
                    "request_id": "3f2b9c1e8a7d4e6f9b0c1d2e3f4a5b6c",
                    "details": {"document_id": "d7c1a2b3"},
                }
            }
        }
    )

    error: ErrorBody


class HealthResponse(BaseModel):
    """Service status. Reports whether keys are configured, never their values."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "status": "ok",
                "app_name": "Production RAG Service",
                "version": "0.1.0",
                "environment": "development",
                "llm_configured": True,
                "embeddings_configured": True,
            }
        }
    )

    status: str = Field(description="'ok' when the API process is running.")
    app_name: str
    version: str
    environment: str
    llm_configured: bool = Field(description="Groq key and answer model are set.")
    embeddings_configured: bool = Field(description="Gemini key and embedding model are set.")


DOCUMENT_EXAMPLE = {
    "document_id": "7576edb531d98488",
    "filename": "NIST.AI.100-1.pdf",
    "file_type": "pdf",
    "size_bytes": 1946127,
    "page_count": 48,
    "section_count": 48,
    "chunk_count": 152,
    "chunk_config": "A",
    "created_at": "2026-11-22T12:00:44.483341+00:00",
}


class DocumentResponse(BaseModel):
    """Safe metadata returned for an ingested document."""

    model_config = ConfigDict(json_schema_extra={"example": DOCUMENT_EXAMPLE})

    document_id: str = Field(description="First 16 hex characters of the file's SHA-256 hash.")
    filename: str = Field(description="Sanitised file name.")
    file_type: str = Field(description="'pdf' or 'docx'.")
    size_bytes: int
    page_count: int | None = Field(description="Pages in the PDF (or page labels in a DOCX).")
    section_count: int = Field(description="Citable sections (pages or headings) with text.")
    chunk_count: int = Field(description="Chunks stored in the vector database.")
    chunk_config: str = Field(description="Chunking configuration used: 'A' (1000/150) or 'B' (500/75).")
    created_at: str = Field(description="Upload time, ISO 8601 UTC.")


class DocumentListResponse(BaseModel):
    model_config = ConfigDict(json_schema_extra={"example": {"documents": [DOCUMENT_EXAMPLE], "total": 1}})

    documents: list[DocumentResponse]
    total: int


# --- Questions and search ---------------------------------------------------

# Whitespace is stripped first, so "   " is rejected as empty (422).
QuestionText = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=MAX_QUESTION_CHARS)]
DocumentId = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{16}$")]


class AskRequest(BaseModel):
    """A question about the uploaded documents."""

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "examples": [
                {"question": "What are the four functions of the AI RMF Core?"},
                {"question": "What is prompt injection?", "top_k": 5, "document_ids": ["6e711bdf3b6053db"]},
            ]
        },
    )

    question: QuestionText = Field(description=f"1 to {MAX_QUESTION_CHARS} characters after trimming.")
    top_k: int | None = Field(default=None, ge=1, le=MAX_TOP_K, description="Chunks to retrieve (default 5).")
    document_ids: list[DocumentId] | None = Field(
        default=None, max_length=MAX_DOCUMENT_FILTER, description="Only search these documents."
    )


class SearchRequest(BaseModel):
    """Retrieval only: returns the most similar chunks without calling the LLM."""

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={"examples": [{"query": "measure function of the AI RMF", "top_k": 3}]},
    )

    query: QuestionText
    top_k: int | None = Field(default=None, ge=1, le=MAX_TOP_K)
    document_ids: list[DocumentId] | None = Field(default=None, max_length=MAX_DOCUMENT_FILTER)


class CitationOut(BaseModel):
    """A source the answer cites. ``citation_id`` matches the [C#] marker in the answer."""

    citation_id: str
    chunk_id: str
    document_id: str
    filename: str
    page: int | None
    section: str | None
    chunk_index: int


class RetrievedChunkOut(CitationOut):
    """A retrieved chunk with its similarity score (cosine, 0 to 1)."""

    score: float
    text: str
    cited: bool = Field(description="True if the answer cites this chunk.")


class AskResponse(BaseModel):
    """The answer, its citations and every retrieved chunk."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "answerable": True,
                "answer": "The AI RMF Core is organized around four high-level functions: GOVERN, MAP, MEASURE, and MANAGE [C1].",
                "general_knowledge": None,
                "refusal_reason": None,
                "citations": [
                    {
                        "citation_id": "C1",
                        "chunk_id": "7576edb531d98488:76",
                        "document_id": "7576edb531d98488",
                        "filename": "NIST.AI.100-1.pdf",
                        "page": 25,
                        "section": "AI RMF Core",
                        "chunk_index": 76,
                    }
                ],
                "retrieved_count": 1,
                "retrieved_chunks": [
                    {
                        "citation_id": "C1",
                        "chunk_id": "7576edb531d98488:76",
                        "document_id": "7576edb531d98488",
                        "filename": "NIST.AI.100-1.pdf",
                        "page": 25,
                        "section": "AI RMF Core",
                        "chunk_index": 76,
                        "score": 0.7888,
                        "text": "Part 2: Core and Profiles\n5. AI RMF Core\nThe AI RMF Core provides outcomes and actions ...",
                        "cited": True,
                    }
                ],
                "latency_ms": 1060.7,
                "request_id": "3f2b9c1e8a7d4e6f9b0c1d2e3f4a5b6c",
            }
        }
    )

    answerable: bool = Field(description="True only for an answer grounded in the documents with valid citations.")
    answer: str = Field(description="Answer with [C#] markers, or the refusal message.")
    general_knowledge: str | None = Field(
        description="Background from the model's general knowledge. Not from your documents and never cited."
    )
    refusal_reason: Literal["not_in_documents", "off_topic", "no_documents", "output_blocked"] | None
    citations: list[CitationOut]
    retrieved_count: int
    retrieved_chunks: list[RetrievedChunkOut]
    latency_ms: float
    request_id: str


class SearchResponse(BaseModel):
    results: list[RetrievedChunkOut]
    retrieved_count: int
    latency_ms: float
