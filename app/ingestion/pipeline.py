"""Ingestion pipeline, part 1: validate -> extract -> chunk.

``prepare_document`` is the single entry point the document service (and the
evaluation script) will call. It is pure CPU work with no network or storage,
so the API layer should run it in a worker thread (``asyncio.to_thread``) to
keep the server responsive. Embedding and storage are added in the next step.
"""

from app.config import ChunkConfig, Settings
from app.errors import EmptyDocumentError
from app.ingestion.chunk import chunk_document
from app.ingestion.extract import NO_TEXT_MESSAGE, extract_text
from app.ingestion.models import PreparedDocument
from app.ingestion.validation import validate_upload


def prepare_document(
    filename: str | None,
    data: bytes,
    settings: Settings,
    chunk_config: ChunkConfig | None = None,
) -> PreparedDocument:
    """Validate, extract and chunk one uploaded file.

    Raises an ``AppError`` subclass (400/413/415) when the file is rejected.
    ``chunk_config`` defaults to the configuration selected in settings.
    """
    config = chunk_config or settings.active_chunk_config
    validated = validate_upload(filename, data, settings)
    extracted = extract_text(validated, settings)
    chunks = chunk_document(validated, extracted, config, settings.min_chunk_chars)
    if not chunks:
        raise EmptyDocumentError(NO_TEXT_MESSAGE)

    return PreparedDocument(
        document_id=validated.document_id,
        filename=validated.filename,
        file_type=validated.file_type,
        sha256=validated.sha256,
        size_bytes=validated.size_bytes,
        page_count=extracted.page_count,
        section_count=len(extracted.sections),
        empty_pages=extracted.empty_pages,
        chunk_config=config.name,
        chunks=chunks,
    )
