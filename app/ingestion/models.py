"""Plain data objects passed between the ingestion steps.

These are internal pipeline types (dataclasses), separate from the API models
in ``app/schemas.py``. Each step takes the previous step's output:

    ValidatedFile -> ExtractedDocument (sections) -> list[Chunk]
"""

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class ValidatedFile:
    """An upload that passed every safety check and is safe to parse."""

    filename: str  # sanitised display name, e.g. "NIST.AI.100-1.pdf"
    file_type: str  # "pdf" or "docx"
    data: bytes
    sha256: str  # hash of the raw bytes, used for duplicate detection

    @property
    def size_bytes(self) -> int:
        return len(self.data)

    @property
    def document_id(self) -> str:
        """Stable ID derived from the content hash (same bytes -> same ID)."""
        return self.sha256[:16]


@dataclass(frozen=True)
class Section:
    """A citable unit of text: one PDF page, or one DOCX heading/page section."""

    text: str
    order: int  # position of the section within the document, from 0
    page: int | None = None  # 1-based page number in the source PDF
    section: str | None = None  # heading path, e.g. "Part 1 > Audience"


@dataclass
class ExtractedDocument:
    """All text extracted from one file, split into citable sections."""

    file_type: str
    sections: list[Section]
    page_count: int | None = None  # PDFs, and DOCX files that carry page labels
    empty_pages: list[int] = field(default_factory=list)  # pages with no extractable text
    removed_boilerplate: list[str] = field(default_factory=list)  # repeated header/footer lines dropped

    @property
    def char_count(self) -> int:
        return sum(len(s.text) for s in self.sections)


@dataclass(frozen=True)
class Chunk:
    """One retrievable passage plus the metadata needed to cite it."""

    chunk_id: str  # "{document_id}:{chunk_index}"
    document_id: str
    filename: str
    file_type: str
    chunk_index: int  # position of the chunk within the document, from 0
    text: str
    page: int | None
    section: str | None
    section_order: int  # which Section the chunk came from
    char_start: int  # offsets of ``text`` inside its Section's text
    char_end: int
    sha256: str
    chunk_config: str  # "A" or "B"

    def metadata(self) -> dict[str, Any]:
        """Metadata for the vector store (everything except the text)."""
        data = asdict(self)
        data.pop("text")
        return data


@dataclass
class PreparedDocument:
    """Result of validate -> extract -> chunk, ready to embed and store."""

    document_id: str
    filename: str
    file_type: str
    sha256: str
    size_bytes: int
    page_count: int | None
    section_count: int
    empty_pages: list[int]
    chunk_config: str
    chunks: list[Chunk]
