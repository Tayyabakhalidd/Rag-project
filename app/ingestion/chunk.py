"""Chunking: split each Section into overlapping passages small enough to embed.

We use LangChain's ``RecursiveCharacterTextSplitter`` (only this splitter; the
rest of the pipeline is our own code). It tries to cut at the most natural
boundary first: blank line, then line break, then sentence end, then word,
and only splits inside a word as a last resort.

Chunks are made **within** one Section (one PDF page or one DOCX section), never
across two, so every chunk has exactly one page/section to cite. The trade-off
is that a passage running over a page break is split at the break.

Two configurations are compared in the evaluation (see ``CHUNK_CONFIGS``):
A = 1000 characters with 150 overlap, B = 500 characters with 75 overlap.

How overlap is made
-------------------
The splitter's built-in overlap can only repeat *whole pieces*. Our text is
made of ~90-character lines, so on the real corpus config B's "75 overlap"
produced no overlap at all for 55-70% of neighbouring chunks (measured; see
DESIGN.md). To get the overlap we actually asked for:

1. the splitter picks the cut points, aiming at ``chunk_size - chunk_overlap``
   characters with no overlap of its own;
2. each chunk (except the first in a section) is then extended backwards by up
   to ``chunk_overlap`` characters, starting at a word boundary.

So a chunk is never longer than ``chunk_size``, neighbours share close to
``chunk_overlap`` characters, and every chunk is still an exact slice of its
section's text (``char_start``/``char_end`` point to it).
"""

import re

from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import ChunkConfig
from app.ingestion.models import Chunk, ExtractedDocument, ValidatedFile

# Natural boundaries, from strongest to weakest.
SEPARATORS = ["\n\n", "\n", ". ", "; ", ", ", " ", ""]


def make_splitter(config: ChunkConfig) -> RecursiveCharacterTextSplitter:
    """Build the splitter that chooses cut points (overlap is added afterwards)."""
    return RecursiveCharacterTextSplitter(
        chunk_size=config.chunk_size - config.chunk_overlap,
        chunk_overlap=0,
        separators=SEPARATORS,
        keep_separator="end",  # a sentence keeps its full stop instead of the next chunk starting with ". "
        strip_whitespace=True,
        add_start_index=True,  # records where each piece starts inside the section text
        length_function=len,  # sizes are in characters
    )


def _visible_chars(text: str) -> int:
    return len(re.sub(r"\s", "", text))


def overlap_start(text: str, start: int, overlap: int) -> int:
    """Move ``start`` back by up to ``overlap`` characters, landing on a word start.

    The new start is never inside a word: if ``start - overlap`` falls mid-word,
    we move forward to the next word. If no word boundary exists in that window,
    the chunk simply gets no overlap.
    """
    if overlap <= 0 or start <= 0:
        return start
    candidate = max(0, start - overlap)
    if candidate > 0 and not text[candidate - 1].isspace():
        match = re.search(r"\s", text[candidate:start])
        if match is None:
            return start
        candidate += match.end()
    while candidate < start and text[candidate].isspace():
        candidate += 1
    return candidate


def chunk_document(
    validated: ValidatedFile,
    extracted: ExtractedDocument,
    config: ChunkConfig,
    min_chunk_chars: int,
) -> list[Chunk]:
    """Split every section of a document and attach citation metadata to each chunk."""
    splitter = make_splitter(config)
    chunks: list[Chunk] = []
    for section in extracted.sections:
        for position, piece in enumerate(splitter.create_documents([section.text])):
            if _visible_chars(piece.page_content) < min_chunk_chars:
                continue  # stray fragments such as a lone figure label
            piece_start = int(piece.metadata["start_index"])
            end = piece_start + len(piece.page_content)
            # The first piece of a section gets no overlap: sections never share text.
            start = piece_start if position == 0 else overlap_start(section.text, piece_start, config.chunk_overlap)
            text = section.text[start:end]
            index = len(chunks)
            chunks.append(
                Chunk(
                    chunk_id=f"{validated.document_id}:{index}",
                    document_id=validated.document_id,
                    filename=validated.filename,
                    file_type=validated.file_type,
                    chunk_index=index,
                    text=text,
                    page=section.page,
                    section=section.section,
                    section_order=section.order,
                    char_start=start,
                    char_end=start + len(text),
                    sha256=validated.sha256,
                    chunk_config=config.name,
                )
            )
    return chunks
