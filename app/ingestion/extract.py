"""Text extraction: turn a validated PDF or DOCX file into citable sections.

PDF  -> one Section per page, numbered from 1. If the PDF has bookmarks, each
        page also gets the heading path of the bookmark it falls under.
DOCX -> a new Section starts at every heading. Two kinds of heading are handled:
        * page labels written as headings ("Page 12"), which the corpus DOCX
          files carry from their PDF conversion: they set the page number;
        * ordinary headings ("Heading 1/2/3"): they build the section path,
          e.g. "Risks > Prompt injection".
        Tables become one "cell | cell | cell" line per row.

Both formats then get the same clean-up:
* characters normalised (ligatures such as "ﬁ" become "fi"), whitespace
  normalised, and words split by a line-end hyphen re-joined;
* header and footer lines that repeat on most pages (for example
  "NIST AI 100-1 AI RMF 1.0" or "Page 7") removed, so they don't pollute
  every chunk.

Scanned pages have no text layer; they are reported in ``empty_pages``. OCR is
out of scope. A document with no text at all is rejected with 400.
"""

import io
import logging
import re
import unicodedata
from collections import Counter
from collections.abc import Iterable

import docx
from docx.table import Table
from docx.text.paragraph import Paragraph
from pypdf import PdfReader

from app.config import Settings
from app.errors import AppError, EmptyDocumentError, FileTooLargeError, InvalidDocumentError
from app.ingestion.models import ExtractedDocument, Section, ValidatedFile

logger = logging.getLogger("app.ingestion")

EDGE_LINES = 2  # how many lines at the top and bottom of a page count as header/footer
PAGE_LABEL = re.compile(r"^\s*page\s+(\d{1,5})\s*$", re.IGNORECASE)
HEADING_STYLE = re.compile(r"^heading\s+(\d)$", re.IGNORECASE)
PAGE_NUMBER_KEYS = {"#", "page #", "- # -", "# of #", "page # of #"}
NO_TEXT_MESSAGE = "No extractable text was found. Scanned documents need OCR, which this service does not perform."


# --------------------------------------------------------------------------
# Shared text clean-up
# --------------------------------------------------------------------------

def normalize_lines(text: str) -> list[str]:
    """Normalise characters and whitespace; return the non-empty lines of ``text``.

    NFKC turns typographic ligatures into plain letters ("ﬁ" -> "fi"), so a
    search for "confidentiality" matches text extracted as "conﬁdentiality".
    """
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\x00", "").replace("\r\n", "\n").replace("\r", "\n").replace(" ", " ")
    lines = (re.sub(r"[ \t\f\v]+", " ", line).strip() for line in text.split("\n"))
    return [line for line in lines if line]


def join_lines(lines: list[str]) -> str:
    """Join lines and re-attach words split by a hyphen at a line end ("mea-\\nsure")."""
    text = "\n".join(lines)
    return re.sub(r"(\w)-\n([a-z])", r"\1\2", text).strip()


def _line_key(line: str) -> str:
    """Compare lines ignoring case and digits, so "Page 7" matches "Page 8"."""
    return re.sub(r"\d+", "#", line.strip().lower())


def remove_repeated_lines(pages: list[list[str]]) -> tuple[list[list[str]], list[str]]:
    """Drop header/footer lines that repeat at the top or bottom of most pages.

    A line is treated as boilerplate when (ignoring digits) it appears among the
    first or last ``EDGE_LINES`` lines of at least half of the pages, or when it
    is a bare page number. Only lines in those edge positions are removed, so a
    sentence in the body is never touched.

    Returns the cleaned pages and one example of each removed line.
    """
    pages_with_text = [p for p in pages if p]
    if len(pages_with_text) < 4:
        return pages, []

    edge_counts: Counter[str] = Counter()
    for lines in pages_with_text:
        edge = lines[:EDGE_LINES] + lines[-EDGE_LINES:]
        edge_counts.update({_line_key(line) for line in edge})
    threshold = max(3, len(pages_with_text) // 2)
    boilerplate = {key for key, count in edge_counts.items() if count >= threshold} | PAGE_NUMBER_KEYS

    cleaned: list[list[str]] = []
    examples: dict[str, str] = {}
    for lines in pages:
        last = len(lines) - EDGE_LINES
        kept = []
        for i, line in enumerate(lines):
            key = _line_key(line)
            if (i < EDGE_LINES or i >= last) and key in boilerplate:
                examples.setdefault(key, line)
                continue
            kept.append(line)
        cleaned.append(kept)
    return cleaned, sorted(examples.values())


# --------------------------------------------------------------------------
# PDF
# --------------------------------------------------------------------------

LONG_TITLE_PREFIX = 30  # long headings may wrap onto two lines, so compare a prefix
TOC_LINE = re.compile(r"^(.*?)[\s.]*\d+$")  # "1.2 Risk Tolerance ........ 7"


def _heading_key(text: str) -> str:
    """Comparable form of a heading: NFKC, lower case, single spaces, no leading "1.2.3"."""
    text = re.sub(r"\s+", " ", unicodedata.normalize("NFKC", text)).strip().lower()
    return re.sub(r"^[\d.]+\s+", "", text)


def _is_heading_line(line_key: str, title_key: str) -> bool:
    """True if a page line *is* the heading (not just a sentence starting with the word)."""
    if line_key == title_key:
        return True
    # "Appendix A: Descriptions of ..." is often printed with "Appendix A:" on its own line.
    label, colon, _ = title_key.partition(":")
    if colon and line_key == label + ":":
        return True
    return len(title_key) > LONG_TITLE_PREFIX and line_key.startswith(title_key[:LONG_TITLE_PREFIX])


def _verify_bookmark_pages(
    starts: list[tuple[int, int, str]], pages: list[list[str]]
) -> list[tuple[int, int, str]]:
    """Keep only bookmarks whose heading really appears where they start.

    Bookmark targets in real PDFs are often imprecise (in NIST AI 100-1 three
    appendices all point to page 38). For each bookmark we look for its heading
    line on the target page, then on later pages, skipping table-of-contents
    pages. A bookmark that cannot be confirmed is dropped together with its
    children, so a child is never attached to the wrong parent.
    """
    title_keys = [_heading_key(title) for _, _, title in starts]
    line_keys = [[_heading_key(line) for line in lines] for lines in pages]

    def is_toc_page(page_index: int) -> bool:
        listed = 0
        for line_key in line_keys[page_index]:
            match = TOC_LINE.match(line_key)
            if match and any(_is_heading_line(match.group(1), t) for t in title_keys):
                listed += 1
        return listed >= 5

    toc_pages = {p for p in range(len(pages)) if is_toc_page(p)}
    verified: list[tuple[int, int, str]] = []
    dropped_depth: int | None = None
    for (page_index, depth, title), title_key in zip(starts, title_keys, strict=True):
        if dropped_depth is not None and depth > dropped_depth:
            continue  # child of a dropped bookmark
        dropped_depth = None
        found = next(
            (
                p
                for p in range(page_index, len(pages))
                if p not in toc_pages and any(_is_heading_line(k, title_key) for k in line_keys[p])
            ),
            None,
        )
        if found is None:
            dropped_depth = depth
        else:
            verified.append((found, depth, title))
    return verified


def _pdf_outline_sections(reader: PdfReader, pages: list[list[str]]) -> list[str | None]:
    """Map each page index to the (verified) bookmark heading path it falls under.

    Best effort: a PDF without bookmarks (or with broken ones) gets ``None``
    for every page. A page where a new bookmark starts is assigned to that new
    bookmark, even if the top of the page still belongs to the previous one.
    """
    page_count = len(pages)
    starts: list[tuple[int, int, str]] = []  # (page index, depth, title)

    def walk(items: Iterable, depth: int) -> None:
        for item in items:
            if isinstance(item, list):
                walk(item, depth + 1)
                continue
            try:
                page_index = reader.get_destination_page_number(item)
                title = str(item.title).strip()
            except Exception:  # noqa: BLE001 - malformed bookmark: skip it
                continue
            if page_index is not None and 0 <= page_index < page_count and title:
                starts.append((page_index, depth, title))

    try:
        walk(reader.outline, 0)
    except Exception:  # noqa: BLE001 - no usable outline
        return [None] * page_count

    starts = _verify_bookmark_pages(starts, pages)
    starts.sort(key=lambda entry: entry[0])  # stable: keeps document order within a page
    sections: list[str | None] = []
    stack: list[str] = []
    next_entry = 0
    for page_index in range(page_count):
        while next_entry < len(starts) and starts[next_entry][0] <= page_index:
            _, depth, title = starts[next_entry]
            stack = stack[:depth] + [title]
            next_entry += 1
        sections.append(" > ".join(stack) if stack else None)
    return sections


def extract_pdf(data: bytes, settings: Settings) -> ExtractedDocument:
    """Extract one Section per PDF page."""
    # pypdf is parsing untrusted bytes, so any failure means "corrupt file".
    try:
        reader = PdfReader(io.BytesIO(data))
        if reader.is_encrypted and not reader.decrypt(""):
            raise InvalidDocumentError("The PDF is password-protected.")
        page_count = len(reader.pages)
    except AppError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise InvalidDocumentError("The PDF is corrupt or cannot be read.") from exc

    if page_count > settings.max_pdf_pages:
        raise FileTooLargeError(
            f"The PDF has {page_count} pages; the limit is {settings.max_pdf_pages}.",
            details={"page_count": page_count, "max_pages": settings.max_pdf_pages},
        )

    raw_pages: list[list[str]] = []
    for index, page in enumerate(reader.pages):
        try:
            raw_pages.append(normalize_lines(page.extract_text() or ""))
        except Exception:  # noqa: BLE001 - one unreadable page should not fail the file
            logger.warning("Could not read text on PDF page %d", index + 1)
            raw_pages.append([])

    pages, removed = remove_repeated_lines(raw_pages)
    outline = _pdf_outline_sections(reader, pages)

    sections: list[Section] = []
    empty_pages: list[int] = []
    for index, lines in enumerate(pages):
        text = join_lines(lines)
        if not text:
            empty_pages.append(index + 1)
            continue
        sections.append(Section(text=text, order=len(sections), page=index + 1, section=outline[index]))

    if not sections:
        raise EmptyDocumentError(NO_TEXT_MESSAGE, details={"page_count": page_count})
    return ExtractedDocument(
        file_type="pdf",
        sections=sections,
        page_count=page_count,
        empty_pages=empty_pages,
        removed_boilerplate=removed,
    )


# --------------------------------------------------------------------------
# DOCX
# --------------------------------------------------------------------------

def _table_lines(table: Table) -> list[str]:
    """One "cell | cell" line per row; merged cells are listed once."""
    lines = []
    for row in table.rows:
        cells: list[str] = []
        for cell in row.cells:
            text = " ".join(normalize_lines(cell.text))
            if text and (not cells or cells[-1] != text):
                cells.append(text)
        if cells:
            lines.append(" | ".join(cells))
    return lines


def extract_docx(data: bytes, settings: Settings) -> ExtractedDocument:
    """Extract Sections from a DOCX file, split at headings and page labels."""
    try:
        document = docx.Document(io.BytesIO(data))
        blocks = list(document.iter_inner_content())  # paragraphs and tables, in order
        # python-docx's ``paragraph.style`` re-scans the style table on every call
        # (about 15 s for the 2,400-paragraph corpus file), so map style IDs to
        # names once and read each paragraph's raw style ID instead.
        style_names = {style.style_id: style.name for style in document.styles}
    except Exception as exc:  # noqa: BLE001 - untrusted file
        raise InvalidDocumentError("The DOCX file is corrupt or cannot be read.") from exc

    # Each raw section is (page, heading path, lines).
    raw: list[tuple[int | None, str | None, list[str]]] = []
    current_page: int | None = None
    heading_stack: list[tuple[int, str]] = []
    buffer: list[str] = []
    has_body = False  # a heading with no text under it becomes part of the path only
    labelled_pages: set[int] = set()

    def flush() -> None:
        nonlocal has_body
        if buffer and has_body:
            path = " > ".join(title for _, title in heading_stack) or None
            raw.append((current_page, path, buffer.copy()))
        buffer.clear()
        has_body = False

    for block in blocks:
        if isinstance(block, Table):
            rows = _table_lines(block)
            buffer.extend(rows)
            has_body = has_body or bool(rows)
            continue
        if not isinstance(block, Paragraph):
            continue
        text = " ".join(normalize_lines(block.text))
        if not text:
            continue
        style_id = block._p.style  # raw w:pStyle value, or None for the default style
        style_name = style_names.get(style_id, "") if style_id else ""
        heading = HEADING_STYLE.match(style_name)
        page_label = PAGE_LABEL.match(text)

        if heading and page_label:
            # A page label from the PDF conversion: it marks a page, not a topic.
            flush()
            current_page = int(page_label.group(1))
            labelled_pages.add(current_page)
        elif heading:
            flush()
            level = int(heading.group(1))
            while heading_stack and heading_stack[-1][0] >= level:
                heading_stack.pop()
            heading_stack.append((level, text))
            buffer.append(text)  # keep the heading as context in the chunk text
        else:
            buffer.append(text)
            has_body = True
    flush()

    cleaned, removed = remove_repeated_lines([lines for _, _, lines in raw])
    sections: list[Section] = []
    pages_with_text: set[int] = set()
    for (page, path, _), lines in zip(raw, cleaned, strict=True):
        text = join_lines(lines)
        if not text:
            continue
        if page is not None:
            pages_with_text.add(page)
        sections.append(Section(text=text, order=len(sections), page=page, section=path))

    if not sections:
        raise EmptyDocumentError(NO_TEXT_MESSAGE)
    page_count = max(labelled_pages) if labelled_pages else None
    return ExtractedDocument(
        file_type="docx",
        sections=sections,
        page_count=page_count,
        empty_pages=sorted(labelled_pages - pages_with_text),
        removed_boilerplate=removed,
    )


def extract_text(validated: ValidatedFile, settings: Settings) -> ExtractedDocument:
    """Dispatch to the extractor for the file's type."""
    if validated.file_type == "pdf":
        return extract_pdf(validated.data, settings)
    return extract_docx(validated.data, settings)
