"""Build small test documents in memory, so tests need no binary fixture files.

* ``make_pdf`` writes a minimal valid PDF by hand (one text line per ``Tj``).
* ``add_bookmarks`` / ``encrypt_pdf`` post-process it with pypdf.
* ``make_docx`` builds a Word file with python-docx from a simple block list.
"""

import io
import zipfile

import docx
from pypdf import PdfWriter


def _pdf_escape(text: str) -> str:
    return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def make_pdf(pages: list[list[str]]) -> bytes:
    """Return PDF bytes with one page per item; each page lists its text lines."""
    count = len(pages)
    page_ids = [4 + 2 * i for i in range(count)]
    content_ids = [5 + 2 * i for i in range(count)]
    objects: dict[int, bytes] = {
        1: b"<< /Type /Catalog /Pages 2 0 R >>",
        2: f"<< /Type /Pages /Kids [{' '.join(f'{p} 0 R' for p in page_ids)}] /Count {count} >>".encode(),
        3: b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /Encoding /WinAnsiEncoding >>",
    }
    for i, lines in enumerate(pages):
        ops = ["BT", "/F1 11 Tf", "14 TL", "72 740 Td"]
        for line in lines:
            ops += [f"({_pdf_escape(line)}) Tj", "T*"]
        ops.append("ET")
        stream = "\n".join(ops).encode("cp1252")
        objects[content_ids[i]] = b"<< /Length %d >>\nstream\n" % len(stream) + stream + b"\nendstream"
        objects[page_ids[i]] = (
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            f"/Resources << /Font << /F1 3 0 R >> >> /Contents {content_ids[i]} 0 R >>"
        ).encode()

    out = bytearray(b"%PDF-1.4\n")
    offsets: dict[int, int] = {}
    for number in sorted(objects):
        offsets[number] = len(out)
        out += f"{number} 0 obj\n".encode() + objects[number] + b"\nendobj\n"
    xref_at = len(out)
    size = max(objects) + 1
    out += f"xref\n0 {size}\n0000000000 65535 f \n".encode()
    for number in range(1, size):
        out += f"{offsets[number]:010d} 00000 n \n".encode()
    out += f"trailer\n<< /Size {size} /Root 1 0 R >>\nstartxref\n{xref_at}\n%%EOF\n".encode()
    return bytes(out)


def _rewrite(pdf: bytes, change) -> bytes:
    writer = PdfWriter(clone_from=io.BytesIO(pdf))
    change(writer)
    buffer = io.BytesIO()
    writer.write(buffer)
    return buffer.getvalue()


def add_bookmarks(pdf: bytes, bookmarks: list[tuple[str, int, str | None]]) -> bytes:
    """Add bookmarks as (title, 0-based page index, parent title or None)."""

    def change(writer: PdfWriter) -> None:
        created = {}
        for title, page_index, parent in bookmarks:
            created[title] = writer.add_outline_item(
                title=title, page_number=page_index, parent=created.get(parent)
            )

    return _rewrite(pdf, change)


def encrypt_pdf(pdf: bytes, password: str) -> bytes:
    """Protect a PDF with a user password (RC4, needs no extra crypto library)."""
    return _rewrite(pdf, lambda writer: writer.encrypt(password, algorithm="RC4-128"))


def make_docx(blocks: list[tuple]) -> bytes:
    """Build a DOCX from blocks: ("h", level, text), ("p", text) or ("table", [[cells], ...])."""
    document = docx.Document()
    for block in blocks:
        kind = block[0]
        if kind == "h":
            document.add_heading(block[2], level=block[1])
        elif kind == "p":
            document.add_paragraph(block[1])
        elif kind == "table":
            rows = block[1]
            table = document.add_table(rows=len(rows), cols=len(rows[0]))
            for r, row in enumerate(rows):
                for c, value in enumerate(row):
                    table.cell(r, c).text = value
    buffer = io.BytesIO()
    document.save(buffer)
    return buffer.getvalue()


def make_zip(entries: dict[str, bytes]) -> bytes:
    """Build a ZIP archive (used for fake or malicious DOCX files)."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, content in entries.items():
            archive.writestr(name, content)
    return buffer.getvalue()
