"""Upload validation: decide whether a file is safe and supported *before* parsing it.

Checks, in order:

1. The file name is sanitised (no folders, no control characters).
2. The extension is supported (.pdf or .docx)                -> 415 otherwise
3. The file is not empty                                     -> 400
4. The file is not larger than ``MAX_UPLOAD_MB``             -> 413
5. The bytes really are that type (file signature)           -> 415
6. DOCX only: the ZIP is a real Word document and not a
   "ZIP bomb" that expands to a huge size                     -> 415 / 400

The extension alone is never trusted: a text file renamed to ``.pdf`` fails
step 5.
"""

import hashlib
import io
import re
import unicodedata
import zipfile
from pathlib import PurePosixPath, PureWindowsPath
from typing import Protocol

from app.config import SUPPORTED_EXTENSIONS, Settings
from app.errors import EmptyDocumentError, FileTooLargeError, InvalidDocumentError, UnsupportedFileTypeError
from app.ingestion.models import ValidatedFile

PDF_SIGNATURE = b"%PDF-"
ZIP_SIGNATURE = b"PK\x03\x04"
OLE_SIGNATURE = b"\xd0\xcf\x11\xe0"  # legacy .doc and password-protected Office files
DOCX_REQUIRED_PARTS = ("[Content_Types].xml", "word/document.xml")
MAX_FILENAME_CHARS = 200
READ_CHUNK_BYTES = 1024 * 1024


class AsyncReadable(Protocol):
    """Anything with an async ``read(size)``, such as FastAPI's ``UploadFile``."""

    async def read(self, size: int = -1) -> bytes: ...


async def read_upload_limited(upload: AsyncReadable, max_bytes: int) -> bytes:
    """Read an upload in 1 MB pieces, stopping as soon as it exceeds ``max_bytes``.

    This avoids loading a huge file fully into memory just to reject it.
    """
    buffer = bytearray()
    while piece := await upload.read(READ_CHUNK_BYTES):
        buffer.extend(piece)
        if len(buffer) > max_bytes:
            raise FileTooLargeError(details={"max_bytes": max_bytes})
    return bytes(buffer)


def sanitize_filename(raw_name: str | None) -> str:
    """Keep only the base name, drop control characters and limit the length."""
    name = raw_name or ""
    # Strip any folder part, whether written with / or \ (e.g. "../../evil.pdf").
    name = PureWindowsPath(PurePosixPath(name).name).name
    name = "".join(ch for ch in name if unicodedata.category(ch)[0] != "C")
    name = re.sub(r"\s+", " ", name).strip(" .")
    if len(name) > MAX_FILENAME_CHARS:
        stem, dot, ext = name.rpartition(".")
        name = (stem[: MAX_FILENAME_CHARS - len(ext) - 1] + dot + ext) if dot else name[:MAX_FILENAME_CHARS]
    return name or "unnamed"


def detect_file_type(filename: str) -> str:
    """Map the extension to a supported type, or raise 415."""
    extension = PurePosixPath(filename.lower()).suffix
    file_type = SUPPORTED_EXTENSIONS.get(extension)
    if file_type is None:
        raise UnsupportedFileTypeError(
            details={"extension": extension or "(none)", "supported": sorted(SUPPORTED_EXTENSIONS)}
        )
    return file_type


def _check_pdf_signature(data: bytes) -> None:
    # The PDF header must appear near the start of the file.
    if PDF_SIGNATURE not in data[:1024]:
        raise UnsupportedFileTypeError("The file has a .pdf extension but is not a PDF.")


def _check_docx_container(data: bytes, settings: Settings) -> None:
    if data.startswith(OLE_SIGNATURE):
        raise UnsupportedFileTypeError(
            "This looks like a legacy .doc or password-protected Word file. Save it as a normal .docx and retry."
        )
    if not data.startswith(ZIP_SIGNATURE):
        raise UnsupportedFileTypeError("The file has a .docx extension but is not a Word document.")
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            entries = archive.infolist()
            names = {entry.filename for entry in entries}
            if not all(part in names for part in DOCX_REQUIRED_PARTS):
                raise UnsupportedFileTypeError("The ZIP file is not a Word (.docx) document.")
            if len(entries) > settings.max_docx_zip_entries:
                raise InvalidDocumentError("The DOCX file contains too many internal parts.")
            # ZIP-bomb guard: a small file must not expand to a huge one.
            uncompressed = sum(entry.file_size for entry in entries)
            if uncompressed > settings.max_docx_uncompressed_mb * 1024 * 1024:
                raise InvalidDocumentError(
                    "The DOCX file expands to an unsafe size.",
                    details={"max_uncompressed_mb": settings.max_docx_uncompressed_mb},
                )
    except zipfile.BadZipFile as exc:
        raise InvalidDocumentError("The DOCX file is corrupt.") from exc


def validate_upload(raw_filename: str | None, data: bytes, settings: Settings) -> ValidatedFile:
    """Run every upload check and return a ``ValidatedFile`` or raise an ``AppError``."""
    filename = sanitize_filename(raw_filename)
    file_type = detect_file_type(filename)

    if not data:
        raise EmptyDocumentError("The uploaded file is empty.")
    if len(data) > settings.max_upload_bytes:
        raise FileTooLargeError(details={"max_bytes": settings.max_upload_bytes, "size_bytes": len(data)})

    if file_type == "pdf":
        _check_pdf_signature(data)
    else:
        _check_docx_container(data, settings)

    return ValidatedFile(
        filename=filename,
        file_type=file_type,
        data=data,
        sha256=hashlib.sha256(data).hexdigest(),
    )
