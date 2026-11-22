"""Tests for ingestion part 1: upload validation, text extraction and chunking.

Most tests build tiny documents in memory (tests/factories.py). The last group
runs the real files in corpus/ with both chunk configurations.
"""

import asyncio
import hashlib

import pytest

from app.config import CHUNK_CONFIGS, CORPUS_DIR, ChunkConfig, Settings
from app.errors import (
    EmptyDocumentError,
    FileTooLargeError,
    InvalidDocumentError,
    UnsupportedFileTypeError,
)
from app.ingestion.chunk import chunk_document, overlap_start
from app.ingestion.extract import extract_docx, extract_pdf, extract_text, normalize_lines, remove_repeated_lines
from app.ingestion.pipeline import prepare_document
from app.ingestion.validation import read_upload_limited, sanitize_filename, validate_upload
from tests.factories import add_bookmarks, encrypt_pdf, make_docx, make_pdf, make_zip

LONG_TEXT = " ".join(f"Sentence {i} explains how AI risk is governed, mapped, measured and managed." for i in range(60))
# Short lines, so even config B's 75-character overlap can hold whole lines.
LONG_LINES = [LONG_TEXT[i : i + 40] for i in range(0, len(LONG_TEXT), 40)]


@pytest.fixture
def ingest_settings(tmp_path) -> Settings:
    return Settings(_env_file=None, environment="test", data_dir=tmp_path)


def simple_pdf() -> bytes:
    return make_pdf([["First page about governance."], ["Second page about measurement."]])


def simple_docx() -> bytes:
    return make_docx([("h", 1, "Overview"), ("p", "The overview paragraph has enough words to keep.")])


# ==========================================================================
# Validation
# ==========================================================================

class TestValidation:
    def test_valid_pdf_and_docx_pass(self, ingest_settings):
        for name, data, kind in [("a.pdf", simple_pdf(), "pdf"), ("b.DOCX", simple_docx(), "docx")]:
            validated = validate_upload(name, data, ingest_settings)
            assert validated.file_type == kind
            assert validated.sha256 == hashlib.sha256(data).hexdigest()
            assert validated.document_id == validated.sha256[:16]

    @pytest.mark.parametrize("name", ["virus.exe", "notes.txt", "README", "slides.pptx"])
    def test_unsupported_extension_is_415(self, ingest_settings, name):
        with pytest.raises(UnsupportedFileTypeError) as exc:
            validate_upload(name, b"some bytes", ingest_settings)
        assert exc.value.status_code == 415

    def test_text_file_renamed_to_pdf_is_415(self, ingest_settings):
        with pytest.raises(UnsupportedFileTypeError):
            validate_upload("fake.pdf", b"just plain text, not a pdf", ingest_settings)

    def test_non_zip_docx_is_415(self, ingest_settings):
        with pytest.raises(UnsupportedFileTypeError):
            validate_upload("fake.docx", b"plain text pretending to be docx", ingest_settings)

    def test_zip_that_is_not_word_is_415(self, ingest_settings):
        with pytest.raises(UnsupportedFileTypeError):
            validate_upload("archive.docx", make_zip({"hello.txt": b"hi"}), ingest_settings)

    def test_legacy_or_password_protected_word_file_is_415(self, ingest_settings):
        with pytest.raises(UnsupportedFileTypeError) as exc:
            validate_upload("old.docx", b"\xd0\xcf\x11\xe0" + b"\x00" * 100, ingest_settings)
        assert "password-protected" in exc.value.message

    def test_empty_file_is_400(self, ingest_settings):
        with pytest.raises(EmptyDocumentError) as exc:
            validate_upload("empty.pdf", b"", ingest_settings)
        assert exc.value.status_code == 400

    def test_oversized_file_is_413(self, tmp_path):
        settings = Settings(_env_file=None, data_dir=tmp_path, max_upload_mb=1)
        with pytest.raises(FileTooLargeError) as exc:
            validate_upload("big.pdf", b"%PDF-1.4\n" + b"0" * (1024 * 1024), settings)
        assert exc.value.status_code == 413

    def test_docx_zip_bomb_is_rejected(self, tmp_path):
        settings = Settings(_env_file=None, data_dir=tmp_path, max_docx_uncompressed_mb=1)
        bomb = make_zip(
            {
                "[Content_Types].xml": b"<Types/>",
                "word/document.xml": b"<w:document/>",
                "word/media/huge.bin": b"\x00" * (3 * 1024 * 1024),  # compresses to a few KB
            }
        )
        assert len(bomb) < 100_000
        with pytest.raises(InvalidDocumentError) as exc:
            validate_upload("bomb.docx", bomb, settings)
        assert exc.value.status_code == 400

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("../../etc/passwd.pdf", "passwd.pdf"),
            ("C:\\Users\\me\\report.docx", "report.docx"),
            ("bad\x00name\x07.pdf", "badname.pdf"),
            ("", "unnamed"),
            (None, "unnamed"),
        ],
    )
    def test_sanitize_filename(self, raw, expected):
        assert sanitize_filename(raw) == expected

    def test_long_filename_keeps_extension(self):
        name = sanitize_filename("x" * 500 + ".pdf")
        assert len(name) <= 200 and name.endswith(".pdf")

    def test_read_upload_limited_stops_early_when_too_large(self):
        class FakeUpload:
            def __init__(self, pieces: int) -> None:
                self.remaining = pieces
                self.reads = 0

            async def read(self, size: int = -1) -> bytes:
                self.reads += 1
                if self.remaining == 0:
                    return b""
                self.remaining -= 1
                return b"x" * size

        small = FakeUpload(pieces=2)
        assert len(asyncio.run(read_upload_limited(small, max_bytes=5 * 1024 * 1024))) == 2 * 1024 * 1024

        huge = FakeUpload(pieces=1000)
        with pytest.raises(FileTooLargeError):
            asyncio.run(read_upload_limited(huge, max_bytes=3 * 1024 * 1024))
        assert huge.reads == 4  # stopped right after passing the limit, not after 1000 reads


# ==========================================================================
# PDF extraction
# ==========================================================================

class TestPdfExtraction:
    def test_pages_are_numbered_from_one_and_blank_pages_flagged(self, ingest_settings):
        pdf = make_pdf([["Alpha page text here."], [], ["Gamma page text here."]])
        extracted = extract_pdf(pdf, ingest_settings)
        assert extracted.page_count == 3
        assert [s.page for s in extracted.sections] == [1, 3]
        assert extracted.empty_pages == [2]
        assert "Alpha" in extracted.sections[0].text

    def test_pdf_with_no_text_is_400(self, ingest_settings):
        with pytest.raises(EmptyDocumentError) as exc:
            extract_pdf(make_pdf([[], []]), ingest_settings)
        assert "OCR" in exc.value.message

    def test_corrupt_pdf_is_invalid(self, ingest_settings):
        with pytest.raises(InvalidDocumentError):
            extract_pdf(b"%PDF-1.4\nthis is not really a pdf body", ingest_settings)

    def test_password_protected_pdf_is_invalid(self, ingest_settings):
        locked = encrypt_pdf(simple_pdf(), password="secret")
        with pytest.raises(InvalidDocumentError) as exc:
            extract_pdf(locked, ingest_settings)
        assert "password" in exc.value.message

    def test_too_many_pages_is_413(self, tmp_path):
        settings = Settings(_env_file=None, data_dir=tmp_path, max_pdf_pages=2)
        with pytest.raises(FileTooLargeError):
            extract_pdf(make_pdf([["a page"], ["b page"], ["c page"]]), settings)

    def test_repeated_header_and_footer_are_removed(self, ingest_settings):
        topics = ["governance", "mapping", "measurement", "management", "profiles", "appendix"]
        pages = [
            ["ACME Risk Report 2026", f"This page is about {t}.", "Middle line.", f"More on {t} here.", f"Page {n}"]
            for n, t in enumerate(topics, start=1)
        ]
        extracted = extract_pdf(make_pdf(pages), ingest_settings)
        assert len(extracted.sections) == 6
        for section, topic in zip(extracted.sections, topics, strict=True):
            assert "ACME Risk Report" not in section.text
            assert f"Page {section.page}" not in section.text
            assert f"This page is about {topic}." in section.text
            assert f"More on {topic} here." in section.text
        assert "ACME Risk Report 2026" in extracted.removed_boilerplate

    def test_body_line_matching_header_text_is_kept(self):
        # The header text also appears in the middle of each page: only the top copy goes.
        words = ["alpha", "beta", "gamma", "delta", "epsilon"]
        pages = [["Header", f"intro {w}", "Header", f"detail {w}", f"closing {w}"] for w in words]
        cleaned, removed = remove_repeated_lines(pages)
        assert removed == ["Header"]
        for page, w in zip(cleaned, words, strict=True):
            assert page == [f"intro {w}", "Header", f"detail {w}", f"closing {w}"]

    def test_hyphenated_words_are_joined(self, ingest_settings):
        pdf = make_pdf([["Risk mea-", "surement needs care.", "AI-", "Based stays hyphenated."]])
        text = extract_pdf(pdf, ingest_settings).sections[0].text
        assert "measurement needs care." in text
        assert "AI-\nBased" in text  # only joined when the next word starts lower-case

    def test_ligatures_and_spaces_are_normalised(self):
        # Real PDFs often extract "fi"/"fl" as single ligature characters.
        lines = normalize_lines("con\ufb01dentiality  and\u00a0\ufb02ow\r\n\r\n  next   line ")
        assert lines == ["confidentiality and flow", "next line"]

    def test_bookmarks_give_verified_section_paths(self, ingest_settings):
        pdf = make_pdf(
            [
                ["Cover page of the report"],
                ["1 Introduction", "Intro body text."],
                ["Intro continues here."],
                ["2 Methods", "Methods body text."],
            ]
        )
        pdf = add_bookmarks(
            pdf,
            [
                ("Introduction", 1, None),
                ("Methods", 1, None),  # wrong target page: really on page 4
                ("Ghost chapter", 2, None),  # not in the text at all: dropped
            ],
        )
        sections = {s.page: s.section for s in extract_pdf(pdf, ingest_settings).sections}
        assert sections == {1: None, 2: "Introduction", 3: "Introduction", 4: "Methods"}


# ==========================================================================
# DOCX extraction
# ==========================================================================

class TestDocxExtraction:
    def test_heading_paths_follow_the_heading_levels(self, ingest_settings):
        data = make_docx(
            [
                ("h", 1, "Risks"),
                ("h", 2, "Prompt injection"),
                ("p", "Attackers hide instructions in input."),
                ("h", 2, "Data leakage"),
                ("p", "Models may reveal private data."),
                ("h", 1, "Controls"),
                ("p", "Apply least privilege."),
            ]
        )
        sections = extract_docx(data, ingest_settings).sections
        assert [s.section for s in sections] == ["Risks > Prompt injection", "Risks > Data leakage", "Controls"]
        assert sections[0].text.startswith("Prompt injection")  # heading kept as context
        assert all(s.page is None for s in sections)

    def test_page_labels_set_page_numbers(self, ingest_settings):
        data = make_docx(
            [
                ("p", "Converted document note."),
                ("h", 1, "Page 1"),
                ("p", "Text from the first page."),
                ("h", 1, "Page 2"),
                ("p", "Text from the second page."),
            ]
        )
        extracted = extract_docx(data, ingest_settings)
        assert [(s.page, s.section) for s in extracted.sections] == [(None, None), (1, None), (2, None)]
        assert all("Page 1" not in s.text and "Page 2" not in s.text for s in extracted.sections)
        assert extracted.page_count == 2

    def test_tables_become_pipe_separated_rows(self, ingest_settings):
        data = make_docx([("h", 1, "Matrix"), ("table", [["Risk", "Owner"], ["Bias", "Data team"]])])
        text = extract_docx(data, ingest_settings).sections[0].text
        assert "Risk | Owner" in text
        assert "Bias | Data team" in text

    def test_docx_with_no_text_is_400(self, ingest_settings):
        with pytest.raises(EmptyDocumentError):
            extract_docx(make_docx([]), ingest_settings)

    def test_corrupt_docx_is_invalid(self, ingest_settings):
        broken = make_zip({"[Content_Types].xml": b"not xml", "word/document.xml": b"<<<"})
        with pytest.raises(InvalidDocumentError):
            extract_docx(broken, ingest_settings)


# ==========================================================================
# Chunking
# ==========================================================================

class TestChunking:
    def test_planned_configurations(self):
        assert (CHUNK_CONFIGS["A"].chunk_size, CHUNK_CONFIGS["A"].chunk_overlap) == (1000, 150)
        assert (CHUNK_CONFIGS["B"].chunk_size, CHUNK_CONFIGS["B"].chunk_overlap) == (500, 75)

    @pytest.mark.parametrize("config_name", ["A", "B"])
    def test_chunks_respect_size_overlap_and_offsets(self, ingest_settings, config_name):
        config = CHUNK_CONFIGS[config_name]
        data = make_pdf([LONG_LINES])
        validated = validate_upload("long.pdf", data, ingest_settings)
        extracted = extract_text(validated, ingest_settings)
        section_text = extracted.sections[0].text
        chunks = chunk_document(validated, extracted, config, ingest_settings.min_chunk_chars)

        assert len(chunks) > 3
        for chunk in chunks:
            assert len(chunk.text) <= config.chunk_size
            assert section_text[chunk.char_start : chunk.char_end] == chunk.text
            assert chunk.page == 1
            assert chunk.chunk_config == config_name
        for previous, current in zip(chunks, chunks[1:], strict=False):
            shared = previous.char_end - current.char_start
            # Neighbours share close to the configured overlap (minus at most one word).
            assert config.chunk_overlap - 20 <= shared <= config.chunk_overlap
            assert not section_text[current.char_start - 1].isalnum()  # overlap starts at a word boundary

    def test_overlap_start_lands_on_word_boundaries(self):
        text = "alpha beta gamma delta epsilon"
        assert overlap_start(text, text.index("delta"), 8) == text.index("gamma")
        assert overlap_start(text, text.index("delta"), 0) == text.index("delta")
        assert overlap_start(text, 0, 10) == 0
        assert overlap_start("abcdefghij", 8, 3) == 8  # no word boundary: no overlap

    def test_config_b_makes_more_chunks_than_a(self, ingest_settings):
        data = make_pdf([LONG_LINES])
        a = prepare_document("x.pdf", data, ingest_settings, CHUNK_CONFIGS["A"])
        b = prepare_document("x.pdf", data, ingest_settings, CHUNK_CONFIGS["B"])
        assert len(b.chunks) > len(a.chunks)

    def test_chunks_never_cross_sections_and_carry_metadata(self, ingest_settings):
        data = make_docx(
            [
                ("h", 1, "Page 1"),
                ("p", LONG_TEXT[:1500]),
                ("h", 1, "Page 2"),
                ("p", LONG_TEXT[1500:2600]),
            ]
        )
        doc = prepare_document("pages.docx", data, ingest_settings, CHUNK_CONFIGS["B"])
        assert {c.page for c in doc.chunks} == {1, 2}
        assert [c.chunk_index for c in doc.chunks] == list(range(len(doc.chunks)))
        assert [c.chunk_id for c in doc.chunks] == [f"{doc.document_id}:{i}" for i in range(len(doc.chunks))]
        metadata = doc.chunks[0].metadata()
        assert "text" not in metadata
        for key in ("document_id", "filename", "page", "section", "chunk_index", "char_start", "char_end"):
            assert key in metadata
        assert metadata["filename"] == "pages.docx"

    def test_tiny_fragments_are_dropped(self, ingest_settings):
        data = make_pdf([["Fig. 2"], ["A real page with a proper sentence about AI risk."]])
        doc = prepare_document("tiny.pdf", data, ingest_settings)
        assert [c.page for c in doc.chunks] == [2]

    def test_custom_config_is_supported(self, ingest_settings):
        doc = prepare_document("x.pdf", simple_pdf(), ingest_settings, ChunkConfig("tiny", 40, 5))
        assert all(len(c.text) <= 40 for c in doc.chunks)


# ==========================================================================
# Pipeline
# ==========================================================================

class TestPipeline:
    def test_prepare_document_summary(self, ingest_settings):
        data = simple_pdf()
        doc = prepare_document("../uploads/report.pdf", data, ingest_settings)
        assert doc.filename == "report.pdf"
        assert doc.file_type == "pdf"
        assert doc.page_count == 2
        assert doc.chunk_config == ingest_settings.chunk_config == "A"
        assert doc.size_bytes == len(data)
        assert doc.document_id == hashlib.sha256(data).hexdigest()[:16]

    def test_same_bytes_give_same_id_and_hash(self, ingest_settings):
        first = prepare_document("one.docx", simple_docx(), ingest_settings)
        second = prepare_document("renamed.docx", simple_docx(), ingest_settings)
        assert (first.document_id, first.sha256) == (second.document_id, second.sha256)

    def test_rejections_propagate_as_app_errors(self, ingest_settings):
        with pytest.raises(UnsupportedFileTypeError):
            prepare_document("evil.exe", b"MZ...", ingest_settings)


# ==========================================================================
# Real corpus (corpus/ is committed to the repository)
# ==========================================================================

CORPUS_FILES = sorted(p for p in CORPUS_DIR.glob("*") if p.suffix.lower() in (".pdf", ".docx"))


def test_corpus_contains_only_source_documents():
    names = [p.name.lower() for p in CORPUS_DIR.iterdir()]
    assert names, "corpus/ is empty"
    assert not any("portfolio-01" in n or "project plan" in n or "rag service plan" in n for n in names)


@pytest.mark.parametrize("config_name", ["A", "B"])
@pytest.mark.parametrize("path", CORPUS_FILES, ids=[p.name for p in CORPUS_FILES])
def test_corpus_documents_chunk_with_citable_metadata(path, config_name, ingest_settings):
    config = CHUNK_CONFIGS[config_name]
    doc = prepare_document(path.name, path.read_bytes(), ingest_settings, config)

    assert doc.chunks
    assert len({c.chunk_id for c in doc.chunks}) == len(doc.chunks)
    for chunk in doc.chunks:
        assert chunk.filename == path.name
        assert len(chunk.text) <= config.chunk_size
        assert chunk.page is not None or chunk.section_order == 0  # only a DOCX preface lacks a page
        assert "NIST AI 100-1 AI RMF 1.0" not in chunk.text  # running header removed


def test_corpus_pdf_sections_come_from_verified_bookmarks(ingest_settings):
    path = CORPUS_DIR / "NIST.AI.100-1.pdf"
    extracted = extract_pdf(path.read_bytes(), ingest_settings)
    by_page = {s.page: s.section for s in extracted.sections}
    assert extracted.page_count == 48
    assert by_page[6] == "Executive Summary"
    assert by_page[40].startswith("Appendix A")
    assert by_page[47].startswith("Appendix D")
