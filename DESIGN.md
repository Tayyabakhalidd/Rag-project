# Design

> Work in progress. Each section is written when its decision is made. Numbers
> quoted here come from runs of the code (see `notebooks/rag_pipeline.ipynb`).
> Retrieval and answer quality results will be in `EVAL.md`.

## Ingestion flow

```
upload -> validate -> extract -> chunk -> (embed -> store: next step)
          app/ingestion/validation.py
                      app/ingestion/extract.py
                                 app/ingestion/chunk.py
          orchestrated by app/ingestion/pipeline.py::prepare_document
```

`prepare_document` contains no HTTP code, so the API route, the notebook and
the evaluation script all call the same function. It is CPU-bound, so the API
will run it in a worker thread.

## Upload validation

| Check | Result if it fails |
|---|---|
| File name reduced to a safe base name (no folders, no control characters, max 200 chars) | name is cleaned, upload continues |
| Extension is `.pdf` or `.docx` | 415 `unsupported_file_type` |
| File is not empty | 400 `empty_document` |
| Size ≤ `MAX_UPLOAD_MB` (default 20 MB), enforced while reading the stream | 413 `file_too_large` |
| File signature matches the extension (`%PDF-`; DOCX = ZIP with `word/document.xml`) | 415 |
| Legacy `.doc` / password-protected Word file (OLE signature) | 415, with advice |
| DOCX uncompressed size ≤ 100 MB and ≤ 5,000 parts (ZIP-bomb guard) | 400 `invalid_document` |
| PDF not password-protected, not corrupt, ≤ 500 pages | 400 / 413 |

The extension alone is never trusted. The SHA-256 of the raw bytes is used to
detect duplicates, and its first 16 hex characters become the `document_id`, so
the same bytes always get the same ID.

## Text extraction

**PDF: `pypdf`, one section per page.** Page numbers are the page's position
in the PDF, counted from 1. They are not the printed page labels; for example,
NIST AI 100-1's printed "Page 5" is PDF page 10. Clean-up:

* Unicode NFKC normalisation, so ligatures such as `ﬁ` become `fi`;
* words split by a hyphen at the end of a line are re-joined when the next
  line starts in lower case (`mea-\nsure` becomes `measure`). This can wrongly
  join a real hyphenated compound that breaks at a line end;
* header and footer lines that repeat at the top or bottom of at least half of
  the pages are removed. Only those top/bottom positions are touched. On the
  corpus this removes `NIST AI 100-1 AI RMF 1.0` and `Page N`.

**PDF section names come from bookmarks, checked against the text.** NIST AI
100-1's bookmarks are imprecise: "Executive Summary" and "Part 1" both point to
page 1, and appendices A–C all point to page 38. A bookmark is used only if its
heading line appears on its target page or on a later page. Table-of-contents
pages are skipped, and a bookmark that can't be confirmed is dropped together
with its children. This puts the appendices on their real pages (40, 43, 45,
47). Limitation: names are per page, so when two sections start on the same
page, the later one names the whole page.

**DOCX: `python-docx`, split at headings.** The corpus DOCX files were
converted from PDFs, and each original page is marked by a `Heading 1`
paragraph `Page N`. These labels set the chunk's `page`, so DOCX answers can
cite the source page. Ordinary `Heading 1–9` paragraphs build the section path,
for example `Risks > Prompt injection`. The corpus DOCX files have none, so
their `section` is empty. Tables become one `cell | cell` line per row. Running
headers and footers are removed in the same way as for PDFs.

python-docx's `paragraph.style` re-reads the style table on every call. In
profiling, that took about 15 of 16 seconds for the 2,400-paragraph NIST AI
600-1 file. Looking style names up once cut extraction to 0.4 s with identical
output.

**Scanned pages and OCR.** A page without a text layer is listed in
`empty_pages`. A document with no text at all is rejected with 400. OCR is out
of scope.

## Chunking

**Splitter: LangChain `RecursiveCharacterTextSplitter`** (the only LangChain
component used). It cuts at the strongest available boundary: blank line, line,
sentence, clause, word. Sizes are in characters.

**Chunks never cross a section.** Every chunk therefore has exactly one page or
section to cite. The cost is that a passage running over a page break is split
there.

**Overlap is added by our own code.** The splitter's built-in overlap only
repeats whole pieces. In this corpus the pieces are lines of about 90
characters, so the built-in overlap was much smaller than configured. Measured
on the corpus with the built-in overlap:

| Config | Target overlap | Mean actual overlap | Neighbour pairs with no overlap |
|---|---|---|---|
| A 1000/150 | 150 | 78–109 chars | 0–16% |
| B 500/75 | 75 | 14–21 chars | 55–70% |

So the splitter now picks cut points at `chunk_size − overlap`, and each chunk
is extended backwards by up to `overlap` characters, starting at a word
boundary. Measured after the change: mean overlap of 144–146 characters (A) and
69–70 characters (B), with almost no zero-overlap pairs. Chunks are still at
most `chunk_size` characters and are exact slices of their section's text
(`char_start` and `char_end` are stored).

**Configurations compared in the evaluation**

| Config | Size | Overlap | Chunks on the corpus |
|---|---|---|---|
| A | 1000 | 150 | 511 |
| B | 500 | 75 | 1,000 |

Fragments with fewer than 20 visible characters (for example a lone figure
label) are dropped. **The winning configuration is not chosen yet.** It will be
picked from the retrieval metrics in `EVAL.md`.

**Chunk metadata:** `chunk_id` (`{document_id}:{chunk_index}`), `document_id`,
`filename`, `file_type`, `page`, `section`, `chunk_index`, `section_order`,
`char_start`, `char_end`, `sha256`, `chunk_config`.

**Known retrieval risk to check in the evaluation:** the OWASP DOCX keeps its
table of contents (pages 3–4) as text, and dot-leader lines that list many
titles may match many questions.
