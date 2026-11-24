# Design

This document explains how the service is built and why each choice was made: what was chosen, the alternatives considered, and the trade-offs. Numbers quoted here come from real runs of the code. The notebook (`notebooks/rag_pipeline.ipynb`) and the evaluation (`EVAL.md`) show where each one comes from.

## 1. Architecture

```
Browser (HTML forms + Pico.css, no JavaScript)
   |  same origin, no CORS
   v
FastAPI app ── middleware: request ID, latency log, generic 500
   ├── UI routes   (app/api/ui.py)      ─┐  thin: read the request, call a service,
   ├── JSON API    (app/api/routes.py,   │  return a response model or template
   │                app/api/qa.py)      ─┘
   v
Services (all business logic)
   ├── DocumentService  upload / list / get / delete
   └── QAService        guardrails -> retrieval -> gate -> LLM -> checks
          ├── guardrails.py   input patterns, output checks, rate limiter
          ├── retrieval.py    embed query -> Chroma similarity search
          ├── generation.py   prompt -> Groq -> parse JSON -> validate citations
          ├── ingestion/      validate -> extract -> chunk -> embed (cached)
          └── storage/        SQLite registry + embedding cache, Chroma vectors
External: Groq (answers, judge), Gemini (embeddings)
```

`app/main.py::create_app` only wires things together. It builds one registry, one vector collection, one cached embedder, the two services and the rate limiters, and registers routers and error handlers. Tests call `create_app(settings, embedder=FakeEmbedder(), llm=FakeLLM())`, so the test suite exercises the real code with deterministic fakes instead of the network.

The HTML pages and the JSON API call the same service methods, so they behave identically: the same guardrails, errors and answers.

## 2. Corpus

| Document | Format indexed | Pages | Why it is in the corpus |
|---|---|---|---|
| NIST AI 100-1, *AI Risk Management Framework 1.0* | PDF | 48 | The core reference for AI risk; well structured, with bookmarks |
| NIST AI 600-1, *Generative AI Profile* | DOCX (converted from PDF in Word) | 64 | Generative-AI risks; long action tables |
| OWASP *Top 10 for LLM Applications 2025* | DOCX (converted from PDF in Word) | 45 | Concrete LLM security risks; matches this project's guardrails |

Sources and licences are in `corpus/ATTRIBUTION.md`. Why this is a fair test:

- **Both required file types are covered.** One PDF and two DOCX files exercise both extractors.
- **The documents overlap in topic but not in wording.** NIST calls hallucinations "confabulation", while OWASP files them under "Misinformation". That tests retrieval under vocabulary mismatch and supports multi-document questions.
- **The text is dense and the same terms repeat.** "Risk", "AI actors" and "trustworthy" appear throughout, so retrieval has to rank closely related passages rather than pick the one obvious match.
- **They are public documents.** Nothing private is sent to the free-tier APIs.

Each document is indexed in one format only. Indexing the same text as both PDF and DOCX would fill the results with near-duplicate chunks.

## 3. Ingestion

```
upload -> validate -> extract -> chunk -> embed -> store
          validation.py extract.py chunk.py embed.py  storage/
orchestrated by ingestion/pipeline.py::prepare_document and services/document_service.py
```

`prepare_document` contains no HTTP code, so the API, the notebook and the evaluation all call the same function. It is CPU-bound and runs in a worker thread (`asyncio.to_thread`). Embedding calls are asynchronous, and Chroma and SQLite calls run in worker threads, so one upload does not block other requests.

**All-or-nothing upload.**

1. The content hash is reserved in SQLite (`UNIQUE` on `sha256`). A second copy gets `409`, even when two identical uploads arrive at the same moment.
2. The chunks are embedded and written to Chroma.
3. The registry row is marked `ready`.

If step 2 or 3 fails (or the request is cancelled), the vectors written so far and the reserved row are removed, and the client gets `503`. Listing and answering only see `ready` documents. A test checks that the same file uploads normally after a failed attempt.

**Deletion.** Vectors are deleted first by a `document_id` metadata filter. The store then confirms that none remain, and only then is the registry row removed. If the vector delete fails, the row is kept and the client gets `503`, so a retry works. Tests check that deleted chunks never appear in search results or answers.

### Upload validation

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

The extension alone is never trusted. The SHA-256 of the raw bytes detects duplicates, and its first 16 hex characters become the `document_id`, so the same bytes always get the same ID. Limitation: a re-saved DOCX has different bytes and is not detected as a duplicate.

### Text extraction

**PDF: `pypdf`, one section per page.** Page numbers are the page's position in the PDF, counted from 1, not the printed page labels. For example, NIST AI 100-1's printed "Page 5" is PDF page 10. Clean-up:

* **Unicode NFKC normalisation**, so ligatures such as `ﬁ` become `fi`.
* **De-hyphenation.** A word split by a hyphen at the end of a line is re-joined when the next line starts in lower case (`mea-\nsure` becomes `measure`). This can wrongly join a real hyphenated compound that breaks at a line end.
* **Header and footer removal.** Lines that repeat at the top or bottom of at least half of the pages are removed, and only in those positions. On the corpus this removes `NIST AI 100-1 AI RMF 1.0` and `Page N`.

**PDF section names come from bookmarks, checked against the text.** NIST AI 100-1's bookmarks are imprecise: "Executive Summary" and "Part 1" both point to page 1, and appendices A–C all point to page 38.

- A bookmark is used only if its heading line appears on its target page or a later one.
- Table-of-contents pages are skipped.
- A bookmark that can't be confirmed is dropped together with its children.

This puts the appendices on their real pages (40, 43, 45, 47). Limitation: names are per page, so when two sections start on the same page, the later one names the whole page.

**DOCX: `python-docx`, split at headings.**

- **Page labels.** The corpus DOCX files were converted from PDFs, and each original page is marked by a `Heading 1` paragraph `Page N`. These labels set the chunk's `page`, so DOCX answers can cite the source page.
- **Section paths.** Ordinary `Heading 1–9` paragraphs build the section path, for example `Risks > Prompt injection`. The corpus DOCX files have none, so their `section` is empty.
- **Tables** become one `cell | cell` line per row.
- **Running headers and footers** are removed in the same way as for PDFs.

The Word conversion left some damage. Words are occasionally split by a stray space, such as `carbon-i ntensive` in NIST AI 600-1. The long action tables also become flat `cell | cell` text, which reads less naturally than prose.

python-docx's `paragraph.style` re-reads the style table on every call. In profiling, that took about 15 of 16 seconds for the 2,400-paragraph NIST AI 600-1 file. Looking style names up once cut extraction to 0.4 s with identical output.

**Scanned pages and OCR.** A page without a text layer is listed in `empty_pages`, and a document with no text at all is rejected with `400`. OCR is out of scope.

### Chunking

**Splitter: LangChain `RecursiveCharacterTextSplitter`**, the only LangChain component used. It cuts at the strongest available boundary: blank line, line, sentence, clause, word. Sizes are in characters.

**Chunks never cross a section.** Every chunk therefore has exactly one page or section to cite. The cost is that a passage running over a page break is split there.

**Overlap is added by our own code.** The splitter's built-in overlap only repeats whole pieces. In this corpus the pieces are lines of about 90 characters, so the built-in overlap was much smaller than configured:

| Config | Target overlap | Mean actual overlap (built-in) | Neighbour pairs with no overlap |
|---|---|---|---|
| A 1000/150 | 150 | 78–109 chars | 0–16% |
| B 500/75 | 75 | 14–21 chars | 55–70% |

So the splitter picks cut points at `chunk_size − overlap`, and each chunk is extended backwards by up to `overlap` characters, starting at a word boundary. After the change the mean overlap is 144–146 characters for A and 69–70 for B. Chunks are still at most `chunk_size` characters and are exact slices of their section's text (`char_start` and `char_end` are stored).

| Config | Size | Overlap | Chunks on the corpus |
|---|---|---|---|
| A | 1000 | 150 | 511 |
| B | 500 | 75 | 1,000 |

Fragments with fewer than 20 visible characters, such as a lone figure label, are dropped. **Configuration B (500/75) is the selected default** after two A/B evaluation runs. B ranked relevant passages higher and showed a small answer-correctness edge; A retained more relevant evidence in its top five. See [EVAL.md](EVAL.md) for the metrics and trade-off.

**Chunk metadata:** `chunk_id` (`{document_id}:{chunk_index}`), `document_id`, `filename`, `file_type`, `page`, `section`, `chunk_index`, `section_order`, `char_start`, `char_end`, `sha256`, `chunk_config`.

## 4. Embedding model

**Choice: Google `gemini-embedding-001`, 768 dimensions, through the Gemini API free tier.**

| | |
|---|---|
| Quality | A current, general-purpose embedding model that ranks highly on public retrieval benchmarks. All 5 reworded (vocabulary-mismatch) golden questions found their source in the top 5 (EVAL.md). |
| Dimensions | Configurable (Matryoshka-style truncation). 768 keeps the Chroma index and the SQLite cache small with little quality loss; set by `EMBEDDING_DIMENSIONS`. |
| Language support | Multilingual (100+ languages). Not used here, but it would make Bonus B possible without changing models. |
| Cost | Free tier, rate-limited. The cache makes re-indexing and re-running the evaluation almost free. |
| Local or API | API. Text leaves the machine, which is acceptable for this public corpus. |
| Latency | About 0.5–1 s per batch of 50 chunks when not rate-limited. A question needs one call, or none when the cache already has it. |

**Alternatives rejected:**

- **A local `sentence-transformers` model (e.g. `bge-small`, `all-MiniLM`).** Private and free, but it needs PyTorch: a large install and slow CPU encoding on an 8 GB laptop. The smaller models are also weaker on dense, formal text.
- **OpenAI `text-embedding-3-small`.** Strong and cheap, but it needs a paid account.
- **Cohere and Voyage.** Good retrieval models, but another account, and no clear gain for this corpus.

The client is the installed `openai` SDK pointed at Google's OpenAI-compatible endpoint, so no extra SDK is needed. A real-API detail found during the first live run: this endpoint returns each embedding with `index: null`, so the adapter keeps the response order instead of sorting by index. A regression test covers this.

**Embedding cache.** SQLite table `embedding_cache` stores one vector per cache key. The key is the SHA-256 of model ID, dimensions and text, so the text itself is not used as a key. Texts are sent in batches of 50 (`EMBEDDING_BATCH_SIZE`), and each finished batch is cached immediately. A free-tier limit hit halfway through a large document therefore loses no work. Questions go through the same cache, so a repeated question costs no embedding call.

**Free-tier behaviour seen in the evaluation.** About 100 texts per minute were accepted before `429` responses. The free daily quota ended after about 940 texts. The SDK retries twice, and then the service returns `503 upstream_rate_limited`. The evaluation script waits and retries instead, and the cache keeps everything embedded before the limit.

## 5. Vector database

**Choice: Chroma, persistent and local** (`chromadb.PersistentClient`, files under `DATA_DIR/chroma`, anonymous telemetry turned off).

| | |
|---|---|
| Index | HNSW (approximate nearest neighbours), Chroma's built-in index |
| Similarity | Cosine (`hnsw:space = cosine`). The score shown to users is `1 − cosine distance`, so higher means more similar |
| Stored per chunk | Embedding, the chunk text, and all chunk metadata above (Chroma metadata cannot hold `None`, so empty fields are left out) |
| Filters | `document_id` metadata filter, used for deletion and for the optional `document_ids` filter on `/ask` and `/search` |
| Scale | Comfortable for thousands to low millions of chunks on one machine. Beyond that, or with several API servers, a client–server store (Qdrant, pgvector, or Chroma server) is needed, which would only change `storage/vector_store.py` |

**Alternatives rejected:**

- **FAISS.** Fast, but it has no metadata or delete-by-filter, so we would have to build that ourselves.
- **Qdrant.** A strong choice, but it needs Docker or a server process. It stays the fallback.
- **pgvector.** Needs PostgreSQL.
- **Pinecone.** A hosted account and network latency, for a project that runs locally.

**Registry: SQLite.**

- **Duplicates.** `UNIQUE(sha256)` gives atomic duplicate detection.
- **Listing.** The listing doesn't depend on the vector store.
- **Embedding cache.** The cache lives in the same file.

The cost is a second store to keep consistent, which is handled by the all-or-nothing upload and the delete order above. Every Chroma error is mapped to `503 upstream_unavailable` ("vector database unavailable").

## 6. Retrieval settings

| Setting | Value | Why |
|---|---|---|
| top-k | 5 (1–10 per request) | Five chunks of selected config B are about 2,500 characters, which leaves room for multi-part questions in the LLM prompt. EVAL.md reports hit@1/3/5 |
| Chunking config | B (500/75), selected | Across two prompt-v2 runs, B had higher MRR (0.764 vs 0.732), hit@1 (0.625 vs 0.583), and hit@3 (0.917 vs 0.875). A had better recall@5 (0.917 vs 0.875); see EVAL.md for the selection rationale. |
| Off-topic floor | 0.55 | Off-topic golden questions scored 0.43–0.48 and in-domain ones at least 0.635. Below the floor the question is refused without calling the LLM (cheap, and visible as a guardrail) |
| Answer threshold | 0.65 | Answerable golden questions scored at least 0.718. Below this, the documents are not trusted to answer, so no cited answer is given. Unanswerable questions overlap with answerable ones in score, so the LLM's `answerable` flag catches them, not this threshold |
| Filters | optional `document_ids` | Ask about specific documents; unknown IDs return `404` |
| Hybrid search / reranking | not used | Left as future work. The failure analysis in EVAL.md shows where they would help |

Both thresholds were calibrated from the best-chunk scores of the golden-set questions (notebook Part 3, EVAL.md). The same small set is then used for evaluation, which flatters the results; EVAL.md lists this as a limitation.

## 7. LLM

**Choice: Groq free tier, `openai/gpt-oss-120b` for answers. The evaluation judge is a different, smaller model, `openai/gpt-oss-20b`.**

| | |
|---|---|
| Answer quality | A 120B-parameter open-weight reasoning model. It follows the JSON format and citation rules reliably, as the smoke test and evaluation show. `reasoning_effort = low` keeps it fast; the setting is sent only to model families that accept it |
| Latency | About 1.0–1.2 s per answer when not throttled (notebook Part 2, and the first evaluation question at 1,061 ms). Under free-tier throttling in the evaluation, p50 was 9.3 s (EVAL.md) |
| Cost per question | $0 on the free tier. One question uses about 2,000–3,000 tokens (5 chunks + prompt + answer). The free tier's per-minute and per-day token limits are the real constraint |
| Context window | 128k tokens, far more than the ~2–3k this service sends |
| Data privacy | Questions and retrieved chunks are sent to Groq. Acceptable for a public corpus; not for confidential documents without a paid plan with suitable terms, or a local model |

The client is the `openai` SDK at Groq's OpenAI-compatible endpoint, with JSON mode (`response_format = json_object`). The timeout is `LLM_TIMEOUT_SECONDS` (30 s), with bounded SDK retries with backoff on 429, 5xx and connection errors (`EXTERNAL_MAX_RETRIES`). Failures become `503` with a safe message.

**Alternatives rejected:**

- **OpenAI or Anthropic hosted models.** High quality, but paid.
- **Gemini Flash.** Possible, but it would share one provider's quota with the embeddings.
- **A local model through Ollama.** Private, but too slow and too large for an 8 GB laptop.
- **Llama 3.3 70B on Groq.** A good fallback. Switching is a one-line `.env` change (`LLM_MODEL`).

**Judge.** The judge uses a different model, so the answer model never grades itself. Both models are from the same family, so they may share blind spots. That is why the judge is checked against hand labels (EVAL.md).

## 8. Prompt design

The prompt is in `app/generation.py`.

- **Chunks are numbered data.** Each retrieved chunk is wrapped as `<document id="C1" source="NIST.AI.100-1.pdf, page 25"> … </document>`. A chunk cannot close its own tag: `neutralize_tags` breaks any `<document` or `</document` inside the text.
- **Answer only from the chunks.** The system prompt says so, and says to set `answerable: false` instead of guessing.
- **Forced citations.** A `[C#]` marker goes after every sentence that uses a chunk, and only the given IDs may be cited.
- **Text in `<document>` tags is data, never instructions.** This is the document guardrail (Bonus F): an uploaded file that says "ignore previous instructions" is quoted, not obeyed. A test checks that such text arrives inside its tag.
- **General knowledge is separate.** It is unverified: in the notebook, the model's background on EU AI Act fines appears to give the 2021 draft's figure. It may only go in the `general_knowledge` field, with no `[C#]` markers; the UI shows it in a box labelled "Not from your documents".
- **Partial answers (prompt v2).** If the chunks cover only part of a question, the model answers that part with citations and says which part the documents do not cover. Prompt v1 refused the whole question (EVAL.md, q24).
- **Background for uncovered questions (prompt v2).** If an in-domain question is not answerable from the chunks, the model writes 1–3 sentences of labelled background. Under v1 it never did, although the approved design requires it.
- **Off-topic flag.** Questions unrelated to the corpus topic (`CORPUS_TOPIC` setting) set `off_topic: true`.
- **Canary.** A random canary token is generated per process and placed in the system prompt, with an instruction never to reveal the prompt.
- **JSON reply.** The model replies with one JSON object, parsed leniently (code fences or text around the object are tolerated) and validated with Pydantic. An unreadable reply is a `503`, not a guess.

After the call, `validate_citations` keeps only markers that point to retrieved chunks and drops invented ones. If the model listed valid citations but put no markers in the text, they are appended to the answer. During testing, the smaller Groq model did exactly that.

### The outcomes

| Situation | How it is detected | Response |
|---|---|---|
| Answered | best score ≥ answer threshold, the LLM says answerable, ≥ 1 valid citation | 200, cited answer (+ optional labelled general knowledge) |
| Not in documents | LLM says not answerable, best score below the answer threshold, or no valid citation survives | 200, `refusal_reason: not_in_documents`, zero citations, labelled general knowledge |
| Off-topic | best score below the off-topic floor (no LLM call), or the LLM flags it | 200, `refusal_reason: off_topic`, polite refusal only |
| No documents | nothing uploaded yet | 200, `refusal_reason: no_documents`, no API calls |
| Output blocked | the answer contains a key, a key-like string, the canary or system-prompt text | 200, `refusal_reason: output_blocked`, answer withheld |
| Injection / jailbreak | input guardrail patterns | 400 `guardrail_blocked` |

A refusal is `200` because the request was valid, and the answer is that the documents don't cover it.

## 9. Guardrails

| Layer | What it checks | Where | Result |
|---|---|---|---|
| Input: shape | Empty or whitespace-only, more than 1,000 characters, invalid `top_k`, unknown fields | Pydantic schema (`schemas.py`) | 422 |
| Input: injection and jailbreak | "ignore previous instructions", "show me your system prompt", "developer mode", "you are now DAN", "reveal your API key", fake `SYSTEM:` headers, `<document>` tags | `guardrails.check_question`, after NFKC normalisation, lower-casing, and removal of zero-width characters and punctuation | 400 `guardrail_blocked` with `details.reason` |
| Input: off-topic | Similarity floor + the LLM's topic flag | `QAService` | 200 refusal |
| Documents | Hidden instructions treated as data | prompt design + `neutralize_tags` | quoted, not followed |
| Output: citations | Citations must be retrieved chunks; an answer with none left becomes "not in documents" | `generation.validate_citations` | invented citations removed |
| Output: leaks | Configured API keys, key-like patterns (`sk-`, `gsk_`, `AIza…`), the canary, verbatim system-prompt sentences | `guardrails.check_output` | 200 `output_blocked` |
| Abuse | 10 questions and 10 uploads per minute per client IP (token bucket) | `guardrails.RateLimiter` | 429 + `Retry-After` |

**Pattern design.** The corpus is *about* these attacks, so a naive filter would refuse legitimate questions such as "What is system prompt leakage?" or "What is a jailbreak?". The patterns therefore target commands addressed to the assistant ("ignore *your* instructions", "show me *your* system prompt") rather than mentions of the topic. Tests cover both lists: attacks that must be blocked, and in-domain questions about attacks that must pass. A test also checks that no golden-set question is blocked.

**Known limit.** Patterns can be bypassed by creative wording, and a question *describing* an attack in imperative form ("what happens if I say ignore previous instructions") is blocked. The prompt design and the output checks are the second and third layers.

## 10. API design

- **Versioned paths** under `/api/v1`. Documents are a resource: `POST`, `GET` the list, `GET` one by ID, `DELETE` by ID. `/ask` and `/search` are actions.
- **`/search` returns chunks and scores without calling the LLM.** It makes retrieval testable, and it is what the evaluation measures.
- **`/ask` returns everything the UI needs:** the answer, the citations (each with `citation_id`, `document_id`, `filename`, `page`, `section`, `chunk_index`), and every retrieved chunk with its score and a `cited` flag.
- **One error shape:** `{"error": {"code", "message", "request_id", "details"}}`. FastAPI's 404, 405 and 422 handlers are replaced so they use it too. The 422 handler reports *where* validation failed, never the submitted value, so questions are not echoed.
- **Request IDs.** Every response carries `X-Request-ID`, and the same ID appears in the error body and in the logs.
- **Swagger.** Every model has examples in Swagger at `/docs`. The UI routes are hidden from it.

### Error handling

| Status | When |
|---|---|
| 200 | Answer, refusal, search, listing |
| 201 | Document uploaded and indexed |
| 204 | Document deleted |
| 400 | Empty file, no extractable text, corrupt or password-protected file, ZIP bomb, blocked question |
| 404 | Unknown document ID (also in `document_ids`), unknown route |
| 409 | Duplicate upload (`details` names the existing document) |
| 413 | File over `MAX_UPLOAD_MB`, PDF over 500 pages |
| 415 | Not a PDF/DOCX, including a renamed text file or `.exe` |
| 422 | Invalid request data |
| 429 | This service's rate limit, with `Retry-After` |
| 503 | Gemini, Groq or Chroma unavailable, timing out, or not configured. A provider's quota is `upstream_rate_limited` |
| 500 | Anything unexpected: a generic message, with details only in the server log |

## 11. Frontend

**Choice: one server-rendered Jinja2 page with Pico.css (CDN), no JavaScript.**

The page shows:

- an upload form;
- the document table, with a Delete button per row (HTML forms cannot send `DELETE`, so it posts to `/ui/documents/{id}/delete`);
- the question form, with an optional top-k;
- the answer, with clickable `[C#]` markers and a Sources list;
- general knowledge in a separate dashed box labelled "Not from your documents";
- refusals in their own style;
- a "Retrieved chunks (N)" section listing each chunk's similarity score as a number and a bar, its source, and a "cited" highlight.

Successful uploads and deletes redirect back to the page (Post/Redirect/Get), so refreshing does not resubmit. Errors are rendered with the real status code and the error code, for example "409 Duplicate". Jinja2 autoescaping is on, so text from documents or the LLM cannot inject HTML; a test checks this.

**Alternatives rejected:** React or Next.js. They would earn bonus marks, but they mean a second toolchain, CORS, and code the student must also explain. Pico.css gives a clean, accessible look for almost no code.

## 12. Reliability and logging

- **Async external calls.** Embeddings and LLM calls use the async OpenAI SDK. File parsing, SQLite and Chroma run in worker threads.
- **Timeouts and retries everywhere.** The LLM times out after 30 s and embeddings after 20 s, each with bounded retries; Chroma errors map to 503.
- **One JSON log line per request,** with request ID, method, path (no query string), status, latency and error code. Questions are logged only by length and a short hash. Request bodies, headers and document text are never logged. API keys are redacted from every line, including tracebacks.
- **Secrets.** Keys exist only in `.env`, which is ignored by Git, and are typed as `SecretStr`, so they never appear in `repr()` or logs. `/health` reports *whether* keys are set, never their values.

## 13. Testing approach

- **Notebook first.** It runs the real pipeline step by step on the corpus: Part 1 extraction and chunking, Part 2 embedding, retrieval and answering, Part 3 calibration and evaluation.
- **Then pytest against the API**, through FastAPI's `TestClient`, with deterministic fakes injected through `create_app`:
  - `FakeEmbedder`, a bag-of-words hashing embedder, so related texts really are more similar;
  - `FakeLLM`, which returns scripted JSON or raises errors on demand.

  The suite is fast, free and repeatable, and never touches the network.
- **What is covered.** Every status code in the table above; all six answer outcomes; citation validation; prompt-injection, jailbreak and secret-request blocking; leaked keys and canaries being withheld; hidden instructions in documents; rate limiting; upstream timeouts and quota errors; upload rollback; deletion; the HTML pages (including HTML escaping); and the evaluation metrics on hand-made examples.
- **Real-API checks.** The live models are exercised by the evaluation run and the notebook, not by the automated suite.

## 14. Limitations and future work

- **Retrieval.** No hybrid (BM25 + vector) search or reranking yet. EVAL.md shows which failures they would fix.
- **Accounts.** There are no user accounts: everyone sees all documents. JWT authentication is Bonus A and was not attempted.
- **Single process.** The in-memory rate limiter and local Chroma/SQLite are single-process and single-node.
- **Ingestion.** There is no OCR, and re-saved files are not recognised as duplicates.
- **Citation granularity.** Citations are per page, and a passage crossing a page break is split there.
- **Calibration.** Thresholds were calibrated on the golden set itself.
