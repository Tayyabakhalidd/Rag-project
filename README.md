# Production RAG Service

A question-answering service over your own documents. Upload PDF or DOCX files and ask questions in plain language. Answers cite the exact passages they come from, or the service says clearly that the documents don't contain the answer.

*Agentic AI Portfolio Project 01 (IEC).* Built with FastAPI, Gemini embeddings, Chroma, a Groq-hosted LLM and a server-rendered HTML/CSS frontend.

| Document | Contents |
|---|---|
| [DESIGN.md](DESIGN.md) | Architecture and the reasons behind every technical decision |
| [EVAL.md](EVAL.md) | Golden set, metrics, configuration comparison and failure analysis |
| [AI_USAGE.md](AI_USAGE.md) | How AI assistance was used in this project |
| [corpus/ATTRIBUTION.md](corpus/ATTRIBUTION.md) | Sources and licences of the corpus documents |

## What it does

- **Upload and ingest:** PDF and DOCX files are validated, split into page-aware chunks, embedded and stored. Uploading the same file twice returns `409`.
- **Grounded answers:** the most similar chunks are retrieved and the LLM answers only from them, citing `[C1]`, `[C2]` and so on. Every citation is checked against the retrieved chunks.
- **Safe refusals:** "not in your documents" (with general knowledge in a separate, labelled box), a plain refusal for off-topic questions, and `400` for prompt-injection or jailbreak attempts.
- **Deletion:** deleting a document removes its vectors, so its text can never be retrieved or cited again.
- **Frontend:** one page where you upload, list and delete documents, ask a question, and read the answer with its sources and the retrieved chunks with their similarity scores.
- **Evaluation:** a 33-question golden set, hit@k / recall@k / MRR, an LLM judge, and a comparison of two chunking configurations, run with one command.

## Project structure

```
app/
  main.py               app factory: wires settings, services, middleware, routers (no business logic)
  config.py             every setting and constant (pydantic-settings, reads .env)
  schemas.py            request/response models with OpenAPI examples
  errors.py             application errors -> HTTP status + error code
  logging_setup.py      JSON logs with request IDs; secrets redacted
  middleware.py         request ID, access log with latency, generic 500
  guardrails.py         input patterns, output checks, rate limiter
  retrieval.py          query embedding + Chroma similarity search
  generation.py         prompt, Groq client, JSON parsing, citation validation
  api/                  routes.py (health, documents), qa.py (ask, search), ui.py (HTML pages), deps.py
  services/             document_service.py (upload/list/delete), qa_service.py (the answer flow)
  ingestion/            validation.py, extract.py, chunk.py, embed.py, pipeline.py, models.py
  storage/              registry.py (SQLite registry + embedding cache), vector_store.py (Chroma)
frontend/               index.html (Jinja2 template) and styles.css; Pico.css comes from a CDN
corpus/                 the three source documents + ATTRIBUTION.md
data/golden_questions.json   the golden question set
evals/                  run_evaluation.py, metrics.py, judge.py, calibrate_judge.py, results/
notebooks/rag_pipeline.ipynb  step-by-step pipeline notebook (with outputs)
tests/                  pytest unit and API tests (fake embedder and fake LLM)
```

`frontend/app.js` is an empty starter file. The pages use no JavaScript.

## Setup

Requirements: Python 3.12 or newer (developed on 3.14), Windows, macOS or Linux.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1          # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
copy .env.example .env               # macOS/Linux: cp .env.example .env
```

Then open `.env` and fill in your own values. `.env` is listed in `.gitignore`. Never commit it.

### Configuration

All settings live in [app/config.py](app/config.py) and can be overridden with environment variables or `.env`. The ones you need to set are:

| Variable | Purpose |
|---|---|
| `GROQ_API_KEY` | Groq API key (free tier works) |
| `LLM_MODEL` | Groq model ID that writes answers |
| `JUDGE_MODEL` | A different Groq model ID, used only by the evaluation judge |
| `GEMINI_API_KEY` | Google AI Studio key for embeddings |
| `EMBEDDING_MODEL` | Gemini embedding model ID |
| `EMBEDDING_DIMENSIONS` | Embedding size (default 768) |

Optional settings, with their defaults:

| Variable | Default | Purpose |
|---|---|---|
| `DATA_DIR` | `data/runtime` | Chroma files and the SQLite registry. Point it outside OneDrive to avoid sync file locks |
| `CHUNK_CONFIG` | `A` | Chunking configuration for uploads (`A` = 1000/150, `B` = 500/75) |
| `TOP_K` | `5` | Chunks retrieved per question |
| `OFF_TOPIC_FLOOR`, `ANSWER_THRESHOLD` | calibrated values in `config.py` | Similarity cut-offs (see EVAL.md) |
| `QUESTIONS_PER_MINUTE`, `UPLOADS_PER_MINUTE` | `10`, `10` | Per-IP rate limits |
| `MAX_UPLOAD_MB` | `20` | Largest accepted file |
| `LLM_TIMEOUT_SECONDS`, `EMBEDDING_TIMEOUT_SECONDS`, `EXTERNAL_MAX_RETRIES` | `30`, `20`, `2` | Timeouts and retries for external calls |
| `LOG_LEVEL` | `INFO` | Log level |

The service starts without keys. `/health` then reports `llm_configured: false`, and uploads or questions return `503` with a clear message.

## Run the service

```powershell
uvicorn app.main:app --reload
```

- Web page: <http://127.0.0.1:8000/>
- Swagger UI: <http://127.0.0.1:8000/docs>
- Health: <http://127.0.0.1:8000/health>

To index the corpus, upload the three files from `corpus/` through the page or the API.

## API

All JSON endpoints are documented with examples in Swagger at `/docs`.

| Method and path | Purpose | Success | Errors |
|---|---|---|---|
| `GET /health` | Status; whether keys are configured (never their values) | 200 | |
| `POST /api/v1/documents` | Upload a PDF/DOCX (multipart field `file`) | 201 | 400, 409, 413, 415, 422, 429, 503 |
| `GET /api/v1/documents` | List documents | 200 | |
| `GET /api/v1/documents/{id}` | One document's metadata | 200 | 404, 422 |
| `DELETE /api/v1/documents/{id}` | Delete a document and its vectors | 204 | 404, 422, 503 |
| `POST /api/v1/ask` | `{"question", "top_k"?, "document_ids"?}`, returns answer, citations and retrieved chunks | 200 | 400, 404, 422, 429, 503 |
| `POST /api/v1/search` | `{"query", "top_k"?}`, retrieval only, no LLM | 200 | 404, 422, 429, 503 |

Examples (PowerShell 7 or any shell with `curl`):

```bash
curl -F "file=@corpus/NIST.AI.100-1.pdf" http://127.0.0.1:8000/api/v1/documents
curl http://127.0.0.1:8000/api/v1/documents
curl -H "Content-Type: application/json" -d "{\"question\": \"What are the four functions of the AI RMF Core?\"}" http://127.0.0.1:8000/api/v1/ask
```

A refusal is still `200`, with `answerable: false` and a `refusal_reason` of `not_in_documents`, `off_topic`, `no_documents` or `output_blocked`. Every error has the same shape:

```json
{"error": {"code": "duplicate_document", "message": "This file was already uploaded.", "request_id": "…", "details": {"document_id": "…"}}}
```

`429` means this service's own rate limit (with a `Retry-After` header). A provider quota running out is `503 upstream_rate_limited`.

## Run the tests

```powershell
python -m pytest
```

The tests replace Gemini and Groq with deterministic fakes, so they make no network calls and need no keys. They cover ingestion, retrieval, citation checks, guardrails, rate limiting, every API outcome and error code, the HTML pages, and the evaluation metrics. If your machine blocks pytest's default temporary folder, add `--basetemp=.pytest_tmp` and delete that folder afterwards.

## Run the evaluation

```powershell
python -m evals.run_evaluation --all
```

This indexes `corpus/` with both chunking configurations, then runs the retrieval metrics, the answer pass and the judge pass. Results are saved in `evals/results/<timestamp>/` as `summary.md`, `summary.json`, per-question CSVs and JSONL answers. Other options:

```powershell
python -m evals.run_evaluation --all --retrieval-only   # embeddings only, no LLM calls
python -m evals.run_evaluation --rejudge evals/results/<run>   # repeat only the judge pass
python -m evals.calibrate_judge                         # judge vs. human labels (evals/judge_calibration.csv)
```

**Current status:** configuration A is fully evaluated (results in [EVAL.md](EVAL.md)). Configuration B is pending because the Gemini free daily quota ran out while indexing it. After the quota resets (midnight Pacific time), run `python -m evals.run_evaluation --all` to evaluate both configurations with the current prompt.

The run uses real APIs (free tiers). Embeddings are cached in SQLite, so re-runs barely touch the Gemini quota. Calls are made one at a time, and rate-limit errors are retried with growing waits, so a full run takes several minutes.

## Notebook

`notebooks/rag_pipeline.ipynb` runs the pipeline step by step. Part 1 covers validation, extraction and chunking. Part 2 covers embedding, retrieval and answering. Part 3 covers threshold calibration and the evaluation results. It is saved with its outputs. Part 2 needs the keys in `.env`.

## Limitations

- **OCR:** scanned pages have no text layer and cannot be indexed.
- **Re-saved files:** duplicate detection uses the file's bytes, so the same document saved again with different bytes is not caught.
- **Single process:** the rate limiter is in memory for one process, and SQLite and Chroma are single-node.
- **No accounts:** every user sees all documents (no login).
- **Fixed citation granularity:** citations point to a page (PDF, or the DOCX page labels) or a heading section, and a passage that crosses a page break is split there.
- **Guardrail patterns:** they catch common injection and jailbreak wording but can be bypassed. The prompt design and output checks are the second layer.
- **Small calibration set:** the similarity thresholds were calibrated on the same small golden set they are evaluated on (see EVAL.md).
- **Free-tier providers:** the free tiers of Groq and Gemini have rate limits, and their terms allow the provider to use the data sent. Only the public corpus was used.
