"""One-command evaluation of the RAG pipeline.

Usage (from the project root, with the virtual environment active)::

    python -m evals.run_evaluation --all              # configs A and B, every pass
    python -m evals.run_evaluation --all --retrieval-only
    python -m evals.run_evaluation --config B
    python -m evals.run_evaluation --rejudge evals/results/<run-folder>

For each chunking configuration the script:

1. builds a fresh Chroma collection from ``corpus/`` through the real
   ``DocumentService`` (embeddings come from the SQLite cache when available,
   so re-runs cost almost no embedding calls);
2. retrieval pass: embeds each golden question, retrieves the top 5 chunks and
   computes hit@k, recall@k (k = 1, 3, 5) and MRR (no LLM calls);
3. answer pass: runs every question through the real ``QAService``
   (guardrails, thresholds, Groq answer, citation and output checks) and saves
   each answer to ``answers_<config>.jsonl``;
4. judge pass: a different Groq model scores faithfulness and correctness;
   judging reads the saved answers, so it can be repeated with ``--rejudge``.

Results go to ``evals/results/<UTC timestamp>/``: ``summary.json``,
``summary.md``, one per-question CSV per configuration, the answer and judge
JSONL files, and ``judge_calibration_sample.csv`` for hand labelling. Every
summary records the model IDs, thresholds and git commit.

Calls run one at a time with a short pause, and free-tier rate-limit errors
are retried with a growing wait, so a full run takes several minutes.
"""

import argparse
import asyncio
import csv
import json
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.config import CHUNK_CONFIGS, CORPUS_DIR, PROJECT_ROOT, Settings, get_settings
from app.errors import AppError, GuardrailBlockedError, UpstreamRateLimitedError, UpstreamUnavailableError
from app.generation import PROMPT_VERSION, GroqLLM
from app.ingestion.embed import build_embedder
from app.retrieval import Retriever
from app.services.document_service import DocumentService
from app.services.qa_service import QAService
from app.storage.registry import DocumentRegistry
from app.storage.vector_store import ChromaVectorStore
from evals.judge import judge_messages, parse_verdict
from evals.metrics import per_question_retrieval, rate, retrieval_summary

GOLDEN_PATH = PROJECT_ROOT / "data" / "golden_questions.json"
RESULTS_DIR = PROJECT_ROOT / "evals" / "results"
KS = (1, 3, 5)
RETRY_WAITS = (15, 30, 60, 60, 90)  # seconds to wait after each free-tier rate-limit error


def log(message: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {message}", flush=True)


# --- Free-tier friendly wrappers -------------------------------------------

async def patient(call, what: str):
    """Run ``call()``; on a provider rate limit or outage, wait and retry."""
    for attempt, wait in enumerate((*RETRY_WAITS, None), start=1):
        try:
            return await call()
        except (UpstreamRateLimitedError, UpstreamUnavailableError) as exc:
            if wait is None:
                raise
            log(f"  {what}: {exc.code} (attempt {attempt}), waiting {wait}s")
            await asyncio.sleep(wait)


class PatientEmbedder:
    def __init__(self, inner) -> None:
        self.inner = inner

    async def embed(self, texts):
        return await patient(lambda: self.inner.embed(texts), "embedding")


class PatientLLM:
    """Retries rate limits and pauses between calls to respect free-tier limits."""

    def __init__(self, inner: GroqLLM, pause: float) -> None:
        self.inner = inner
        self.pause = pause
        self.calls = 0

    async def complete(self, messages):
        if self.calls:
            await asyncio.sleep(self.pause)
        self.calls += 1
        return await patient(lambda: self.inner.complete(messages), f"LLM {self.inner.model}")


# --- Helpers ---------------------------------------------------------------

def git_commit() -> str:
    try:
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                                cwd=PROJECT_ROOT, check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True,
                               cwd=PROJECT_ROOT, check=True).stdout.strip()
        return f"{commit}{' (uncommitted changes)' if dirty else ''}"
    except Exception:  # noqa: BLE001 - git is optional for running the evaluation
        return "unknown"


def load_golden() -> list[dict[str, Any]]:
    return json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))["questions"]


def chunk_dict(chunk) -> dict[str, Any]:
    return {"citation_id": chunk.citation_id, "chunk_id": chunk.chunk_id, "filename": chunk.filename,
            "page": chunk.page, "section": chunk.section, "score": chunk.score, "text": chunk.text}


def percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round(pct / 100 * (len(ordered) - 1))))
    return ordered[index]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


# --- Passes ----------------------------------------------------------------

async def build_index(settings: Settings, config_name: str, embedder) -> tuple[DocumentRegistry, ChromaVectorStore, dict]:
    """Index every corpus file with one chunking configuration into a fresh collection."""
    eval_dir = settings.data_dir / "eval"
    eval_dir.mkdir(parents=True, exist_ok=True)
    registry = DocumentRegistry(eval_dir / f"registry_{config_name}.sqlite3")
    registry.clear_documents()
    vectors = ChromaVectorStore(eval_dir / "chroma", collection_name=f"eval_{config_name}")
    vectors.reset()
    config_settings = settings.model_copy(update={"chunk_config": config_name})
    service = DocumentService(config_settings, registry, vectors, embedder)
    corpus = {}
    for path in sorted(p for p in CORPUS_DIR.iterdir() if p.suffix.lower() in (".pdf", ".docx")):
        started = time.perf_counter()
        document = await service.upload(path.name, path.read_bytes())
        corpus[path.name] = {"chunks": document["chunk_count"], "pages": document["page_count"]}
        log(f"  indexed {path.name}: {document['chunk_count']} chunks in {time.perf_counter() - started:.1f}s")
    return registry, vectors, corpus


async def retrieval_pass(retriever: Retriever, questions: list[dict], top_k: int) -> list[dict[str, Any]]:
    rows = []
    for q in questions:
        chunks = await retriever.search(q["question"], top_k)
        retrieved = [chunk_dict(c) for c in chunks]
        row = {"id": q["id"], "type": q["type"], "answerable": q["answerable"],
               "best_score": retrieved[0]["score"] if retrieved else None,
               "retrieved": [{k: c[k] for k in ("filename", "page", "score", "chunk_id")} for c in retrieved]}
        if q["answerable"]:
            row.update(per_question_retrieval(retrieved, q["expected_sources"], KS))
        rows.append(row)
    return rows


async def answer_pass(qa: QAService, questions: list[dict]) -> list[dict[str, Any]]:
    rows = []
    for q in questions:
        record: dict[str, Any] = {"id": q["id"], "type": q["type"], "question": q["question"],
                                  "golden_answerable": q["answerable"], "reference_answer": q["reference_answer"]}
        try:
            result = await qa.ask(q["question"])
            record.update({
                "answerable": result.answerable, "refusal_reason": result.refusal_reason,
                "answer": result.answer, "general_knowledge": result.general_knowledge,
                "citations": [chunk_dict(c) for c in result.citations],
                "retrieved": [chunk_dict(c) for c in result.retrieved],
                "best_score": result.best_score, "llm_called": result.llm_called,
                "raw_citation_count": result.raw_citation_count,
                "invalid_citation_count": result.invalid_citation_count,
                "latency_ms": result.latency_ms, "error": None,
            })
        except GuardrailBlockedError as exc:
            record.update({"answerable": False, "refusal_reason": "guardrail_blocked", "error": exc.code,
                           "citations": [], "retrieved": [], "latency_ms": None})
        except AppError as exc:
            record.update({"answerable": False, "refusal_reason": None, "error": exc.code,
                           "citations": [], "retrieved": [], "latency_ms": None})
        outcome = record.get("refusal_reason") or ("answered" if record.get("answerable") else record.get("error"))
        log(f"  {q['id']} ({q['type']}): {outcome}")
        rows.append(record)
    return rows


async def judge_pass(judge: PatientLLM, answers: list[dict]) -> list[dict[str, Any]]:
    """Score answered questions. Refused answerable questions get correctness 0 without a call."""
    verdicts = []
    for record in answers:
        if not record["golden_answerable"]:
            continue
        if not record.get("answerable"):
            verdicts.append({"id": record["id"], "faithful": None, "correctness": 0,
                             "reason": "System did not give a cited answer.", "judged": False})
            continue
        try:
            raw = await judge.complete(judge_messages(record["question"], record["reference_answer"],
                                                      record["answer"], record["citations"]))
            verdict = parse_verdict(raw)
            verdicts.append({"id": record["id"], **verdict.model_dump(), "judged": True})
        except AppError as exc:
            verdicts.append({"id": record["id"], "faithful": None, "correctness": None,
                             "reason": f"judge error: {exc.code}", "judged": False})
        log(f"  judged {record['id']}: {verdicts[-1]['faithful']} / {verdicts[-1]['correctness']}")
    return verdicts


def answer_metrics(answers: list[dict], verdicts: list[dict]) -> dict[str, Any]:
    by_id = {v["id"]: v for v in verdicts}
    answerable = [a for a in answers if a["golden_answerable"]]
    unanswerable = [a for a in answers if a["type"] == "unanswerable"]
    off_topic = [a for a in answers if a["type"] == "off_topic"]
    judged = [by_id[a["id"]] for a in answerable if by_id.get(a["id"], {}).get("judged")]
    correctness = [by_id[a["id"]]["correctness"] for a in answerable if by_id.get(a["id"], {}).get("correctness") is not None]
    raw = sum(a.get("raw_citation_count") or 0 for a in answers)
    invalid = sum(a.get("invalid_citation_count") or 0 for a in answers)
    latencies = [a["latency_ms"] for a in answers if a.get("latency_ms") is not None]
    return {
        "answered_rate_on_answerable": rate(a.get("answerable") is True for a in answerable),
        "false_refusal_rate": rate(a.get("answerable") is not True for a in answerable),
        "faithfulness_rate": rate(v["faithful"] is True for v in judged),
        "mean_correctness_0_2": statistics.mean(correctness) if correctness else None,
        "correctness_2_rate": rate(c == 2 for c in correctness),
        "unanswerable_handled_rate": rate(
            a.get("refusal_reason") == "not_in_documents" and not a.get("citations") for a in unanswerable),
        "off_topic_refusal_rate": rate(a.get("refusal_reason") == "off_topic" for a in off_topic),
        "off_topic_refused_without_llm": rate(a.get("refusal_reason") == "off_topic" and not a.get("llm_called")
                                              for a in off_topic),
        "citation_validity_rate_raw": (raw - invalid) / raw if raw else None,
        "errors": sum(1 for a in answers if a.get("error")),
        "latency_p50_ms": percentile(latencies, 50),
        "latency_p95_ms": percentile(latencies, 95),
        "judged_answers": len(judged),
    }


# --- Output ----------------------------------------------------------------

def write_per_question_csv(path: Path, questions, retrieval_rows, answers, verdicts) -> None:
    retrieval_by_id = {r["id"]: r for r in retrieval_rows}
    answers_by_id = {a["id"]: a for a in answers}
    verdicts_by_id = {v["id"]: v for v in verdicts}
    fields = ["id", "type", "question", "best_score", "hit@1", "hit@3", "hit@5", "recall@5", "rr",
              "top_sources", "outcome", "citations", "faithful", "correctness", "judge_reason"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for q in questions:
            r = retrieval_by_id.get(q["id"], {})
            a = answers_by_id.get(q["id"], {})
            v = verdicts_by_id.get(q["id"], {})
            writer.writerow({
                "id": q["id"], "type": q["type"], "question": q["question"], "best_score": r.get("best_score"),
                "hit@1": r.get("hit@1"), "hit@3": r.get("hit@3"), "hit@5": r.get("hit@5"),
                "recall@5": r.get("recall@5"), "rr": r.get("rr"),
                "top_sources": "; ".join(f"{c['filename']} p{c['page']} ({c['score']:.3f})" for c in r.get("retrieved", [])[:3]),
                "outcome": a.get("refusal_reason") or ("answered" if a.get("answerable") else a.get("error")),
                "citations": "; ".join(f"{c['filename']} p{c['page']}" for c in a.get("citations", [])),
                "faithful": v.get("faithful"), "correctness": v.get("correctness"), "judge_reason": v.get("reason"),
            })


def write_calibration_sample(path: Path, results: dict[str, dict]) -> None:
    """15 answered questions (both configs) with judge scores and empty columns for hand labels."""
    rows = []
    for config_name, data in results.items():
        verdicts = {v["id"]: v for v in data.get("verdicts", [])}
        for a in data.get("answers", []):
            v = verdicts.get(a["id"])
            if v and v.get("judged"):
                rows.append({"config": config_name, "id": a["id"], "question": a["question"],
                             "answer": a["answer"], "reference_answer": a["reference_answer"],
                             "cited_passages": " || ".join(f"[{c['citation_id']}] {c['text'][:400]}" for c in a["citations"]),
                             "judge_faithful": v["faithful"], "judge_correctness": v["correctness"],
                             "human_faithful": "", "human_correctness": ""})
    rows.sort(key=lambda r: (r["id"], r["config"]))
    sample = rows[::2][:15] if len(rows) > 15 else rows  # spread over questions and both configs
    if not sample:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(sample[0]))
        writer.writeheader()
        writer.writerows(sample)


def fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def write_summary_md(path: Path, meta: dict, results: dict[str, dict]) -> None:
    lines = [f"# Evaluation run {meta['run_id']}", "",
             f"- Git commit: `{meta['git_commit']}`",
             f"- Answer model: `{meta['llm_model']}`; judge model: `{meta['judge_model']}`",
             f"- Embedding model: `{meta['embedding_model']}` ({meta['embedding_dimensions']} dimensions)",
             f"- top_k = {meta['top_k']}, off-topic floor = {meta['off_topic_floor']}, "
             f"answer threshold = {meta['answer_threshold']}",
             f"- Prompt version: {meta.get('prompt_version', 'v1')}; golden set: {meta['questions']} questions", "", "## Retrieval", "",
             "| Config | hit@1 | hit@3 | hit@5 | recall@1 | recall@3 | recall@5 | MRR |", "|---|---|---|---|---|---|---|---|"]
    for name, data in results.items():
        s = data["retrieval_summary"]
        lines.append(f"| {name} | " + " | ".join(fmt(s.get(k)) for k in
                     ("hit@1", "hit@3", "hit@5", "recall@1", "recall@3", "recall@5", "mrr")) + " |")
    if any("answer_metrics" in d for d in results.values()):
        keys = ["answered_rate_on_answerable", "false_refusal_rate", "faithfulness_rate", "mean_correctness_0_2",
                "correctness_2_rate", "unanswerable_handled_rate", "off_topic_refusal_rate",
                "off_topic_refused_without_llm", "citation_validity_rate_raw", "errors",
                "latency_p50_ms", "latency_p95_ms", "judged_answers"]
        lines += ["", "## Answers", "", "| Metric | " + " | ".join(results) + " |",
                  "|---|" + "---|" * len(results)]
        for key in keys:
            lines.append(f"| {key} | " + " | ".join(fmt(d.get("answer_metrics", {}).get(key)) for d in results.values()) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# --- Main ------------------------------------------------------------------

def save_run(out_dir: Path, meta: dict, results: dict[str, dict]) -> None:
    """Write summary.json and summary.md (called after every configuration, so a crash loses nothing)."""
    (out_dir / "summary.json").write_text(json.dumps({"meta": meta, "results": results}, indent=2,
                                                     ensure_ascii=False), encoding="utf-8")
    write_summary_md(out_dir / "summary.md", meta, results)


async def run(configs: list[str], retrieval_only: bool, skip_judge: bool, pause: float,
              into: Path | None = None) -> Path:
    settings = get_settings()
    if not settings.gemini_configured:
        sys.exit("GEMINI_API_KEY and EMBEDDING_MODEL must be set in .env to run the evaluation.")
    if not retrieval_only and not (settings.groq_configured and settings.judge_model):
        sys.exit("GROQ_API_KEY, LLM_MODEL and JUDGE_MODEL must be set in .env for the answer and judge passes.")

    questions = load_golden()
    meta = {"git_commit": git_commit(), "llm_model": settings.llm_model,
            "judge_model": settings.judge_model, "embedding_model": settings.embedding_model,
            "embedding_dimensions": settings.embedding_dimensions, "top_k": settings.top_k,
            "off_topic_floor": settings.off_topic_floor, "answer_threshold": settings.answer_threshold,
            "questions": len(questions), "prompt_version": PROMPT_VERSION, "retrieval_only": retrieval_only}
    results: dict[str, dict] = {}
    if into:
        # Add configurations to an existing run (for example after a free-tier quota reset).
        out_dir = into
        previous = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
        settings_keys = ("llm_model", "judge_model", "embedding_model", "embedding_dimensions", "top_k",
                         "off_topic_floor", "answer_threshold", "questions", "prompt_version")
        changed = [k for k in settings_keys if previous["meta"].get(k) != meta[k]]
        if changed:
            sys.exit(f"Settings differ from the run in {out_dir.name}: {changed}. Start a new run instead.")
        results = previous["results"]
        meta = {**previous["meta"], "git_commit_" + "_".join(configs): meta["git_commit"]}
    else:
        run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        out_dir = RESULTS_DIR / run_id
        out_dir.mkdir(parents=True, exist_ok=True)
        meta = {"run_id": run_id, **meta}
    meta.setdefault("configs", {}).update({n: vars(CHUNK_CONFIGS[n]) for n in configs})

    cache = DocumentRegistry(settings.data_dir / "documents.sqlite3")  # shared embedding cache
    embedder = build_embedder(settings, cache)
    patient_embedder = PatientEmbedder(embedder)
    for name in configs:
        log(f"Config {name}: chunk size {CHUNK_CONFIGS[name].chunk_size}, overlap {CHUNK_CONFIGS[name].chunk_overlap}")
        calls_before = embedder.provider_calls
        registry, vectors, corpus = await build_index(settings, name, patient_embedder)
        retriever = Retriever(vectors, patient_embedder)
        log("  retrieval pass")
        retrieval_rows = await retrieval_pass(retriever, questions, settings.top_k)
        data = {"corpus": corpus, "retrieval": retrieval_rows,
                "retrieval_summary": retrieval_summary([r for r in retrieval_rows if r["answerable"]], KS)}
        log(f"  retrieval: {data['retrieval_summary']}")
        if not retrieval_only:
            log("  answer pass")
            qa = QAService(settings.model_copy(update={"chunk_config": name}), registry, retriever,
                           PatientLLM(GroqLLM(settings), pause))
            data["answers"] = await answer_pass(qa, questions)
            write_jsonl(out_dir / f"answers_{name}.jsonl", data["answers"])
            data["verdicts"] = []
            if not skip_judge:
                log("  judge pass")
                judge = PatientLLM(GroqLLM(settings, model=settings.judge_model), pause)
                data["verdicts"] = await judge_pass(judge, data["answers"])
                write_jsonl(out_dir / f"judged_{name}.jsonl", data["verdicts"])
            data["answer_metrics"] = answer_metrics(data["answers"], data["verdicts"])
        data["embedding_requests"] = embedder.provider_calls - calls_before
        write_per_question_csv(out_dir / f"per_question_{name}.csv", questions, retrieval_rows,
                               data.get("answers", []), data.get("verdicts", []))
        results[name] = data
        results = dict(sorted(results.items()))
        save_run(out_dir, meta, results)

    if not retrieval_only and not skip_judge:
        write_calibration_sample(out_dir / "judge_calibration_sample.csv", results)
    log(f"Results written to {out_dir.relative_to(PROJECT_ROOT)}")
    return out_dir


async def rejudge(run_dir: Path, pause: float) -> None:
    """Repeat only the judge pass on saved answers (no new answers are generated)."""
    settings = get_settings()
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    judge = PatientLLM(GroqLLM(settings, model=settings.judge_model), pause)
    for name, data in summary["results"].items():
        answers = read_jsonl(run_dir / f"answers_{name}.jsonl")
        data["verdicts"] = await judge_pass(judge, answers)
        data["answers"] = answers
        data["answer_metrics"] = answer_metrics(answers, data["verdicts"])
        write_jsonl(run_dir / f"judged_{name}.jsonl", data["verdicts"])
    summary["meta"]["judge_model"] = settings.judge_model
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    write_summary_md(run_dir / "summary.md", summary["meta"], summary["results"])
    write_calibration_sample(run_dir / "judge_calibration_sample.csv", summary["results"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate retrieval and answers on the golden set.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--all", action="store_true", help="evaluate configurations A and B")
    group.add_argument("--config", choices=sorted(CHUNK_CONFIGS), help="evaluate one configuration")
    group.add_argument("--rejudge", type=Path, help="re-run only the judge pass on a saved run folder")
    parser.add_argument("--retrieval-only", action="store_true", help="skip the LLM answer and judge passes")
    parser.add_argument("--skip-judge", action="store_true", help="skip the judge pass")
    parser.add_argument("--pause", type=float, default=2.0, help="seconds between LLM calls (free tier)")
    parser.add_argument("--into", type=Path, help="add the configuration(s) to an existing run folder")
    args = parser.parse_args()
    if sys.stdout.encoding.lower() != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8")
    if args.rejudge:
        asyncio.run(rejudge(args.rejudge, args.pause))
        return
    configs = sorted(CHUNK_CONFIGS) if args.all else [args.config]
    asyncio.run(run(configs, args.retrieval_only, args.skip_judge, args.pause, args.into))


if __name__ == "__main__":
    main()
