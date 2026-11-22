# Evaluation

Every number in this file comes from a real run saved in `evals/results/20261122T120041Z/`. That folder holds `summary.md`, `summary.json`, the per-question CSV, every answer and every judge verdict.

> **Status (22 Nov 2026).** Configuration **A** (1000/150) is fully evaluated: retrieval, answers and judge. Configuration **B** (500/75) could not be indexed yet. The Gemini free tier's daily quota ran out after about 940 embeddings, about 400 of B's 1,000 chunks in. The quota resets at midnight Pacific time. B, and the A-vs-B comparison, will be added by re-running `python -m evals.run_evaluation --all`. That run also re-evaluates A with prompt v2 (section 7), so both configurations use the same prompt. Until then, the chunking choice in section 5 is marked **pending**.

## 1. Method and tools

The evaluation is one command:

```powershell
python -m evals.run_evaluation --all
```

For each chunking configuration it:

1. Builds a fresh Chroma collection from `corpus/` through the real `DocumentService`. Embeddings come from the SQLite cache when available.
2. **Retrieval pass:** embeds each question and records the top 5 chunks. No LLM.
3. **Answer pass:** sends every question through the real `QAService`: guardrails, thresholds, the Groq answer, citation and output checks. Each answer is saved.
4. **Judge pass:** a different Groq model scores each answer against the reference. Judging reads the saved answers, so it can be repeated with `--rejudge` without re-answering.

Each summary records the git commit, the model IDs, the prompt version, top-k and both thresholds.

| Run 20261122T120041Z | |
|---|---|
| Answer model | `openai/gpt-oss-120b` (Groq, `reasoning_effort=low`, temperature 0) |
| Judge model | `openai/gpt-oss-20b` (Groq) |
| Embeddings | `gemini-embedding-001`, 768 dimensions |
| Retrieval | top-k 5, cosine similarity, off-topic floor 0.55, answer threshold 0.65 |
| Prompt | v1 |
| Commit | `494ef3b` plus the uncommitted work of this step (the run happened before the commit) |

**Why these tools.**

- **Own metric code instead of RAGAS.** hit@k, recall@k and MRR are a few lines each, are unit-tested on hand-made examples (`tests/test_metrics.py`), and need no LLM. RAGAS would add a large dependency, and its LLM-based metrics would spend the same free Groq quota with less control over the prompts.
- **LLM-as-a-judge** for faithfulness and correctness, because those need reading comprehension. It uses a short written rubric (`evals/judge.py`) on a *different* model from the answer model, so a model never grades its own output.
- **No Langfuse.** Every intermediate result (retrieved chunks, scores, raw citation counts, latency, judge reason) is already saved per question in JSON and CSV, which was enough for this analysis without running a tracing server.

**Relevance is labelled by file and page, not by chunk ID**, because chunk IDs differ between configurations A and B. A retrieved chunk is relevant if it comes from the expected file and one of the listed pages. A question can list several pieces of evidence, each with alternative pages; this matters for multi-part questions.

## 2. Golden set

The golden set is in `data/golden_questions.json`: 33 questions, each with its type, expected sources (file + pages), a reference answer and notes. Every source page was checked against the extracted text of the corpus.

| Type | Count | Purpose |
|---|---|---|
| Single fact | 15 | Core retrieval and citation accuracy (7 from NIST AI 100-1, 4 from AI 600-1, 4 from OWASP) |
| Reworded | 5 | Vocabulary mismatch, e.g. "makes things up" → *confabulation*, "run up the cloud bill" → *Denial of Wallet* |
| Multi-part | 4 | Two documents needed in one answer (NIST + OWASP) |
| In-domain unanswerable | 6 | Topics close to the corpus but absent from it: EU AI Act fines, NIST's budget, a recommended temperature, ISO/IEC 42001, adoption numbers, a 2026 edition. Each absence was confirmed by text search |
| Off-topic | 3 | Cookies, the 2018 World Cup, photosynthesis |

A test (`tests/test_metrics.py`) checks that the set meets the brief: at least 25 questions and at least 5 unanswerable. It also checks that every source file exists and that no golden question is blocked by the input guardrail.

## 3. Metrics

| Metric | Definition |
|---|---|
| hit@k | Share of answerable questions with at least one relevant chunk in the top k |
| recall@k | Share of a question's expected sources found in the top k, averaged over questions |
| MRR | Mean of 1 / rank of the first relevant chunk (0 if none in the top 5) |
| Answered rate / false-refusal rate | Answerable questions that got / did not get a cited answer |
| Faithfulness (judge) | Every claim in the answer is supported by its cited passages (pass/fail) |
| Correctness (judge) | 0 = wrong or missing, 1 = partly right, 2 = all key points; refused answerable questions score 0 |
| Unanswerable handled | `refusal_reason = not_in_documents` and zero citations |
| Off-topic refused | `refusal_reason = off_topic` (and whether it happened before any LLM call) |
| Citation validity (raw) | Share of the model's own citations that point to a retrieved chunk, *before* invalid ones are removed |

## 4. Threshold calibration

These are the best-chunk similarity scores of the golden questions, from the retrieval pass of config A. The same table is in notebook Part 3.

| Question type | n | min | median | max |
|---|---|---|---|---|
| Single | 15 | 0.731 | 0.769 | 0.877 |
| Reworded | 5 | 0.733 | 0.741 | 0.824 |
| Multi-part | 4 | 0.718 | 0.738 | 0.749 |
| Unanswerable | 6 | 0.635 | 0.738 | 0.842 |
| Off-topic | 3 | 0.426 | 0.435 | 0.477 |

- **Off-topic floor = 0.55.** It sits in the gap between off-topic questions (≤ 0.477) and every in-domain question (≥ 0.635). Off-topic questions are refused before any LLM call.
- **Answer threshold = 0.65,** safely below the lowest answerable question (0.718).
- **Unanswerable questions cannot be caught by similarity.** Their scores (0.635–0.842) overlap completely with answerable ones, because they use the corpus's vocabulary (e.g. "EU AI Act" and "high-risk AI systems"). The LLM's `answerable` flag must catch them, and it did (section 6).
- **Caveat:** the thresholds were calibrated on the same 33 questions they are evaluated on, with only 3 off-topic examples. They are a sensible starting point, not proven cut-offs.

## 5. Results

### Retrieval (24 answerable questions)

| Config | hit@1 | hit@3 | hit@5 | recall@1 | recall@3 | recall@5 | MRR |
|---|---|---|---|---|---|---|---|
| A (1000/150) | 0.583 | 0.875 | 0.958 | 0.542 | 0.854 | 0.917 | 0.732 |
| B (500/75) | pending | pending | pending | pending | pending | pending | pending |

### Answers

| Metric | A |
|---|---|
| Answered rate on answerable questions | 0.875 (21 of 24) |
| False-refusal rate | 0.125 (3 of 24) |
| Faithfulness (judge, 21 answers) | 1.000 |
| Mean correctness (0–2, all 24 answerable) | 1.625 |
| Share with correctness 2 | 0.792 (19 of 24) |
| Unanswerable handled correctly | 1.000 (6 of 6) |
| Off-topic refused | 1.000 (3 of 3), all before any LLM call |
| Citation validity (raw model output) | 1.000 |
| Errors | 0 |
| Latency p50 / p95 | 9.3 s / 12.9 s (see the note below) |

**Latency note.** The evaluation pauses 2 s between LLM calls, and the Groq SDK silently waits out free-tier rate limits, so these numbers measure throttling more than the service. Unthrottled single questions took about 1.0–1.2 s in the notebook and on the first evaluation question (1,061 ms).

### Configuration comparison: pending

The plan was to pick between A and B with the retrieval metrics (MRR and recall@5 first, hit@1 as the tie-breaker), then confirm with the answer metrics. **Until B has run, A stays the service default (`CHUNK_CONFIG=A`).** Expectations to be confirmed or refuted by the run:

- B's smaller chunks should rank the exact passage higher, which should improve hit@1. Two of A's failures (q06, q14) are relevant passages buried inside long chunks.
- B's smaller chunks may also lose context that multi-part and definition questions need.

### Per-question results (config A)

| ID | Type | Best score | hit@1 | hit@5 | RR | Outcome | Faithful | Correctness |
|---|---|---|---|---|---|---|---|---|
| q01 | single | 0.789 | 1 | 1 | 1.00 | answered | yes | 2 |
| q02 | single | 0.814 | 1 | 1 | 1.00 | answered | yes | 2 |
| q03 | single | 0.804 | 0 | 1 | 0.20 | answered | yes | 2 |
| q04 | single | 0.802 | 0 | 1 | 0.33 | answered | yes | 2 |
| q05 | single | 0.779 | 1 | 1 | 1.00 | answered | yes | 2 |
| q06 | single | 0.828 | 1 | 1 | 1.00 | answered | yes | **0** |
| q07 | single | 0.877 | 1 | 1 | 1.00 | answered | yes | 2 |
| q08 | single | 0.764 | 1 | 1 | 1.00 | answered | yes | 2 |
| q09 | single | 0.765 | 1 | 1 | 1.00 | answered | yes | 2 |
| q10 | single | 0.741 | 0 | 1 | 0.50 | answered | yes | 2 |
| q11 | single | 0.764 | 1 | 1 | 1.00 | answered | yes | 2 |
| q12 | single | 0.731 | 0 | 1 | 0.33 | answered | yes | 2 |
| q13 | single | 0.738 | 0 | 1 | 0.50 | answered | yes | 2 |
| q14 | single | 0.769 | 1 | 1 | 1.00 | **not_in_documents** | - | 0 |
| q15 | single | 0.758 | 1 | 1 | 1.00 | answered | yes | 2 |
| q16 | reworded | 0.741 | 0 | 1 | 0.50 | answered | yes | 2 |
| q17 | reworded | 0.759 | 0 | 1 | 0.50 | answered | yes | 2 |
| q18 | reworded | 0.734 | 1 | 1 | 1.00 | answered | yes | 2 |
| q19 | reworded | 0.824 | 0 | 1 | 0.50 | answered | yes | 2 |
| q20 | reworded | 0.733 | 1 | 1 | 1.00 | answered | yes | 2 |
| q21 | multi_part | 0.743 | 0 | **0** | 0.00 | **not_in_documents** | - | 0 |
| q22 | multi_part | 0.734 | 1 | 1 | 1.00 | answered | yes | 1 |
| q23 | multi_part | 0.749 | 0 | 1 | 0.20 | answered | yes | 2 |
| q24 | multi_part | 0.718 | 1 | 1 | 1.00 | **not_in_documents** | - | 0 |
| q25–q30 | unanswerable | 0.635–0.842 | | | | not_in_documents (6/6) | | |
| q31–q33 | off_topic | 0.426–0.477 | | | | off_topic (3/3), no LLM call | | |

## 6. Failure analysis

Five questions failed or were only partly right. Each was traced through the saved chunks in `answers_A.jsonl`.

**q21: retrieval fails on a broad two-document question (front matter wins).** *"What does NIST AI 600-1 call AI hallucinations, and which OWASP 2025 risk covers the same problem?"* The top 5 were:

- NIST AI 600-1 pages 1 and 2 (title page and front matter);
- the OWASP title section (which has no page label);
- NIST AI 100-1 page 1;
- OWASP page 1.

These pages mention both document names, so they match a question that *names the documents* better than the passages about confabulation (page 8) and misinformation (page 36) do. hit@5 is 0, and the model correctly refused rather than guess. *Fixes:* query decomposition (one search per sub-question); down-weighting or excluding title and front-matter sections; or hybrid search, where "confabulation" and "misinformation" as keywords would pull the right pages up.

**q14: the page was found, but not the paragraph (a chunk-split effect).** *"What is an embedding inversion attack…?"* Page 33 is ranked 1st and 2nd, so the page-level metric counts it as a hit. The two retrieved chunks, however, cover the risk's *description* and items 1–2. Item 3, "Embedding Inversion Attacks", is in a later chunk of the same page that was not retrieved. The model said the chunks did not contain the answer. That is the right behaviour given what it saw, but it is a false refusal. This also shows a limit of page-level relevance labels: they overstate retrieval quality for long pages. *Fixes:* smaller chunks (config B is the test), or retrieving neighbouring chunks of a hit.

**q24: a partly answerable two-part question was refused entirely.** *"What is information integrity in the Generative AI Profile, and how does OWASP define misinformation?"*

- The OWASP definition (page 36) was retrieved first.
- NIST AI 600-1 page 14 was retrieved, but not the page-13 paragraph that actually defines information integrity.

With half the evidence, the v1 prompt's rule ("if the chunks do not contain the answer, set answerable to false") made the model refuse everything. *Fix applied:* prompt **v2** allows partial answers. It answers the covered part with citations and says which part the documents do not cover. Its effect will be measured in the next run.

**q06: two definitions on neighbouring pages were mixed up.** *"What are AI RMF use-case profiles and temporal profiles?"*

- The temporal-profile chunk (page 38, C1) was retrieved, and so was the *cross-sectoral* profile definition (page 39, C2).
- The chunk containing the use-case profile definition, the start of page 38's section, ranked below the top 5.

The model then described use-case profiles using the cross-sectoral text. The judge gave correctness 0 ("misstates use-case profiles as cross-sectoral") but marked the answer *faithful*. That is a judge error: the claim "use-case profiles are cross-sectoral profiles…" is not supported by C2, which is about a different profile type. *Fixes:* smaller chunks so each definition is its own chunk, and the judge reliability check in section 7.

**q22: incomplete multi-part answer (correctness 1).** Both sources were retrieved (NIST page 20, OWASP page 9), and the answer is faithful. It left out two points of the reference: graceful degradation, and the specific security concerns (adversarial examples, data poisoning, exfiltration). This is answer completeness, not retrieval.

**Patterns behind these failures.**

- **Ranking noise.** Table-of-contents pages and front matter repeat headings and titles, so they rank near the top. In q01 they took positions 4 and 5, and they decided q21.
- **The right page often comes second.** hit@1 is 0.583 against hit@5 0.958: a neighbouring page with similar vocabulary ranks first. A reranker would mostly improve hit@1 and MRR.
- **Refusals are conservative, not hallucinated.** No answer was judged unfaithful, raw citation validity was 100%, and every failure was a refusal or an incomplete answer, never an invented fact.

## 7. Judge reliability

- **Calibration sample.** The run wrote a 15-row sample (`evals/judge_calibration.csv`) with the question, answer, reference, cited passages and the judge's scores, plus empty `human_faithful` and `human_correctness` columns. `python -m evals.calibrate_judge` reports the agreement rates and lists every disagreement. **The hand labels have not been filled in yet**, so no agreement rate is claimed here.
- **One judge error is known (q06).** The judge marked as faithful an answer that attributed one profile type's definition to another. The correctness score caught the problem, but faithfulness did not. The judge is a 20B model from the same family as the answer model, so it may share blind spots.
- **Run-to-run variation.** Groq models do not guarantee identical output even at temperature 0, so individual answers and verdicts can change between runs. The retrieval metrics are deterministic for a fixed index and embedding model: re-running A's retrieval pass from the cache gave identical numbers.

## 8. Changes made because of this evaluation

| Change | Why |
|---|---|
| Off-topic floor 0.55, answer threshold 0.65 | Calibrated from section 4 (the defaults before were placeholders) |
| Prompt v2, rule 4: partial answers | q24 was refused entirely despite half the evidence |
| Prompt v2, rule 5: labelled general background for uncovered in-domain questions | In run A the model never added it, which the approved design requires. The notebook shows the result, and also its risk: the background figure for EU AI Act fines appears to come from the 2021 draft, which is why it is labelled and never cited |
| Gemini embedding adapter keeps the response order when `index` is null | Found on the first live run; the endpoint returns `index: null` |
| Embedding batches cached as they finish; evaluation saves results after each configuration | The free-tier quota ended the first runs partway through |

## 9. Limitations

- **Small set.** 33 questions, with only 3 off-topic and 4 multi-part, so one question moves a rate by 4–25 percentage points.
- **Page-level labels** overstate retrieval quality when a page holds several chunks (q14).
- **Thresholds were calibrated on the evaluation set itself.**
- **The judge is not yet validated against hand labels.**
- **Single run.** No repeated runs to measure answer variance.
- **Latency** was measured under free-tier throttling.
