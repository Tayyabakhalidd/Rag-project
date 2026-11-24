# Evaluation

The original prompt-v1 baseline is saved in `evals/results/20261122T120041Z/`. The current A/B comparison uses two full prompt-v2 runs saved in `evals/results/20261124T132004Z/` and `evals/results/20261124T134115Z/`; each folder contains summaries, per-question CSVs, answers and judge verdicts.

> **Status (24 Nov 2026).** Configurations A (1000/150) and B (500/75) were both evaluated twice with prompt v2. Retrieval scores were identical across the two runs. B is selected as the service default because relevant passages ranked higher (MRR and hit@1/3) and it showed a small average answer-correctness edge. A retrieved more of the expected evidence in the top five (higher recall@5). The choice and trade-off are detailed in section 5. The 15-row judge calibration file is ready for human labels; no human-vs-judge agreement is claimed yet.

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

| Current runs | |
|---|---|
| Run IDs | `20261124T132004Z`, `20261124T134115Z` |
| Answer model | `openai/gpt-oss-120b` (Groq, `reasoning_effort=low`, temperature 0) |
| Judge model | `openai/gpt-oss-20b` (Groq) |
| Embeddings | `gemini-embedding-001`, 768 dimensions |
| Retrieval | top-k 5, cosine similarity, off-topic floor 0.55, answer threshold 0.65 |
| Prompt | v2 for both configurations |
| Code revision | `87454ca`; result files were generated locally after that commit |

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

## 5. Results and configuration choice

### Retrieval (24 answerable questions)

The retrieval pass is deterministic for this corpus, embedding model and index, so both runs produced the same metrics.

| Config | hit@1 | hit@3 | hit@5 | recall@1 | recall@3 | recall@5 | MRR |
|---|---|---|---|---|---|---|---|
| A (1000/150) | 0.583 | 0.875 | **0.958** | 0.542 | 0.854 | **0.917** | 0.732 |
| B (500/75) | **0.625** | **0.917** | 0.917 | **0.604** | **0.875** | 0.875 | **0.764** |

B more often puts a relevant chunk near the top. A more often includes all expected evidence somewhere in the first five. B also fixed three recurring A problems in both runs: q06 (confused definitions), q14 (missed the specific passage), and q24 (partly answered question). q21 remained a miss for both.

### Answer quality across two runs

Groq answers and judge ratings can vary between runs. Values below are the per-run rates and the simple mean of those two runs; this small sample is not a statistical guarantee.

| Metric | A run 1 | A run 2 | A mean | B run 1 | B run 2 | B mean |
|---|---:|---:|---:|---:|---:|---:|
| Answered rate on answerable questions | 0.917 | 0.917 | 0.917 | 0.917 | 0.917 | 0.917 |
| Faithfulness (judge) | 0.955 | 0.909 | 0.932 | 0.909 | 0.909 | 0.909 |
| Mean correctness (0–2) | 1.625 | 1.542 | 1.583 | 1.583 | 1.625 | **1.604** |
| Fully correct share | 0.750 | 0.667 | 0.708 | 0.708 | 0.750 | **0.729** |
| Unanswerable handled correctly | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| Off-topic refused before LLM call | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| Raw citation validity | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |

Both runs had zero API errors and judged 22 answers per configuration. Latency varied because the free-tier providers throttled calls; do not use these runs to claim a reliable speed advantage.

### Decision: use configuration B (500/75)

B is selected as the default because it gives the strongest first-result ranking (higher MRR, hit@1 and hit@3) and a small correctness advantage averaged over two answer runs. A is better if the priority is maximizing recall across all five retrieved chunks. This is a trade-off: B's recall@5 is 0.875 versus A's 0.917. Given the user experience and the repeated q06/q14/q24 improvements with B, the project prioritizes a more useful top-ranked passage and answer quality. The default is now `CHUNK_CONFIG=B`.

The set is small (24 answerable questions), and answer/judge results vary by run. Treat this as a reasoned project choice, not proof that B is universally better.

## 6. Failure analysis

For configuration A with prompt v2, five questions were failed or only partly right. The two runs show which issues the smaller config B chunks helped with; each case was traced through the saved per-question results.

**q21: retrieval fails on a broad two-document question (front matter wins in both configs).** *"What does NIST AI 600-1 call AI hallucinations, and which OWASP 2025 risk covers the same problem?"* The top 5 were:

- NIST AI 600-1 pages 1 and 2 (title page and front matter);
- the OWASP title section (which has no page label);
- NIST AI 100-1 page 1;
- OWASP page 1.

These pages mention both document names, so they match a question that *names the documents* better than the passages about confabulation (page 8) and misinformation (page 36) do. hit@5 is 0 for A and B, and both correctly refused rather than guess. *Fixes:* query decomposition (one search per sub-question); down-weighting or excluding title and front-matter sections; or hybrid search, where "confabulation" and "misinformation" as keywords would pull the right pages up.

**q14: the page was found, but not the paragraph (a chunk-split effect).** *"What is an embedding inversion attack…?"* Page 33 is ranked 1st and 2nd, so the page-level metric counts it as a hit. The two retrieved chunks, however, cover the risk's *description* and items 1–2. Item 3, "Embedding Inversion Attacks", is in a later chunk of the same page that was not retrieved. The model said the chunks did not contain the answer. That is the right behaviour given what it saw, but it is a false refusal. This also shows a limit of page-level relevance labels: they overstate retrieval quality for long pages. *Fixes:* smaller chunks (config B is the test), or retrieving neighbouring chunks of a hit.

**q24: a partly answerable two-part question exposed a prompt-v1 refusal.** *"What is information integrity in the Generative AI Profile, and how does OWASP define misinformation?"*

- The OWASP definition (page 36) was retrieved first.
- NIST AI 600-1 page 14 was retrieved, but not the page-13 paragraph that actually defines information integrity.

Prompt **v2** allows partial answers. In both prompt-v2 runs, config A gave a partial answer (correctness 1), while config B answered both parts (correctness 2).

**q06: two definitions on neighbouring pages were mixed up.** *"What are AI RMF use-case profiles and temporal profiles?"*

- The temporal-profile chunk (page 38, C1) was retrieved, and so was the *cross-sectoral* profile definition (page 39, C2).
- The chunk containing the use-case profile definition, the start of page 38's section, ranked below the top 5.

With config A, the model described use-case profiles using the cross-sectoral text. The judge gave correctness 0 but marked the answer *faithful*; that is the known judge error because the cited passage is about a different profile type. Config B answered q06 correctly (correctness 2) in both runs, consistent with its smaller chunks. The judge still needs human calibration (section 7).

**q22: incomplete multi-part answer (correctness 1).** Both sources were retrieved (NIST page 20, OWASP page 9), and the answer is faithful. It left out two points of the reference: graceful degradation, and the specific security concerns (adversarial examples, data poisoning, exfiltration). This is answer completeness, not retrieval.

**Patterns behind these failures.**

- **Ranking noise.** Table-of-contents pages and front matter repeat titles, so they can outrank the useful passage; this still affects q21.
- **B moves useful material up.** Across both runs, B had higher hit@1 and MRR and fixed q06, q14 and q24, but its lower recall@5 means it can miss some expected evidence.
- **Citation ID checks are not the same as faithfulness.** Raw citation validity was 100%, but the judge found some unsupported or incomplete claims; human calibration is still needed. A reranker or neighbouring-chunk expansion could be future improvements.

## 7. Judge reliability

- **Calibration sample.** The latest run produced a 15-row A/B sample, now copied to `evals/judge_calibration.csv`. It contains questions, answers, references, cited passages and judge scores, plus blank `human_faithful` and `human_correctness` columns. Fill both human columns for every row, then run `python -m evals.calibrate_judge`. **The labels are still blank**, so no judge agreement rate is claimed here.
- **One judge error is known (config A, q06).** The judge marked as faithful an answer that attributed one profile type's definition to another. The correctness score caught the problem, but faithfulness did not. The judge is a 20B model from the same family as the answer model, so it may share blind spots.
- **Run-to-run variation.** Groq models do not guarantee identical output even at temperature 0, so individual answers and verdicts can change between runs. The retrieval metrics are deterministic for a fixed index and embedding model: re-running A's retrieval pass from the cache gave identical numbers.

## 8. Changes made because of this evaluation

| Change | Why |
|---|---|
| Off-topic floor 0.55, answer threshold 0.65 | Calibrated from section 4 (the defaults before were placeholders) |
| Prompt v2, rule 4: partial answers | q24 was refused entirely despite half the evidence |
| Prompt v2, rule 5: labelled general background for uncovered in-domain questions | In the original run A the model never added it, which the approved design requires. The notebook shows the result, and also its risk: the background figure for EU AI Act fines appears to come from the 2021 draft, which is why it is labelled and never cited |
| Gemini embedding adapter keeps the response order when `index` is null | Found on the first live run; the endpoint returns `index: null` |
| Embedding batches cached as they finish; evaluation saves results after each configuration | The free-tier quota ended initial runs partway through; later complete runs and their outputs are preserved under `evals/results/` |

## 9. Limitations

- **Small set.** 33 questions, with only 3 off-topic and 4 multi-part, so one question moves a rate by 4–25 percentage points.
- **Page-level labels** overstate retrieval quality when a page holds several chunks (q14).
- **Thresholds were calibrated on the evaluation set itself.**
- **The judge is not yet validated against hand labels.**
- **Only two answer runs.** They show some run-to-run variation but are not enough to estimate a reliable distribution or confidence interval.
- **Latency** was measured under free-tier throttling.
