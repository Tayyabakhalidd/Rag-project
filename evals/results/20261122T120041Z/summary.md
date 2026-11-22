# Evaluation run 20261122T120041Z

- Git commit: `494ef3b (uncommitted changes)`
- Answer model: `openai/gpt-oss-120b`; judge model: `openai/gpt-oss-20b`
- Embedding model: `gemini-embedding-001` (768 dimensions)
- top_k = 5, off-topic floor = 0.55, answer threshold = 0.65
- Prompt version: v1; golden set: 33 questions

## Retrieval

| Config | hit@1 | hit@3 | hit@5 | recall@1 | recall@3 | recall@5 | MRR |
|---|---|---|---|---|---|---|---|
| A | 0.583 | 0.875 | 0.958 | 0.542 | 0.854 | 0.917 | 0.732 |

## Answers

| Metric | A |
|---|---|
| answered_rate_on_answerable | 0.875 |
| false_refusal_rate | 0.125 |
| faithfulness_rate | 1.000 |
| mean_correctness_0_2 | 1.625 |
| correctness_2_rate | 0.792 |
| unanswerable_handled_rate | 1.000 |
| off_topic_refusal_rate | 1.000 |
| off_topic_refused_without_llm | 1.000 |
| citation_validity_rate_raw | 1.000 |
| errors | 0 |
| latency_p50_ms | 9288.900 |
| latency_p95_ms | 12850.600 |
| judged_answers | 21 |
