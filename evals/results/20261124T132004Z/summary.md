# Evaluation run 20261124T132004Z

- Git commit: `87454ca`
- Answer model: `openai/gpt-oss-120b`; judge model: `openai/gpt-oss-20b`
- Embedding model: `gemini-embedding-001` (768 dimensions)
- top_k = 5, off-topic floor = 0.55, answer threshold = 0.65
- Prompt version: v2; golden set: 33 questions

## Retrieval

| Config | hit@1 | hit@3 | hit@5 | recall@1 | recall@3 | recall@5 | MRR |
|---|---|---|---|---|---|---|---|
| A | 0.583 | 0.875 | 0.958 | 0.542 | 0.854 | 0.917 | 0.732 |
| B | 0.625 | 0.917 | 0.917 | 0.604 | 0.875 | 0.875 | 0.764 |

## Answers

| Metric | A | B |
|---|---|---|
| answered_rate_on_answerable | 0.917 | 0.917 |
| false_refusal_rate | 0.083 | 0.083 |
| faithfulness_rate | 0.955 | 0.909 |
| mean_correctness_0_2 | 1.625 | 1.583 |
| correctness_2_rate | 0.750 | 0.708 |
| unanswerable_handled_rate | 1.000 | 1.000 |
| off_topic_refusal_rate | 1.000 | 1.000 |
| off_topic_refused_without_llm | 1.000 | 1.000 |
| citation_validity_rate_raw | 1.000 | 1.000 |
| errors | 0 | 0 |
| latency_p50_ms | 9833.500 | 3229.500 |
| latency_p95_ms | 12971.400 | 16411.100 |
| judged_answers | 22 | 22 |
