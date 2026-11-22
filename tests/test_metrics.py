"""Unit tests for the evaluation metrics, the golden set and the judge helpers."""

import json

import pytest

from app.config import CORPUS_DIR, PROJECT_ROOT
from app.errors import UpstreamUnavailableError
from app.guardrails import check_question
from evals.calibrate_judge import agreement
from evals.judge import parse_verdict
from evals.metrics import hit_at_k, per_question_retrieval, recall_at_k, reciprocal_rank, retrieval_summary
from evals.run_evaluation import answer_metrics, percentile

A = {"filename": "a.pdf", "pages": [3]}
B = {"filename": "b.docx", "pages": [7, 8]}
RETRIEVED = [
    {"filename": "a.pdf", "page": 1},
    {"filename": "b.docx", "page": 8},
    {"filename": "a.pdf", "page": 3},
]


def test_hit_at_k():
    assert hit_at_k(RETRIEVED, [A], 1) == 0.0
    assert hit_at_k(RETRIEVED, [A], 3) == 1.0
    assert hit_at_k(RETRIEVED, [B], 2) == 1.0  # page 8 is one of the accepted pages


def test_recall_at_k_counts_each_expected_source_once():
    assert recall_at_k(RETRIEVED, [A, B], 1) == 0.0
    assert recall_at_k(RETRIEVED, [A, B], 2) == 0.5
    assert recall_at_k(RETRIEVED, [A, B], 3) == 1.0
    assert recall_at_k(RETRIEVED, [], 3) == 0.0


def test_reciprocal_rank():
    assert reciprocal_rank(RETRIEVED, [A]) == pytest.approx(1 / 3)
    assert reciprocal_rank(RETRIEVED, [A, B]) == 0.5
    assert reciprocal_rank(RETRIEVED, [{"filename": "c.pdf", "pages": [1]}]) == 0.0


def test_same_filename_on_wrong_page_is_not_relevant():
    assert hit_at_k([{"filename": "a.pdf", "page": 4}], [A], 1) == 0.0


def test_retrieval_summary_averages_questions():
    rows = [per_question_retrieval(RETRIEVED, [A]), per_question_retrieval(RETRIEVED, [B])]
    summary = retrieval_summary(rows)
    assert summary["hit@1"] == 0.0 and summary["hit@3"] == 1.0
    assert summary["mrr"] == pytest.approx((1 / 3 + 1 / 2) / 2)
    assert summary["questions"] == 2


def test_answer_metrics_on_hand_made_records():
    answers = [
        {"id": "1", "type": "single", "golden_answerable": True, "answerable": True, "citations": [{}],
         "raw_citation_count": 2, "invalid_citation_count": 1, "latency_ms": 100.0, "llm_called": True},
        {"id": "2", "type": "single", "golden_answerable": True, "answerable": False,
         "refusal_reason": "not_in_documents", "citations": [], "latency_ms": 300.0, "llm_called": True},
        {"id": "3", "type": "unanswerable", "golden_answerable": False, "answerable": False,
         "refusal_reason": "not_in_documents", "citations": [], "latency_ms": 200.0, "llm_called": True},
        {"id": "4", "type": "off_topic", "golden_answerable": False, "answerable": False,
         "refusal_reason": "off_topic", "citations": [], "latency_ms": 50.0, "llm_called": False},
    ]
    verdicts = [{"id": "1", "faithful": True, "correctness": 2, "judged": True},
                {"id": "2", "faithful": None, "correctness": 0, "judged": False}]
    m = answer_metrics(answers, verdicts)
    assert m["answered_rate_on_answerable"] == 0.5 and m["false_refusal_rate"] == 0.5
    assert m["faithfulness_rate"] == 1.0 and m["mean_correctness_0_2"] == 1.0
    assert m["unanswerable_handled_rate"] == 1.0
    assert m["off_topic_refusal_rate"] == 1.0 and m["off_topic_refused_without_llm"] == 1.0
    assert m["citation_validity_rate_raw"] == 0.5
    assert m["latency_p50_ms"] in (100.0, 200.0)


def test_percentile():
    assert percentile([], 50) is None
    assert percentile([1, 2, 3, 4, 5], 50) == 3
    assert percentile([1, 2, 3, 4, 5], 95) == 5


def test_judge_verdict_parsing():
    verdict = parse_verdict('Sure: {"faithful": false, "correctness": 1, "reason": "partly"}')
    assert verdict.faithful is False and verdict.correctness == 1
    with pytest.raises(UpstreamUnavailableError):
        parse_verdict('{"faithful": true, "correctness": 7}')


def test_calibration_agreement():
    rows = [
        {"config": "A", "id": "q1", "judge_faithful": "True", "judge_correctness": "2", "human_faithful": "true", "human_correctness": "2"},
        {"config": "A", "id": "q2", "judge_faithful": "True", "judge_correctness": "2", "human_faithful": "false", "human_correctness": "1"},
        {"config": "B", "id": "q3", "judge_faithful": "False", "judge_correctness": "0", "human_faithful": "", "human_correctness": ""},
    ]
    result = agreement(rows)
    assert result["labelled"] == 2
    assert result["faithfulness_agreement"] == 0.5
    assert result["correctness_within_one"] == 1.0
    assert [r["id"] for r in result["disagreements"]] == ["q2"]


# --- The golden set itself --------------------------------------------------

GOLDEN = json.loads((PROJECT_ROOT / "data" / "golden_questions.json").read_text(encoding="utf-8"))["questions"]


def test_golden_set_meets_the_brief():
    assert len(GOLDEN) >= 25
    assert len({q["id"] for q in GOLDEN}) == len(GOLDEN)
    types = [q["type"] for q in GOLDEN]
    assert types.count("unanswerable") >= 5
    assert types.count("reworded") >= 3 and types.count("multi_part") >= 3
    assert types.count("off_topic") >= 2


def test_golden_sources_point_to_real_corpus_files():
    corpus = {p.name for p in CORPUS_DIR.iterdir()}
    for q in GOLDEN:
        assert q["answerable"] == bool(q["expected_sources"]), q["id"]
        assert q["reference_answer"], q["id"]
        for source in q["expected_sources"]:
            assert source["filename"] in corpus, q["id"]
            assert source["pages"] and all(isinstance(p, int) and p > 0 for p in source["pages"]), q["id"]


def test_no_golden_question_is_blocked_by_the_input_guardrail():
    assert [q["id"] for q in GOLDEN if check_question(q["question"]).blocked] == []
