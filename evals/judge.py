"""LLM-as-a-judge for answer quality.

The judge is a *different* Groq model from the one that writes answers (set by
``JUDGE_MODEL``), so a model is not grading its own output. It scores two
things per answered question:

* faithful (pass/fail): every claim in the answer is supported by the passages
  the answer cites. General-knowledge text is not shown to the judge, because
  it is labelled as not coming from the documents.
* correctness (0-2) against the golden reference answer:
  2 = all key points right, 1 = partly right or incomplete, 0 = wrong or missing.

How far the judge can be trusted is checked against hand labels
(``evals/calibrate_judge.py``).
"""

import json

from pydantic import BaseModel, Field

from app.errors import UpstreamUnavailableError

JUDGE_SYSTEM = """You grade answers produced by a document question-answering system. Be strict and consistent.

You get: a QUESTION, a REFERENCE answer written by a human, the system's ANSWER, and the CITED PASSAGES the answer relied on.

1. faithful: true only if every factual claim in ANSWER is supported by the CITED PASSAGES. If any claim is missing from or contradicts the passages, false. Citation markers like [C1] are not claims.
2. correctness: compare ANSWER with REFERENCE.
   2 = contains all key points of the reference and nothing that contradicts it.
   1 = partly correct: some key points missing, or minor errors.
   0 = wrong, contradicts the reference, or does not answer.

Reply with one JSON object only: {"faithful": true, "correctness": 2, "reason": "one short sentence"}"""


class JudgeVerdict(BaseModel):
    faithful: bool
    correctness: int = Field(ge=0, le=2)
    reason: str = ""


def judge_messages(question: str, reference: str, answer: str, passages: list[dict]) -> list[dict[str, str]]:
    cited = "\n\n".join(
        f"[{p['citation_id']}] ({p['filename']}, page {p.get('page')})\n{p['text']}" for p in passages
    ) or "(no passages)"
    user = (
        f"QUESTION:\n{question}\n\nREFERENCE:\n{reference}\n\nANSWER:\n{answer}\n\n"
        f"CITED PASSAGES:\n{cited}\n\nReply with the JSON object."
    )
    return [{"role": "system", "content": JUDGE_SYSTEM}, {"role": "user", "content": user}]


def parse_verdict(raw: str) -> JudgeVerdict:
    start, end = raw.find("{"), raw.rfind("}")
    try:
        return JudgeVerdict.model_validate(json.loads(raw[start : end + 1]))
    except Exception as exc:  # noqa: BLE001 - any malformed reply is treated the same way
        raise UpstreamUnavailableError("The judge returned an unreadable verdict.") from exc
