"""Check how far the LLM judge agrees with human labels.

1. Run the evaluation; it writes ``judge_calibration_sample.csv`` in its results folder.
2. Copy that file to ``evals/judge_calibration.csv`` and fill in ``human_faithful``
   (true/false) and ``human_correctness`` (0, 1 or 2) for every row by reading
   the answer, the reference answer and the cited passages yourself.
3. Run ``python -m evals.calibrate_judge``. It prints the agreement rates and
   lists every disagreement, to be reviewed in EVAL.md.
"""

import argparse
import csv
import sys
from pathlib import Path

from app.config import PROJECT_ROOT

DEFAULT_PATH = PROJECT_ROOT / "evals" / "judge_calibration.csv"


def to_bool(value: str) -> bool | None:
    value = (value or "").strip().lower()
    if value in ("true", "1", "yes", "y", "pass"):
        return True
    if value in ("false", "0", "no", "n", "fail"):
        return False
    return None


def to_int(value: str) -> int | None:
    try:
        return int(str(value).strip())
    except ValueError:
        return None


def agreement(rows: list[dict]) -> dict:
    labelled = [r for r in rows if to_bool(r["human_faithful"]) is not None and to_int(r["human_correctness"]) is not None]
    if not labelled:
        return {"labelled": 0}
    faithful_agree = [to_bool(r["judge_faithful"]) == to_bool(r["human_faithful"]) for r in labelled]
    correct_agree = [to_int(r["judge_correctness"]) == to_int(r["human_correctness"]) for r in labelled]
    within_one = [abs((to_int(r["judge_correctness"]) or 0) - to_int(r["human_correctness"])) <= 1 for r in labelled]
    disagreements = [r for r, f, c in zip(labelled, faithful_agree, correct_agree) if not (f and c)]
    return {
        "labelled": len(labelled),
        "faithfulness_agreement": sum(faithful_agree) / len(labelled),
        "correctness_exact_agreement": sum(correct_agree) / len(labelled),
        "correctness_within_one": sum(within_one) / len(labelled),
        "disagreements": disagreements,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("path", nargs="?", type=Path, default=DEFAULT_PATH)
    args = parser.parse_args()
    if not args.path.exists():
        sys.exit(f"{args.path} not found. Copy judge_calibration_sample.csv there and add your labels.")
    with args.path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    result = agreement(rows)
    if not result["labelled"]:
        sys.exit("No rows have both human_faithful and human_correctness filled in yet.")
    print(f"Labelled rows: {result['labelled']}")
    print(f"Faithfulness agreement:        {result['faithfulness_agreement']:.0%}")
    print(f"Correctness exact agreement:   {result['correctness_exact_agreement']:.0%}")
    print(f"Correctness within one point:  {result['correctness_within_one']:.0%}")
    for row in result["disagreements"]:
        print(f"- {row['config']} {row['id']}: judge faithful={row['judge_faithful']} correctness={row['judge_correctness']}"
              f" | human faithful={row['human_faithful']} correctness={row['human_correctness']}")


if __name__ == "__main__":
    main()
