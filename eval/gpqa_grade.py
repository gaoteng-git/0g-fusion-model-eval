"""GPQA grading: exact-match on the extracted final letter. No LLM judge --
GPQA has a ground-truth answer, unlike AlpacaEval/Arena-Hard's subjective
pairwise comparison (see md_files/... benchmark-choice discussion for why
this is the whole point of using GPQA).
Run: python3 -m eval.gpqa_grade <replay.jsonl>
"""
import json
import re
import sys

_FINAL_ANSWER_RE = re.compile(r"Final Answer:\s*\**\s*([A-D])\b", re.IGNORECASE)


def extract_final_letter(content):
    """Returns the extracted letter (uppercase) or None if the required
    "Final Answer: X" line is missing -- e.g. the model didn't follow the
    format instruction, or (MiniMax-M3-style) a <think> block swallowed it.
    Takes the LAST match so a stray earlier mention doesn't win."""
    matches = _FINAL_ANSWER_RE.findall(content or "")
    return matches[-1].upper() if matches else None


def _score(rows, key):
    correct = extraction_failed = total = 0
    for row in rows:
        total += 1
        letter = extract_final_letter(row[key]["content"])
        if letter is None:
            extraction_failed += 1
            continue
        if letter == row["correct_letter"]:
            correct += 1
    return {
        "accuracy": correct / total if total else 0.0,
        "correct": correct,
        "extraction_failed": extraction_failed,
        "n": total,
    }


def grade_replay(path):
    with open(path, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f]
    return {"fusion": _score(rows, "fusion"), "baseline": _score(rows, "baseline")}


if __name__ == "__main__":
    print(json.dumps(grade_replay(sys.argv[1]), indent=2))
