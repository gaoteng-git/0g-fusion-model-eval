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


def _score(rows, get_content):
    """`call_failed` (no usable response for this side) is counted separately
    from `extraction_failed` (the call succeeded but the model didn't follow
    the "Final Answer: X" format) -- different failure modes, and conflating
    them would hide which one is actually happening in a real run.

    `get_content(row)` returns a content string or None (meaning
    call_failed), decided by presence, NOT by any row-level "failed" flag:
    that flag can be scoped to a single baseline model while fusion (or a
    different baseline) succeeded in the same row -- keying off it would
    discard perfectly good answers and understate accuracy."""
    correct = extraction_failed = call_failed = total = 0
    for row in rows:
        total += 1
        content = get_content(row)
        if content is None:
            call_failed += 1
            continue
        letter = extract_final_letter(content)
        if letter is None:
            extraction_failed += 1
            continue
        if letter == row["correct_letter"]:
            correct += 1
    return {
        "accuracy": correct / total if total else 0.0,
        "correct": correct,
        "extraction_failed": extraction_failed,
        "call_failed": call_failed,
        "n": total,
    }


def _baseline_content(row, model):
    for b in row.get("baselines") or []:
        if b.get("model") == model and not b.get("failed"):
            return b.get("content")
    return None


def grade_replay(path):
    with open(path, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    fusion_score = _score(rows, lambda r: (r.get("fusion") or {}).get("content"))
    models = sorted({b.get("model") for r in rows for b in (r.get("baselines") or [])})
    baseline_scores = {model: _score(rows, lambda r, m=model: _baseline_content(r, m)) for model in models}
    return {"fusion": fusion_score, "baselines": baseline_scores}


if __name__ == "__main__":
    print(json.dumps(grade_replay(sys.argv[1]), indent=2))
