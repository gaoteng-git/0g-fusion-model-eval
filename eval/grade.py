"""GPQA grading: exact-match on the extracted final letter. No LLM judge --
GPQA has a ground-truth answer, unlike AlpacaEval/Arena-Hard's subjective
pairwise comparison.

Every file eval.panel/eval.fuse/eval.baseline produces already carries its
own `question_id`/`correct_letter`/`content` per row -- a single, self-
sufficient, scoreable unit. So grading N files means scoring each one
independently and reporting them side by side; nothing here ever merges
rows across files, so there is nothing to disagree about between them.

Run: python3 -m eval.grade <file.jsonl> [<file2.jsonl> ...]
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


def grade_file(path):
    """`call_failed` (no usable response at all) is counted separately from
    `extraction_failed` (the call succeeded but the model didn't follow the
    "Final Answer: X" format), and both separately from `no_ground_truth`
    (the row itself has no correct_letter to compare against -- only
    possible via a hand-edited file) -- three different failure modes,
    conflating them would hide which one is actually happening in a real
    run.

    Rows are deduplicated by question_id first (last one in the file wins),
    so `n` always reflects DISTINCT questions -- a file produced by
    concatenating two overlapping (not disjoint) eval.sample.py batches would
    otherwise silently double-count whichever question_ids appear in both,
    inflating `n` and skewing `accuracy` with no error at all. Any duplicates
    found are reported in `duplicate_question_ids` and printed to stderr, so
    an overlapping merge is visible instead of silently wrong."""
    by_id = {}
    occurrences = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            qid = row.get("question_id")
            occurrences[qid] = occurrences.get(qid, 0) + 1
            by_id[qid] = row  # last occurrence wins

    duplicate_ids = sorted(qid for qid, count in occurrences.items() if count > 1)
    if duplicate_ids:
        print(f"eval.grade_duplicate_question_ids path={path!r} ids={duplicate_ids!r} -- probably an "
              f"overlapping (not disjoint) merge; only the last occurrence of each was scored",
              file=sys.stderr)

    correct = extraction_failed = call_failed = no_ground_truth = total = 0
    for qid in by_id:
        row = by_id[qid]
        total += 1
        if row.get("failed") or not row.get("content"):
            call_failed += 1
            continue
        letter = extract_final_letter(row["content"])
        if letter is None:
            extraction_failed += 1
            continue
        if row.get("correct_letter") is None:
            no_ground_truth += 1
            continue
        if letter == row["correct_letter"]:
            correct += 1
    return {
        "accuracy": correct / total if total else 0.0,
        "correct": correct,
        "extraction_failed": extraction_failed,
        "call_failed": call_failed,
        "no_ground_truth": no_ground_truth,
        "duplicate_question_ids": len(duplicate_ids),
        "n": total,
    }


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("usage: python3 -m eval.grade <file.jsonl> [<file2.jsonl> ...]")
    results = {}
    for path in sys.argv[1:]:
        try:
            results[path] = grade_file(path)
        except Exception as e:
            # Each file is graded independently -- one corrupt/unreadable
            # file must not lose the score for every OTHER file on the same
            # command line.
            results[path] = {"error": str(e)}
    print(json.dumps(results, indent=2))
