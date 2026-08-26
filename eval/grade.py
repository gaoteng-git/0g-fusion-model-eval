"""GPQA grading: exact-match on the extracted final letter. No LLM judge --
GPQA has a ground-truth answer, unlike AlpacaEval/Arena-Hard's subjective
pairwise comparison (see md_files/... benchmark-choice discussion for why
this is the whole point of using GPQA).

Takes 1+ files and merges them by question_id before scoring, so a fusion
file (from eval.fuse, has `fusion`) and a baseline file (from eval.baseline,
has `baselines`) can be graded together without ever having been combined
into one file on disk.

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


def _score(rows, get_content):
    """`call_failed` (no usable response for this side) is counted separately
    from `extraction_failed` (the call succeeded but the model didn't follow
    the "Final Answer: X" format) -- different failure modes, and conflating
    them would hide which one is actually happening in a real run.

    `get_content(row)` returns a content string or None (meaning
    call_failed), decided by presence, NOT by any row-level "failed" flag:
    that flag can be scoped to a single baseline model while fusion (or a
    different baseline) succeeded in the same row -- keying off it would
    discard perfectly good answers and understate accuracy.

    `no_ground_truth` (a row with no `correct_letter` at all -- only
    possible via a malformed/hand-edited replay file, never eval.panel's own
    output) is counted separately too, rather than falling through to a
    `letter == None` comparison that's always False: that would silently
    count a good answer as WRONG instead of flagging that this row simply
    can't be graded."""
    correct = extraction_failed = call_failed = no_ground_truth = total = 0
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
        "n": total,
    }


def _baseline_content(row, model):
    for b in row.get("baselines") or []:
        if b.get("model") == model and not b.get("failed"):
            return b.get("content")
    return None


class GradeMergeError(Exception):
    """Two files being merged disagree about what question_id N even IS."""


def load_rows(paths):
    """Merge N files by question_id -- e.g. a fusion file and a baseline
    file for the same questions combine into one logical row per
    question_id (later files' fields take precedence on a literal key
    collision, but `fusion`/`baselines` normally come from different files
    so there's nothing to collide).

    question_id is only a positional index into whatever dataset produced
    it (see replay_io.py's ResumeMismatchError for the same concern within
    one file) -- two files built from different question sets could each
    have a "question_id 0" that means something else entirely. Merging
    them would silently grade one file's answer against the other's
    ground truth, with the result depending on argument ORDER (whichever
    file's `instruction`/`correct_letter` happens to be seen last wins).
    Refuse instead, whenever both sides actually have an instruction to
    compare (a row-level failure -- {"failed": true} with no `instruction`
    at all -- carries no ground truth of its own to conflict with)."""
    merged = {}
    for path in paths:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                qid = row.get("question_id")
                prior = merged.get(qid)
                if prior and prior.get("instruction") is not None and row.get("instruction") is not None:
                    prior_key = (prior["instruction"], prior.get("correct_letter"))
                    row_key = (row["instruction"], row.get("correct_letter"))
                    if prior_key != row_key:
                        raise GradeMergeError(
                            f"question_id={qid!r} means a DIFFERENT question in {path!r} than in an "
                            f"earlier file being merged -- correct_letter {prior_key[1]!r} vs "
                            f"{row_key[1]!r}. Merging them would grade one file's answer against the "
                            f"other's ground truth, and the result would depend on argument order. "
                            f"These files were not built from the same question set; don't grade them "
                            f"together."
                        )
                if (prior and prior.get("fusion") and row.get("fusion")
                        and prior.get("config_id") and row.get("config_id")
                        and prior["config_id"] != row["config_id"]):
                    # Two DIFFERENT fusion results for the same question_id --
                    # e.g. two variant fuse files glued together (`eval.grade
                    # gpqa-fuse-*.jsonl`) instead of graded one at a time.
                    # {**prior, **row} would let the later file's `fusion`
                    # silently win with no error, blending variants into one
                    # "fusion" score that's a per-question mixture of both.
                    raise GradeMergeError(
                        f"question_id={qid!r} has a DIFFERENT fusion result in {path!r} than in an "
                        f"earlier file being merged -- config_id {prior['config_id']!r} vs "
                        f"{row['config_id']!r}. These are two different fusion configs (e.g. different "
                        f"panel variants); grade them separately, not merged into one score."
                    )
                merged[qid] = {**(prior or {}), **row}
    return list(merged.values())


def grade_replay(*paths):
    rows = load_rows(paths)
    fusion_score = _score(rows, lambda r: (r.get("fusion") or {}).get("content"))
    models = sorted({b.get("model") for r in rows for b in (r.get("baselines") or [])})
    baseline_scores = {model: _score(rows, lambda r, m=model: _baseline_content(r, m)) for model in models}
    return {"fusion": fusion_score, "baselines": baseline_scores}


if __name__ == "__main__":
    print(json.dumps(grade_replay(*sys.argv[1:]), indent=2))
