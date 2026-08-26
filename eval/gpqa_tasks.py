"""Load GPQA-style multiple-choice questions and format one into a prompt
(question + 4 shuffled options + the final-answer-letter format instruction).

Expects the official GPQA columns (gpqa_main.csv / the Diamond subset, from
https://github.com/idavidrein/gpqa -- gated, do not commit real questions to
a public repo, see its NOTICE): Question, Correct Answer, Incorrect Answer 1,
Incorrect Answer 2, Incorrect Answer 3. Accepts CSV or JSONL with those same
keys; raises a clear error if a required column is missing rather than
silently mis-scoring every question.

No default path, ever -- every caller passes an explicit --input. (An
earlier version of this eval harness silently switched from a made-up
sample file to a real, gated download the moment that file happened to
exist on disk one directory up; that is exactly the kind of surprise this
module now structurally cannot produce.)
"""
import csv
import json
import os
import random
import string

from mock_fusion_api.panel_config import FINAL_LETTER_INSTRUCTION

SAMPLE_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "eval", "data",
                            "gpqa_sample.jsonl")

REQUIRED_COLUMNS = ("Question", "Correct Answer", "Incorrect Answer 1", "Incorrect Answer 2", "Incorrect Answer 3")


def _read_rows(path):
    if path.endswith(".csv"):
        with open(path, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
    else:
        with open(path, encoding="utf-8") as f:
            rows = [json.loads(line) for line in f if line.strip()]
    for row in rows:
        missing = [c for c in REQUIRED_COLUMNS if c not in row]
        if missing:
            raise ValueError(f"GPQA row missing required column(s) {missing}: got keys {list(row)}")
    return rows


def load_questions(path):
    """-> [(question_id, row), ...], in file order.

    `question_id` is the row's own `question_id` field if present -- that's
    how eval.sample.py tags a row with its ORIGINAL absolute index when it
    extracts a subset into a new file, so results computed from different
    (possibly disjoint) sampled files stay keyed consistently and can be
    merged later. A row with no such field (the common case: reading a
    plain, un-sampled question file) falls back to its 0-based position in
    THIS file.

    Coerced to `int` -- a CSV `--input` with a `question_id` column would
    otherwise carry it as a string (`csv.DictReader` stringifies every
    field), and `int` vs the equivalent `str` are different dict keys and
    different random.Random() seeds, silently breaking every question_id
    match against a JSONL-sourced file and the shuffle's determinism."""
    rows = _read_rows(path)
    return [(int(row["question_id"]) if "question_id" in row else i, row) for i, row in enumerate(rows)]


def format_question(row, question_id):
    """-> (instruction, correct_letter). The A/B/C/D shuffle is seeded by
    `question_id`, not by the row's position in whatever file happens to be
    open -- the same question must shuffle the same way no matter which
    file (the full set, or some sampled subset of it) it's loaded from."""
    options = [row["Correct Answer"], row["Incorrect Answer 1"], row["Incorrect Answer 2"], row["Incorrect Answer 3"]]
    order = list(range(4))
    random.Random(question_id).shuffle(order)  # deterministic per-question, not global RNG state
    shuffled = [options[i] for i in order]
    correct_letter = string.ascii_uppercase[order.index(0)]
    lines = [row["Question"], ""]
    for letter, opt in zip(string.ascii_uppercase, shuffled):
        lines.append(f"{letter}) {opt}")
    lines.append("")
    lines.append(FINAL_LETTER_INSTRUCTION)
    return "\n".join(lines), correct_letter
