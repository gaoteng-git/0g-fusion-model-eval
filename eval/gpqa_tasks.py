"""Load GPQA-style multiple-choice tasks and format them as a single prompt
(question + 4 shuffled options + the final-answer-letter format instruction).

Expects the official GPQA columns (gpqa_main.csv / the Diamond subset, from
https://github.com/idavidrein/gpqa -- gated, do not commit real questions to
a public repo, see its NOTICE): Question, Correct Answer, Incorrect Answer 1,
Incorrect Answer 2, Incorrect Answer 3. Accepts CSV or JSONL with those same
keys; raises a clear error if a required column is missing rather than
silently mis-scoring every task.
"""
import csv
import json
import os
import random
import string

from mock_fusion_api.panel_config import FINAL_LETTER_INSTRUCTION

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # .../0g-fusion-model-eval
SAMPLE_PATH = os.path.join(_REPO_ROOT, "eval", "data", "gpqa_sample.jsonl")

# Real GPQA Diamond download (see download_gpqa_hf.py), expected one directory
# above this repo -- deliberately outside the repo tree so it can never be
# git-added by accident (see .gitignore's eval/data/* rule for the same reason).
REAL_DEFAULT_PATH = os.path.join(os.path.dirname(_REPO_ROOT), "gpqa_diamond.jsonl")

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


def _format_question(row, index):
    options = [row["Correct Answer"], row["Incorrect Answer 1"], row["Incorrect Answer 2"], row["Incorrect Answer 3"]]
    order = list(range(4))
    random.Random(index).shuffle(order)  # deterministic per-question shuffle, not global RNG state
    shuffled = [options[i] for i in order]
    correct_letter = string.ascii_uppercase[order.index(0)]
    lines = [row["Question"], ""]
    for letter, opt in zip(string.ascii_uppercase, shuffled):
        lines.append(f"{letter}) {opt}")
    lines.append("")
    lines.append(FINAL_LETTER_INSTRUCTION)
    return "\n".join(lines), correct_letter


def load_tasks(path=None, limit=None):
    if path is None:
        # Prefer the real downloaded dataset when it's present; otherwise fall
        # back to the made-up sample so tests / a fresh clone still work with
        # no setup. This means the default silently switches from placeholder
        # to real questions the moment gpqa_diamond.jsonl shows up one
        # directory above this repo -- worth knowing before assuming a run
        # used the sample set.
        path = REAL_DEFAULT_PATH if os.path.exists(REAL_DEFAULT_PATH) else SAMPLE_PATH
    rows = _read_rows(path)
    if limit:
        rows = rows[:limit]
    tasks = []
    for i, row in enumerate(rows):
        instruction, correct_letter = _format_question(row, i)
        tasks.append({"question_id": i, "instruction": instruction, "correct_letter": correct_letter})
    return tasks
