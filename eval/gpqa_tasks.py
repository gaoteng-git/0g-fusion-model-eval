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

SAMPLE_PATH = os.path.join(os.path.dirname(__file__), "data", "gpqa_sample.jsonl")
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
    path = path or SAMPLE_PATH
    rows = _read_rows(path)
    if limit:
        rows = rows[:limit]
    tasks = []
    for i, row in enumerate(rows):
        instruction, correct_letter = _format_question(row, i)
        tasks.append({"question_id": i, "instruction": instruction, "correct_letter": correct_letter})
    return tasks
