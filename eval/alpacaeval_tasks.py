"""Load AlpacaEval-style tasks. Accepts either the official alpaca_eval.json export
(a list of {"instruction", "dataset", "generator"} objects) or the bundled offline
sample (same field names, one JSON object per line)."""
import json
import os

SAMPLE_PATH = os.path.join(os.path.dirname(__file__), "data", "alpaca_eval_sample.jsonl")


def load_tasks(path=None, limit=None):
    path = path or SAMPLE_PATH
    with open(path, encoding="utf-8") as f:
        if path.endswith(".json"):
            tasks = json.load(f)
        else:
            tasks = [json.loads(line) for line in f if line.strip()]
    return tasks[:limit] if limit else tasks
