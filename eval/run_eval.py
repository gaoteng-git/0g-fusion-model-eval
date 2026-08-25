"""Call the fusion API and a baseline API once per GPQA task (no tools, single
round each, thinking on for both), write a replay JSONL. Both calls go through
the same generic call_api() -- the eval code never touches fusion internals
directly. The judge inside the fusion pipeline is asked NOT to think
(reasoning_effort=none, handled inside mock_fusion_api); this script only
controls the two things it actually calls: the fusion endpoint and the
baseline endpoint, both with thinking on.
Run: python3 -m eval.run_eval [--limit N]
"""
import argparse
import json
import os
import time

from .gpqa_tasks import load_tasks
from .client import call_api


def run(fusion_url, fusion_model, baseline_url, baseline_model, out_path, limit=None, experiment=None):
    tasks = load_tasks(limit=limit)
    with open(out_path, "w", encoding="utf-8") as f:
        for task in tasks:
            messages = [{"role": "user", "content": task["instruction"]}]
            fusion_resp = call_api(fusion_url, fusion_model, messages, allow_tool_call_output=False,
                                    experiment=experiment, question_id=task["question_id"])
            baseline_resp = call_api(baseline_url, baseline_model, messages, reasoning_effort="high",
                                      experiment=experiment)
            row = {
                "schema": "0g.fusion_eval.gpqa.replay.v1",
                "question_id": task["question_id"],
                "instruction": task["instruction"],
                "correct_letter": task["correct_letter"],
                "fusion": {
                    "model": fusion_model,
                    "content": fusion_resp["choices"][0]["message"]["content"],
                    "reasoning_content": fusion_resp["choices"][0]["message"].get("reasoning_content"),
                    "raw_response": fusion_resp,
                },
                "baseline": {
                    "model": baseline_model,
                    "content": baseline_resp["choices"][0]["message"]["content"],
                    "reasoning_content": baseline_resp["choices"][0]["message"].get("reasoning_content"),
                    "raw_response": baseline_resp,
                },
                "config_id": f"gpqa-v1-{fusion_model}-vs-{baseline_model}",
            }
            f.write(json.dumps(row) + "\n")
    return out_path


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--fusion-url", default="http://localhost:8000")
    p.add_argument("--fusion-model", default="0g/fusion-preview")
    p.add_argument("--baseline-url", default="http://localhost:8000")
    p.add_argument("--baseline-model", default="baseline-model")
    p.add_argument("--out", default=None)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--experiment", default=None,
                    help="Experiment name used to name per-call log files "
                         "(call_logs/<experiment>__<role>__<model>.jsonl, see llm_client.py). "
                         "Defaults to <fusion-model>-vs-<baseline-model> if not given.")
    args = p.parse_args()
    out = args.out or os.path.join(os.path.dirname(__file__), "results", f"run_{int(time.time())}.jsonl")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    experiment = args.experiment or f"{args.fusion_model}-vs-{args.baseline_model}"
    print(run(args.fusion_url, args.fusion_model, args.baseline_url, args.baseline_model, out, args.limit,
               experiment))
