"""Re-run judge+synthesis against a swapped-in panel member, reusing the
other, unchanged panel members' outputs from an earlier run_eval.py replay
file -- instead of paying to re-call the whole panel for every variant.

Reads a base replay file (produced by run_eval.py against a fixed panel,
e.g. the 4 "always-on" panel members), and for each question:
  1. pulls that question's already-computed panel entries out of
     fusion.raw_response["0g_fusion"]["panel"] (run_eval.py already stores
     the full fusion response there, panel breakdown included -- nothing
     extra needs to be captured/changed in run_eval.py for this to work),
  2. sends them back as `cached_panel`, plus the new candidate model as
     `extra_panel_models`, so the fusion pipeline only actually calls the
     new model, then re-runs judge+synthesis on the merged 5-member panel.

The baseline side isn't re-run here -- it didn't change between variants,
so the base replay's baseline row is carried over unchanged into each
variant's output row (grade_replay only needs baseline.content +
correct_letter, both present as-is).

Run:
  python3 -m eval.run_variant --base-replay eval/results/run_BASE.jsonl \\
      --variant-model xiaomi/mimo-v2.5-pro --out eval/results/variant_mimo.jsonl
"""
import argparse
import json
import os
import time

from .client import call_api


def run(base_replay_path, fusion_url, fusion_model, variant_model, out_path, limit=None, experiment=None):
    with open(base_replay_path, encoding="utf-8") as f:
        base_rows = [json.loads(line) for line in f if line.strip()]
    if limit:
        base_rows = base_rows[:limit]

    with open(out_path, "w", encoding="utf-8") as out_f:
        for base_row in base_rows:
            cached_panel = base_row["fusion"]["raw_response"]["0g_fusion"]["panel"]
            messages = [{"role": "user", "content": base_row["instruction"]}]
            fusion_resp = call_api(
                fusion_url, fusion_model, messages,
                cached_panel=cached_panel, extra_panel_models=[variant_model],
                experiment=experiment,
            )
            row = {
                "schema": "0g.fusion_eval.gpqa.replay.v1",
                "question_id": base_row["question_id"],
                "instruction": base_row["instruction"],
                "correct_letter": base_row["correct_letter"],
                "fusion": {
                    "model": fusion_model,
                    "content": fusion_resp["choices"][0]["message"]["content"],
                    "reasoning_content": fusion_resp["choices"][0]["message"].get("reasoning_content"),
                    "raw_response": fusion_resp,
                },
                "baseline": base_row["baseline"],  # unchanged from the base run -- not re-called
                "config_id": f"gpqa-v1-variant-{variant_model}-on-{base_row['config_id']}",
                "variant_of": base_row["config_id"],
                "cached_panel_models": [p["model"] for p in cached_panel],
                "variant_model": variant_model,
            }
            out_f.write(json.dumps(row) + "\n")
    return out_path


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base-replay", required=True, help="Replay JSONL from a prior run_eval.py run.")
    p.add_argument("--fusion-url", default="http://localhost:8000")
    p.add_argument("--fusion-model", default="0g/fusion-preview")
    p.add_argument("--variant-model", required=True, help="The one new panel model to actually call.")
    p.add_argument("--out", default=None)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--experiment", default=None,
                    help="Defaults to variant-<variant-model>.")
    args = p.parse_args()
    out = args.out or os.path.join(os.path.dirname(__file__), "results", f"variant_{int(time.time())}.jsonl")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    experiment = args.experiment or f"variant-{args.variant_model}"
    print(run(args.base_replay, args.fusion_url, args.fusion_model, args.variant_model, out, args.limit, experiment))
