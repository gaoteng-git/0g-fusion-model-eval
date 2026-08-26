"""Compute ONE panel model's answers over a question file. --model is always
exactly one model -- to build a 5-member panel, run this 5 times (once per
model, each with its own --out); nothing here decides panel composition,
that's eval.fuse's job once it's given the resulting files.

No resume, no --limit, no notion of "already have some of this": every run
computes every question in --input, fresh, and OVERWRITES --out completely.
Want to compute only a few questions (to debug, or to add more later without
re-paying for what's already done)? Make a smaller/different --input with
eval.sample.py first -- that tool, not this one, is where "which questions"
gets decided.

Run:
  python3 -m eval.panel --model minimax-m3 --api-url http://localhost:8000 \\
      --input eval/samples/first5.jsonl --out eval/results/panel-minimax-m3.jsonl
"""
import argparse
import json
import os
import sys

from .gpqa_tasks import load_questions, format_question
from .client import call_api


def run(api_url, model, input_path, out_path, experiment=None):
    questions = load_questions(input_path)
    # eval/results/ is gitignored, so it doesn't exist on a fresh clone -- a
    # bare open() would fail before writing anything.
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for question_id, row in questions:
            instruction, correct_letter = format_question(row, question_id)
            base = {"question_id": question_id, "instruction": instruction, "correct_letter": correct_letter,
                    "model": model}
            messages = [{"role": "user", "content": instruction}]
            # The call AND the row construction must stay inside this one try
            # block: indexing the response is itself a failure point, so a
            # 200-but-wrong-shape response would crash the whole run if the
            # row were built outside it.
            try:
                resp = call_api(api_url, model, messages, reasoning_effort="high", experiment=experiment,
                                 role="panel")
                message = resp["choices"][0]["message"]
                row_out = {**base, "content": message["content"], "reasoning": message.get("reasoning_content")}
            except Exception as e:
                print(f"eval.panel_question_failed question_id={question_id!r} model={model!r} "
                      f"error={str(e)!r}", file=sys.stderr)
                row_out = {**base, "failed": True, "error": str(e)}
            f.write(json.dumps(row_out) + "\n")
    return out_path


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True, help="Exactly one panel model.")
    p.add_argument("--api-url", default="http://localhost:8000")
    p.add_argument("--input", required=True, help="A question file (see eval.sample.py).")
    p.add_argument("--out", required=True, help="Always overwritten.")
    p.add_argument("--experiment", default=None, help="Names call_logs/<experiment>__panel__<model>.jsonl.")
    args = p.parse_args()
    print(run(args.api_url, args.model, args.input, args.out, args.experiment))
