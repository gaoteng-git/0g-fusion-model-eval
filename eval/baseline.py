"""Compute 1+ baseline model answers for GPQA questions -- completely
independent of panel/fusion (unlike eval.panel/eval.fuse, this needs no
input file at all, just the question set). Accumulates into a row's
`baselines` list across repeated calls targeting the same --out (add model
2, then model 3, then retry a previously-failed one) rather than replacing
it -- --models only names what THIS call should end up having present; it
never removes an already-present different model.

Run:
  python3 -m eval.baseline --baseline-url http://localhost:8000 \\
      --models gpt-5.6-sol,claude-fable-5 --experiment gpqa-baselines
"""
import argparse
import json
import sys

from .gpqa_tasks import load_tasks
from .client import call_api
from .replay_io import (ResumeMismatchError, carry_over_unprocessed, default_out_path,  # noqa: F401
                         ensure_out_dir, load_existing)

SCHEMA = "0g.fusion_eval.gpqa.baselines.v1"


def _base_row(qid, task):
    return {"schema": SCHEMA, "question_id": qid, "instruction": task["instruction"],
            "correct_letter": task["correct_letter"]}


def run(baseline_url, models, out_path, limit=None, experiment=None, resume=True):
    ensure_out_dir(out_path)
    tasks = load_tasks(limit=limit)
    expected = {t["question_id"]: (t["instruction"], t["correct_letter"]) for t in tasks}
    existing = load_existing(out_path, expected, expected_schema=SCHEMA) if resume else {}

    skipped = failed = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for task in tasks:
            qid = task["question_id"]
            prior = existing.get(qid) or {**_base_row(qid, task), "baselines": []}
            prior_baselines = prior.get("baselines") or []
            have = {b["model"] for b in prior_baselines if not b.get("failed")}
            need = [m for m in models if m not in have]
            if not need:
                skipped += 1
                f.write(json.dumps(prior) + "\n")
                continue

            keep = [b for b in prior_baselines if b["model"] not in need]
            messages = [{"role": "user", "content": task["instruction"]}]
            new_entries = []
            for bm in need:
                # The call AND the entry construction must stay inside this
                # one try block: indexing the response is itself a failure
                # point, so a 200-but-wrong-shape response would crash the
                # whole run if the entry were built outside it.
                try:
                    resp = call_api(baseline_url, bm, messages, reasoning_effort="high", experiment=experiment)
                    new_entries.append({
                        "model": bm,
                        "content": resp["choices"][0]["message"]["content"],
                        "reasoning_content": resp["choices"][0]["message"].get("reasoning_content"),
                        "raw_response": resp,
                    })
                except Exception as e:
                    failed += 1
                    print(f"eval.baseline_question_failed question_id={qid!r} model={bm!r} error={str(e)!r}",
                          file=sys.stderr)
                    new_entries.append({"model": bm, "failed": True, "error": str(e)})

            row = {**_base_row(qid, task), "baselines": keep + new_entries}
            f.write(json.dumps(row) + "\n")
        if limit:
            skipped += carry_over_unprocessed(f, existing, expected)
    if skipped:
        print(f"eval.baseline_resumed skipped={skipped} (already had every requested model for these "
              f"questions at {out_path!r})", file=sys.stderr)
    if failed:
        print(f"eval.baseline_summary total={len(tasks)} skipped={skipped} failed={failed}", file=sys.stderr)
    return out_path


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--baseline-url", default="http://localhost:8000")
    p.add_argument("--models", required=True, help="Comma-separated baseline model(s) to have present.")
    p.add_argument("--out", default=None, help="Defaults to results/<experiment>.jsonl.")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--experiment", default=None, help="Defaults to baselines-<models joined by '+'>.")
    p.add_argument("--no-resume", action="store_true", help="Ignore any existing rows at --out.")
    args = p.parse_args()
    models = list(dict.fromkeys(m.strip() for m in args.models.split(",") if m.strip()))
    if not models:
        sys.exit("--models must name at least one model")
    experiment = args.experiment or f"baselines-{'+'.join(models)}"
    out = args.out or default_out_path(experiment)
    print(run(args.baseline_url, models, out, args.limit, experiment, resume=not args.no_resume))
