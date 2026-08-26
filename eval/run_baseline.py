"""Call ONE baseline model for each question in an existing replay file
(produced by run_eval.py or run_variant.py), WITHOUT re-calling fusion -- the
fusion side is copied through unchanged from the base row, and the new
model's answer is appended into the row's `baselines` list. Use this to add
another baseline (a 2nd, 3rd, ...) to an already-completed run without
re-paying for the fusion pipeline (run_eval.py's own --baseline-model already
takes a comma-separated list, but that only helps when you know all the
baselines you want up front, before the fusion side is computed).

Resume: if --out already has a row for a question (from an earlier
run_baseline.py call, possibly for a different model), that row's own
`baselines` list is the starting point -- so running this script twice with
two different --baseline-model values against the same --out accumulates
both, and pointing --out at --base-replay itself accumulates in place. If the
target model already has a non-failed entry in that list, nothing is called
for that question; if its entry is there but FAILED, it is retried and
replaced. Catch-and-continue matches run_eval.py/run_variant.py: a question
whose row has no fusion data to copy through, or whose call fails, is logged
and written without aborting the run.

Run:
  python3 -m eval.run_baseline --base-replay eval/results/gpqa-main.jsonl \\
      --baseline-model claude-fable-5 --experiment gpqa-baseline-fable5
"""
import argparse
import json
import sys

from .client import call_api
from .replay_io import (ResumeMismatchError, carry_over_unprocessed, default_out_path,  # noqa: F401
                        ensure_out_dir, load_existing)


def run(base_replay_path, baseline_url, baseline_model, out_path, limit=None, experiment=None, resume=True):
    ensure_out_dir(out_path)
    with open(base_replay_path, encoding="utf-8") as f:
        base_rows = [json.loads(line) for line in f if line.strip()]
    if limit:
        base_rows = base_rows[:limit]

    expected = {r.get("question_id"): (r.get("instruction"), r.get("correct_letter")) for r in base_rows}
    existing = load_existing(out_path, expected) if resume else {}

    skipped = failed = 0
    with open(out_path, "w", encoding="utf-8") as out_f:
        for base_row in base_rows:
            qid = base_row.get("question_id")
            # Whatever --out already has for this question (from an earlier
            # call, maybe for a different model) is the accumulation point;
            # otherwise start from --base-replay's own row.
            source_row = existing.get(qid, base_row)

            # Nothing to attach a baseline answer to: a row that failed
            # outright, or (a hand-built/foreign file) one with no fusion
            # side at all. Checked BEFORE the call so a row that can't be
            # written isn't paid for either.
            if source_row.get("failed") or "fusion" not in source_row:
                failed += 1
                print(f"eval.run_baseline_question_skipped question_id={qid!r} "
                      f"reason='no fusion data available for this question'", file=sys.stderr)
                out_f.write(json.dumps({**source_row, "baseline_model_requested": baseline_model}) + "\n")
                continue

            other_baselines = []
            already_have = None
            for b in source_row.get("baselines") or []:
                if b.get("model") != baseline_model:
                    other_baselines.append(b)
                elif not b.get("failed"):
                    already_have = b
            if already_have is not None:
                skipped += 1
                out_f.write(json.dumps(source_row) + "\n")
                continue

            messages = [{"role": "user", "content": base_row["instruction"]}]
            # The call AND the new-entry construction must stay inside this
            # one try block: indexing the response is itself a failure point,
            # so a 200-but-wrong-shape response would crash the whole run if
            # the entry were built outside it (same lesson as run_eval.py).
            try:
                baseline_resp = call_api(baseline_url, baseline_model, messages, reasoning_effort="high",
                                          experiment=experiment)
                new_entry = {
                    "model": baseline_model,
                    "content": baseline_resp["choices"][0]["message"]["content"],
                    "reasoning_content": baseline_resp["choices"][0]["message"].get("reasoning_content"),
                    "raw_response": baseline_resp,
                }
            except Exception as e:
                failed += 1
                print(f"eval.run_baseline_question_failed question_id={qid!r} error={str(e)!r}", file=sys.stderr)
                new_entry = {"model": baseline_model, "failed": True, "error": str(e)}

            baselines = other_baselines + [new_entry]
            out_f.write(json.dumps({
                "schema": "0g.fusion_eval.gpqa.replay.v1",
                "question_id": qid,
                "instruction": base_row["instruction"],
                "correct_letter": base_row["correct_letter"],
                "fusion": source_row["fusion"],  # unchanged -- not re-called
                "baselines": baselines,
                "config_id": f"gpqa-v1-{source_row['fusion']['model']}-vs-"
                              f"{'+'.join(b['model'] for b in baselines)}",
                "fusion_of": source_row.get("config_id") or base_row.get("config_id"),
            }) + "\n")
        # --limit only: without one, `expected` IS the whole dataset/base file and
        # anything else in --out is stale, not something to preserve.
        carried = carry_over_unprocessed(out_f, existing, expected) if limit else 0
    if skipped:
        print(f"eval.run_baseline_resumed skipped={skipped} ({baseline_model!r} already had a successful "
              f"result for those questions, reused as-is without calling anything)", file=sys.stderr)
    if carried:
        print(f"eval.run_baseline_carried_over kept={carried} prior row(s) for questions outside this "
              f"--limit window (they are not re-run, just not thrown away)", file=sys.stderr)
    if failed:
        print(f"eval.run_baseline_summary total={len(base_rows)} skipped={skipped} failed={failed}",
              file=sys.stderr)
    return out_path


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base-replay", required=True,
                    help="Replay JSONL from a prior run_eval.py/run_variant.py/run_baseline.py run -- any "
                         "file with a `fusion` field per row.")
    p.add_argument("--baseline-url", default="http://localhost:8000")
    p.add_argument("--baseline-model", required=True, help="The one new baseline model to add.")
    p.add_argument("--out", default=None,
                    help="Defaults to results/<experiment>.jsonl. Point this at the SAME file across "
                         "repeated runs (with different --baseline-model each time) to accumulate "
                         "multiple baselines into one file.")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--experiment", default=None, help="Defaults to baseline-<baseline-model>.")
    p.add_argument("--no-resume", action="store_true",
                    help="Ignore any existing rows at --out and overwrite from scratch.")
    args = p.parse_args()
    experiment = args.experiment or f"baseline-{args.baseline_model}"
    out = args.out or default_out_path(experiment)
    print(run(args.base_replay, args.baseline_url, args.baseline_model, out, args.limit, experiment,
               resume=not args.no_resume))
