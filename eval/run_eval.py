"""Call the fusion API once and 0+ baseline models once each per GPQA task (no
tools, single round each, thinking on for both), write a replay JSONL. All
calls go through the same generic call_api() -- the eval code never touches
fusion internals directly. The judge inside the fusion pipeline is asked NOT
to think (reasoning_effort=none, handled inside mock_fusion_api); this script
only controls what it actually calls: the fusion endpoint and each baseline
endpoint, all with thinking on.

--baseline-model is a comma-separated list (0, 1, or N models); each question's
row gets "fusion" (one dict) and "baselines" (a list, one entry per model,
same order as --baseline-model). Fusion failing fails the whole row (there's
nothing useful to keep without it); each baseline is tried independently, so
one baseline model failing doesn't discard fusion's answer or another
baseline's -- it's just recorded as {"model": ..., "failed": true, "error":
...} in that one list slot. Either way, the row is written and the loop moves
on to the next question (eval.run_eval_question_failed / _baseline_failed to
stderr) -- gpqa_grade.py scores "fusion" and each baseline model separately,
by presence of a usable `content`, not by any row-level flag (a row can have
a good fusion answer and a failed baseline, or vice versa).

Resume: the default --out is keyed by --experiment (results/<experiment>.jsonl),
not a timestamp -- so re-running the SAME experiment name (e.g. after
reviewing a --limit 5 smoke test and now wanting the full 198) targets the
same file. On start, any row already in that file for a question_id, that
did NOT fail, is reused as-is and its API calls are skipped -- only
new/previously-failed questions are actually called. Resume is per-QUESTION,
not per-baseline: a reused row keeps whatever baseline entries it already has,
so a baseline that failed (or a baseline model added to --baseline-model after
the fact) is NOT filled in by re-running this script -- run() counts those and
says so on stderr; run_baseline.py is the tool that fills them in without
re-paying for fusion. Pass --no-resume to ignore whatever's there and
overwrite from scratch (e.g. after changing fusion_model/baseline_model under
the same experiment name -- resuming across a config change would silently mix
rows from two different configs into one file; run() warns to stderr if it
detects that specific mismatch, but doesn't block it).
Run: python3 -m eval.run_eval [--limit N]
"""
import argparse
import json
import sys

from .gpqa_tasks import load_tasks
from .client import call_api
from .replay_io import (ResumeMismatchError, carry_over_unprocessed, default_out_path,  # noqa: F401
                        ensure_out_dir, load_existing)


def run(fusion_url, fusion_model, baseline_url, baseline_models, out_path, limit=None, experiment=None,
        resume=True):
    """baseline_models: list of 0+ model names, called for every question
    alongside fusion. 0 -> no baseline calls at all, row["baselines"] = [].
    Each baseline is called independently (its own try/except): one baseline
    failing doesn't discard fusion's result or another baseline's -- only
    fusion failing fails the whole row, since fusion is the one thing every
    row needs to be useful at all."""
    ensure_out_dir(out_path)
    tasks = load_tasks(limit=limit)
    config_id = f"gpqa-v1-{fusion_model}-vs-{'+'.join(baseline_models) or 'none'}"
    expected = {t["question_id"]: (t["instruction"], t["correct_letter"]) for t in tasks}
    existing = load_existing(out_path, expected) if resume else {}

    if existing:
        prior_config_ids = {r.get("config_id") for r in existing.values() if r.get("config_id")}
        if prior_config_ids and prior_config_ids != {config_id}:
            print(f"eval.run_eval_resume_config_mismatch out_path={out_path!r} "
                  f"prior_config_id(s)={sorted(prior_config_ids)!r} this_run_config_id={config_id!r} "
                  f"-- reusing prior rows anyway; pass --no-resume if that's not what you want",
                  file=sys.stderr)

    skipped = failed = 0
    incomplete = {}  # baseline model -> how many reused rows still lack a usable answer from it
    with open(out_path, "w", encoding="utf-8") as f:
        for task in tasks:
            qid = task["question_id"]
            prior = existing.get(qid)
            if prior is not None and not prior.get("failed"):
                skipped += 1
                have = {b.get("model") for b in prior.get("baselines") or [] if not b.get("failed")}
                for bm in baseline_models:
                    if bm not in have:
                        incomplete[bm] = incomplete.get(bm, 0) + 1
                f.write(json.dumps(prior) + "\n")
                continue
            messages = [{"role": "user", "content": task["instruction"]}]
            # Fusion's call AND its result must stay inside this one try
            # block: indexing the response is itself a failure point, so a
            # call that returns 200 with an unexpected shape (e.g. an empty
            # choices list) would crash the whole run if used outside it.
            try:
                fusion_resp = call_api(fusion_url, fusion_model, messages, allow_tool_call_output=False,
                                        experiment=experiment, question_id=qid)
                fusion_data = {
                    "model": fusion_model,
                    "content": fusion_resp["choices"][0]["message"]["content"],
                    "reasoning_content": fusion_resp["choices"][0]["message"].get("reasoning_content"),
                    "raw_response": fusion_resp,
                }
            except Exception as e:
                failed += 1
                print(f"eval.run_eval_question_failed question_id={qid!r} error={str(e)!r}", file=sys.stderr)
                f.write(json.dumps({
                    "schema": "0g.fusion_eval.gpqa.replay.v1",
                    "question_id": qid,
                    "instruction": task["instruction"],
                    "correct_letter": task["correct_letter"],
                    "config_id": config_id,
                    "failed": True,
                    "error": str(e),
                }) + "\n")
                continue

            baselines = []
            for bm in baseline_models:
                try:
                    baseline_resp = call_api(baseline_url, bm, messages, reasoning_effort="high",
                                              experiment=experiment)
                    baselines.append({
                        "model": bm,
                        "content": baseline_resp["choices"][0]["message"]["content"],
                        "reasoning_content": baseline_resp["choices"][0]["message"].get("reasoning_content"),
                        "raw_response": baseline_resp,
                    })
                except Exception as e:
                    print(f"eval.run_eval_baseline_failed question_id={qid!r} baseline_model={bm!r} "
                          f"error={str(e)!r}", file=sys.stderr)
                    baselines.append({"model": bm, "failed": True, "error": str(e)})

            f.write(json.dumps({
                "schema": "0g.fusion_eval.gpqa.replay.v1",
                "question_id": qid,
                "instruction": task["instruction"],
                "correct_letter": task["correct_letter"],
                "fusion": fusion_data,
                "baselines": baselines,
                "config_id": config_id,
            }) + "\n")
        # --limit only: without one, `expected` IS the whole dataset/base file and
        # anything else in --out is stale, not something to preserve.
        carried = carry_over_unprocessed(f, existing, expected) if limit else 0
    if skipped:
        print(f"eval.run_eval_resumed skipped={skipped} (already had a successful result at {out_path!r}, "
              f"reused as-is without calling anything)", file=sys.stderr)
    if carried:
        print(f"eval.run_eval_carried_over kept={carried} prior row(s) for questions outside this "
              f"--limit window (they are not re-run, just not thrown away)", file=sys.stderr)
    if incomplete:
        print(f"eval.run_eval_baselines_still_missing {incomplete!r} -- reused rows have no usable answer "
              f"from these baseline model(s) (they failed earlier, or were added to --baseline-model after "
              f"those rows were written). Resume is per-question, so re-running THIS script will not fill "
              f"them in; run `python3 -m eval.run_baseline --base-replay {out_path} --out {out_path} "
              f"--baseline-model <model>` to add just the missing side.", file=sys.stderr)
    if failed:
        print(f"eval.run_eval_summary total={len(tasks)} skipped={skipped} failed={failed}", file=sys.stderr)
    return out_path


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--fusion-url", default="http://localhost:8000")
    p.add_argument("--fusion-model", default="0g/fusion-preview")
    p.add_argument("--baseline-url", default="http://localhost:8000")
    p.add_argument("--baseline-model", default="baseline-model",
                    help="Comma-separated list of 0+ baseline model names, called for every question "
                         "alongside fusion (e.g. 'gpt-5.6-sol,claude-fable-5'). Pass an empty string for "
                         "no baselines at all (fusion only). Repeats are collapsed to one call.")
    p.add_argument("--out", default=None,
                    help="Defaults to results/<experiment>.jsonl -- keyed by experiment name, not a "
                         "timestamp, so re-running the same --experiment resumes into the same file.")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--experiment", default=None,
                    help="Experiment name used to name per-call log files "
                         "(call_logs/<experiment>__<role>__<model>.jsonl, see llm_client.py) AND, if "
                         "--out isn't given, the default output file. "
                         "Defaults to <fusion-model>-vs-<baseline-model(s)> if not given.")
    p.add_argument("--no-resume", action="store_true",
                    help="Ignore any existing rows at --out and overwrite from scratch, instead of "
                         "reusing already-succeeded questions and only (re-)running the rest.")
    args = p.parse_args()
    # dict.fromkeys, not set(): order is the documented order of the
    # `baselines` list, and a repeated name (a copy-paste typo in a
    # comma-separated flag) would otherwise be called -- and paid for -- twice
    # per question, for a duplicate list entry grading can't even distinguish.
    baseline_models = list(dict.fromkeys(m.strip() for m in args.baseline_model.split(",") if m.strip()))
    experiment = args.experiment or f"{args.fusion_model}-vs-{'+'.join(baseline_models) or 'none'}"
    out = args.out or default_out_path(experiment)
    print(run(args.fusion_url, args.fusion_model, args.baseline_url, baseline_models, out, args.limit,
               experiment, resume=not args.no_resume))
