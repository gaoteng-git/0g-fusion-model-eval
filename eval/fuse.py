"""Run judge+synthesis over an already-built panel file (eval/panel.py) --
no panel calls happen here, the panel is used exactly as given. This is the
one step that actually costs judge+synthesis money, kept separate from
eval/panel.py on purpose: build, inspect, and reuse a panel as many times as
you like, then pay for a real fusion result against it exactly when ready.

Run:
  python3 -m eval.fuse --fusion-url http://localhost:8000 \\
      --panel eval/results/gpqa-panel-hy3.jsonl --experiment gpqa-fuse-hy3
"""
import argparse
import json
import sys

from .client import call_api
from .replay_io import (ResumeMismatchError, carry_over_unprocessed, default_out_path,  # noqa: F401
                         ensure_out_dir, load_existing)

SCHEMA = "0g.fusion_eval.gpqa.replay.v1"


def run(fusion_url, fusion_model, panel_path, out_path, limit=None, experiment=None, resume=True):
    with open(panel_path, encoding="utf-8") as f:
        panel_rows = [json.loads(line) for line in f if line.strip()]
    if limit:
        panel_rows = panel_rows[:limit]

    qid_counts = {}
    for r in panel_rows:
        qid_counts[r.get("question_id")] = qid_counts.get(r.get("question_id"), 0) + 1
    dupes = sorted(q for q, count in qid_counts.items() if count > 1)
    if dupes:
        raise ValueError(
            f"{panel_path!r} has duplicate question_id(s) {dupes!r} -- refusing to fuse: judge+synthesis "
            f"would be paid for twice for those questions. Deduplicate the panel file first."
        )

    expected = {r.get("question_id"): (r.get("instruction"), r.get("correct_letter")) for r in panel_rows}
    existing = load_existing(out_path, expected, expected_schema=SCHEMA) if resume else {}
    ensure_out_dir(out_path)

    skipped = failed = 0
    with open(out_path, "w", encoding="utf-8") as out_f:
        for panel_row in panel_rows:
            qid = panel_row.get("question_id")
            prior = existing.get(qid)
            if prior is not None and not prior.get("failed"):
                skipped += 1
                out_f.write(json.dumps(prior) + "\n")
                continue

            if panel_row.get("failed") or not panel_row.get("panel"):
                failed += 1
                print(f"eval.fuse_question_skipped question_id={qid!r} reason='no panel available'",
                      file=sys.stderr)
                out_f.write(json.dumps({
                    "schema": SCHEMA,
                    "question_id": qid,
                    "instruction": panel_row.get("instruction"),
                    "correct_letter": panel_row.get("correct_letter"),
                    "failed": True,
                    "error": panel_row.get("error", "no panel available"),
                }) + "\n")
                continue

            # The call AND the row construction (including reading instruction/
            # correct_letter/panel back out of panel_row) must all stay inside
            # this one try block: any of those is itself a failure point on a
            # malformed panel row, and the except branch below only uses
            # .get() so it can't ALSO raise while building the failure row.
            try:
                messages = [{"role": "user", "content": panel_row["instruction"]}]
                panel_models = [p["model"] for p in panel_row["panel"]]
                fusion_resp = call_api(fusion_url, fusion_model, messages, cached_panel=panel_row["panel"],
                                        experiment=experiment, question_id=qid)
                row = {
                    "schema": SCHEMA,
                    "question_id": qid,
                    "instruction": panel_row["instruction"],
                    "correct_letter": panel_row["correct_letter"],
                    "panel": panel_row["panel"],
                    "fusion": {
                        "model": fusion_model,
                        "content": fusion_resp["choices"][0]["message"]["content"],
                        "reasoning_content": fusion_resp["choices"][0]["message"].get("reasoning_content"),
                        "raw_response": fusion_resp,
                    },
                    "config_id": f"gpqa-v1-{fusion_model}-on-panel:{'+'.join(panel_models)}",
                }
            except Exception as e:
                failed += 1
                print(f"eval.fuse_question_failed question_id={qid!r} error={str(e)!r}", file=sys.stderr)
                row = {
                    "schema": SCHEMA,
                    "question_id": qid,
                    "instruction": panel_row.get("instruction"),
                    "correct_letter": panel_row.get("correct_letter"),
                    "panel": panel_row.get("panel"),
                    "failed": True,
                    "error": str(e),
                }
            out_f.write(json.dumps(row) + "\n")
        # Unlike eval.panel/eval.baseline (whose `expected` is load_tasks()'s
        # full dataset when --limit is unset), `expected` here is only
        # whatever --panel currently contains -- which can be SMALLER than a
        # prior run's, for reasons that have nothing to do with THIS run's
        # --limit (an interrupted eval.panel run, a hand-edited/concatenated
        # panel file, an earlier --no-resume --limit). Always carry forward
        # rows outside that window, or a shrunk --panel silently deletes
        # already-paid judge+synthesis results instead of just not touching
        # them.
        skipped += carry_over_unprocessed(out_f, existing, expected)
    if skipped:
        print(f"eval.fuse_resumed skipped={skipped} (already had a successful result at {out_path!r})",
              file=sys.stderr)
    if failed:
        print(f"eval.fuse_summary total={len(panel_rows)} skipped={skipped} failed={failed}", file=sys.stderr)
    return out_path


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--fusion-url", default="http://localhost:8000")
    p.add_argument("--fusion-model", default="0g/fusion-preview")
    p.add_argument("--panel", required=True, help="A panel file from eval.panel.")
    p.add_argument("--out", default=None, help="Defaults to results/<experiment>.jsonl.")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--experiment", default=None, help="Defaults to fuse-on-<panel filename>.")
    p.add_argument("--no-resume", action="store_true", help="Ignore any existing rows at --out.")
    args = p.parse_args()
    experiment = args.experiment or f"fuse-on-{args.panel.rsplit('/', 1)[-1].removesuffix('.jsonl')}"
    out = args.out or default_out_path(experiment)
    print(run(args.fusion_url, args.fusion_model, args.panel, out, args.limit, experiment,
              resume=not args.no_resume))
