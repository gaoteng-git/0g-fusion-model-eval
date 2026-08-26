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
from .replay_io import ResumeMismatchError, add_out_args, default_out_path, ensure_out_dir, load_existing, run_replay

SCHEMA = "0g.fusion_eval.gpqa.replay.v1"


def _base_row(qid, panel_row):
    return {"schema": SCHEMA, "question_id": qid, "instruction": panel_row.get("instruction"),
            "correct_letter": panel_row.get("correct_letter")}


def _config_id(fusion_model, panel_row):
    panel_models = [p["model"] for p in panel_row.get("panel") or []]
    return f"gpqa-v1-{fusion_model}-on-panel:{'+'.join(panel_models)}"


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
    # load_existing() only checks instruction/correct_letter/schema -- it has
    # no idea what a "panel" even means. But two panel files can easily share
    # a question set while differing in panel COMPOSITION (that's the whole
    # point of eval.panel --reuse variants), and --experiment is just a
    # string the operator has to remember to change alongside --panel. Catch
    # that here, before --out is opened for writing, so a stale --experiment
    # aimed at a different --panel is refused instead of silently reporting
    # "already fused" against the WRONG panel's judge+synthesis result.
    for panel_row in panel_rows:
        qid = panel_row.get("question_id")
        prior = existing.get(qid)
        if prior is None or prior.get("failed") or not prior.get("config_id"):
            continue
        this_config_id = _config_id(fusion_model, panel_row)
        if prior["config_id"] != this_config_id:
            raise ResumeMismatchError(
                f"{out_path!r} already has a row for question_id={qid!r} fused from a DIFFERENT panel "
                f"than {panel_path!r} gives now -- prior config_id={prior['config_id']!r}, this run "
                f"would produce {this_config_id!r}. --experiment likely wasn't changed along with "
                f"--panel. Use a new --experiment/--out, or --no-resume to recompute."
            )
    ensure_out_dir(out_path)

    def process(panel_row, prior):
        qid = panel_row.get("question_id")
        if prior is not None and not prior.get("failed"):
            return prior, {"skipped": 1}

        if panel_row.get("failed") or not panel_row.get("panel"):
            print(f"eval.fuse_question_skipped question_id={qid!r} reason='no panel available'", file=sys.stderr)
            return ({**_base_row(qid, panel_row), "failed": True,
                     "error": panel_row.get("error", "no panel available")}, {"failed": 1})

        # The call AND the row construction (including reading instruction/
        # correct_letter/panel back out of panel_row) must all stay inside
        # this one try block: any of those is itself a failure point on a
        # malformed panel row, and the except branch below only uses .get()
        # so it can't ALSO raise while building the failure row.
        try:
            messages = [{"role": "user", "content": panel_row["instruction"]}]
            fusion_resp = call_api(fusion_url, fusion_model, messages, cached_panel=panel_row["panel"],
                                    experiment=experiment, question_id=qid)
            row = {
                **_base_row(qid, panel_row),
                "panel": panel_row["panel"],
                "fusion": {
                    "model": fusion_model,
                    "content": fusion_resp["choices"][0]["message"]["content"],
                    "reasoning_content": fusion_resp["choices"][0]["message"].get("reasoning_content"),
                    "raw_response": fusion_resp,
                },
                "config_id": _config_id(fusion_model, panel_row),
            }
            return row, {}
        except Exception as e:
            print(f"eval.fuse_question_failed question_id={qid!r} error={str(e)!r}", file=sys.stderr)
            row = {**_base_row(qid, panel_row), "panel": panel_row.get("panel"), "failed": True, "error": str(e)}
            return row, {"failed": 1}

    counts = run_replay(panel_rows, lambda r: r.get("question_id"), process, out_path, existing, expected)
    skipped, carried, failed = counts.get("skipped", 0), counts.get("carried", 0), counts.get("failed", 0)
    if skipped:
        print(f"eval.fuse_resumed skipped={skipped} (already had a successful result at {out_path!r})",
              file=sys.stderr)
    if carried:
        print(f"eval.fuse_carried_over={carried} (rows outside this run's --panel/--limit window, "
              f"left untouched at {out_path!r})", file=sys.stderr)
    if failed:
        print(f"eval.fuse_summary total={len(panel_rows)} skipped={skipped} failed={failed}", file=sys.stderr)
    return out_path


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--fusion-url", default="http://localhost:8000")
    p.add_argument("--fusion-model", default="0g/fusion-preview")
    p.add_argument("--panel", required=True, help="A panel file from eval.panel.")
    add_out_args(p)
    p.add_argument("--experiment", default=None, help="Defaults to fuse-on-<panel filename>.")
    args = p.parse_args()
    experiment = args.experiment or f"fuse-on-{args.panel.rsplit('/', 1)[-1].removesuffix('.jsonl')}"
    out = args.out or default_out_path(experiment)
    print(run(args.fusion_url, args.fusion_model, args.panel, out, args.limit, experiment,
              resume=not args.no_resume))
