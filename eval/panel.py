"""Compute panel-member answers for GPQA questions -- no judge, no synthesis,
no baselines, so building/extending a panel costs only the panel models
themselves. A building block: `eval.fuse` turns any panel file into a real
fusion result; `eval.baseline` computes baselines completely separately.
Chain them like any other small tool -- each reads/writes a plain JSONL
keyed by question_id, nothing more.

--models is the FULL desired panel for this file, always -- not "add one
more to whatever's already there" (that's what makes this safe against the
old accumulation bug: a panel is stated completely, every time, never
built by implicitly stacking on some prior unstated set).

--reuse <file> pulls already-computed model answers from a DIFFERENT file
(e.g. an earlier 4-model panel) for models in --models that appear there,
so only the rest get called fresh. Resuming the SAME --experiment (--out
already has some/all of this exact --models set for a question) is
separate and automatic, regardless of --reuse.

Run:
  # the 4 fixed panel members, once
  python3 -m eval.panel --fusion-url http://localhost:8000 \\
      --models minimax-m3,kimi-k3,glm-5.2,deepseek-v4-pro --experiment gpqa-panel-fixed4

  # a 5-member variant: reuse the 4 fixed ones, call only the new candidate
  python3 -m eval.panel --fusion-url http://localhost:8000 \\
      --models minimax-m3,kimi-k3,glm-5.2,deepseek-v4-pro,hy3 \\
      --reuse eval/results/gpqa-panel-fixed4.jsonl --experiment gpqa-panel-hy3
"""
import argparse
import json
import os
import sys

from .gpqa_tasks import load_tasks
from .client import call_api
from .replay_io import carry_over_unprocessed, default_out_path, ensure_out_dir, load_existing


SCHEMA = "0g.fusion_eval.gpqa.panel.v1"


def _panel_by_model(row):
    return {p["model"]: p for p in (row.get("panel") or []) if not p.get("failed")}


def _base_row(qid, task):
    return {"schema": SCHEMA, "question_id": qid, "instruction": task["instruction"],
            "correct_letter": task["correct_letter"]}


def run(fusion_url, fusion_model, models, out_path, limit=None, experiment=None, resume=True, reuse_path=None):
    ensure_out_dir(out_path)
    if not fusion_model.startswith("0g/fusion"):
        # mock_fusion_api.handle_chat_completion only enters the panel_only
        # path for a "0g/fusion*" model id -- anything else silently falls
        # through to the plain baseline passthrough instead, which bills one
        # real call per question and returns no `0g_fusion.panel` at all
        # (every question ends up "failed" with a bare KeyError). Refuse
        # before paying for that.
        raise ValueError(f"--fusion-model {fusion_model!r} does not start with '0g/fusion' -- it "
                          f"would never reach the panel_only path and every question would fail "
                          f"after being billed as a plain baseline call instead.")
    if reuse_path and not os.path.exists(reuse_path):
        # load_existing() below would otherwise treat a missing --reuse path
        # exactly like a brand-new --out (returns {}) and silently re-call
        # every model at full price -- a typo'd path must fail loud, not
        # quietly cost 100% instead of the intended ~20%.
        raise FileNotFoundError(
            f"--reuse path {reuse_path!r} does not exist -- refusing to silently treat it as empty "
            f"and re-call every model at full price. Check the path."
        )
    tasks = load_tasks(limit=limit)
    expected = {t["question_id"]: (t["instruction"], t["correct_letter"]) for t in tasks}
    existing = load_existing(out_path, expected, expected_schema=SCHEMA) if resume else {}
    reuse = load_existing(reuse_path, expected, expected_schema=SCHEMA) if reuse_path else {}

    skipped = failed = 0
    dropped = {}  # model -> number of questions where a prior answer for it is being discarded
    with open(out_path, "w", encoding="utf-8") as f:
        for task in tasks:
            qid = task["question_id"]
            prior = existing.get(qid)
            # Read from a prior row even if it's marked failed -- a batch
            # failure (see the except branch below) can still have left some
            # models successfully cached, and those are worth keeping instead
            # of re-calling them too.
            have_here = _panel_by_model(prior) if prior else {}
            # --models is the FULL desired panel, always -- a model present
            # here but not in --models gets dropped from the row, silently
            # except for this count. Surfacing it matters: a near-identical
            # `eval.baseline --models x --out <file>` ACCUMULATES instead of
            # replacing, so typing the panel version of that command by
            # habit would otherwise throw away already-paid panel members
            # with no visible sign anything happened.
            for m in set(have_here) - set(models):
                dropped[m] = dropped.get(m, 0) + 1
            have_reuse = _panel_by_model(reuse.get(qid, {}))
            cached, fresh_models = [], []
            for m in models:
                if m in have_here:
                    cached.append(have_here[m])
                elif m in have_reuse:
                    cached.append(have_reuse[m])
                else:
                    fresh_models.append(m)

            if not fresh_models:
                # Everything needed was already sitting in --out/--reuse --
                # no fresh model to call, so skip the round trip entirely
                # rather than making a call whose only job is to hand back
                # data we already have.
                skipped += 1
                f.write(json.dumps({**_base_row(qid, task), "panel": cached}) + "\n")
                continue

            messages = [{"role": "user", "content": task["instruction"]}]
            # The call AND the row construction must stay inside this one try
            # block: indexing the response is itself a failure point, so a
            # 200-but-wrong-shape response would crash the whole run if the
            # row were built outside it.
            try:
                resp = call_api(fusion_url, fusion_model, messages, panel_only=True,
                                 cached_panel=cached, extra_panel_models=fresh_models,
                                 experiment=experiment, question_id=qid)
                row = {**_base_row(qid, task), "panel": resp["0g_fusion"]["panel"]}
            except Exception as e:
                failed += 1
                print(f"eval.panel_question_failed question_id={qid!r} error={str(e)!r}", file=sys.stderr)
                # `cached` (whatever was already sitting in --out/--reuse
                # before this call) survives even though the fresh call for
                # the rest failed -- a retry can still reuse it instead of
                # re-paying for models that already succeeded.
                row = {**_base_row(qid, task), "panel": cached, "failed": True, "error": str(e)}
            f.write(json.dumps(row) + "\n")
        # Always carry forward rows outside `expected`, not just when --limit
        # is set: `expected` can also shrink because load_tasks()'s underlying
        # dataset file shrank (e.g. re-downloaded with its own --limit) with
        # no --limit given to THIS run at all. Guarding this on `limit` alone
        # missed exactly that case and silently deleted already-paid rows.
        carried = carry_over_unprocessed(f, existing, expected)
    if skipped:
        print(f"eval.panel_resumed skipped={skipped} (already had every requested model for these "
              f"questions at {out_path!r})", file=sys.stderr)
    if carried:
        print(f"eval.panel_carried_over={carried} (rows outside this run's question set, left "
              f"untouched at {out_path!r})", file=sys.stderr)
    if dropped:
        print(f"eval.panel_dropped_models={dropped} (present in {out_path!r} before this run, not in "
              f"--models this time, so removed from these questions' panel -- expected if you meant to "
              f"shrink the panel, a mistake if you meant eval.baseline's accumulate behavior instead)",
              file=sys.stderr)
    if failed:
        print(f"eval.panel_summary total={len(tasks)} skipped={skipped} failed={failed}", file=sys.stderr)
    return out_path


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--fusion-url", default="http://localhost:8000")
    p.add_argument("--fusion-model", default="0g/fusion-preview")
    p.add_argument("--models", required=True, help="Comma-separated full desired panel for this file.")
    p.add_argument("--reuse", default=None, help="Pull already-computed answers from this file where possible.")
    p.add_argument("--out", default=None, help="Defaults to results/<experiment>.jsonl.")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--experiment", default=None, help="Defaults to panel-<models joined by '+'>.")
    p.add_argument("--no-resume", action="store_true", help="Ignore any existing rows at --out.")
    args = p.parse_args()
    models = list(dict.fromkeys(m.strip() for m in args.models.split(",") if m.strip()))
    if not models:
        sys.exit("--models must name at least one model")
    experiment = args.experiment or f"panel-{'+'.join(models)}"
    out = args.out or default_out_path(experiment)
    print(run(args.fusion_url, args.fusion_model, models, out, args.limit, experiment,
              resume=not args.no_resume, reuse_path=args.reuse))
