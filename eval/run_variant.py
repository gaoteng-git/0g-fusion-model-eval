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

The baseline side isn't re-run here -- it didn't change between variants, so
the base replay's `baselines` list is carried over unchanged into every
variant output row, including the rows whose own fusion call failed (grading
scores each side by presence, so a variant file must still report the same
baseline accuracy as its base file, not a lower one).

A question whose call ultimately fails (or whose base row itself was a
{"failed": true} row from run_eval.py, with no cached_panel to reuse) does
NOT abort the run -- it's logged to stderr and written as a failed row, same
convention as run_eval.py -- see its docstring.

Setup guard: this whole mechanism only produces a clean "N fixed + 1
candidate" panel if the base run's own panel had exactly `fixed_count`
members to begin with. If the base run was accidentally configured with the
candidate ALREADY baked in (e.g. ZG_PANEL_MODELS had 5 models instead of 4),
cached_panel carries that leftover candidate forward too, and each variant
run silently ACCUMULATES panel members (5 -> 6 -> 7 across variants) instead
of doing a 1-for-1 swap. run() checks this upfront (before making any calls) against
the first non-failed base row, and again per-question inside the loop as a
belt-and-suspenders check on file consistency -- both raise VariantSetupError,
which is NEVER treated as a per-question failure (it always aborts the
whole run, unlike a transient API error): a misconfigured base run is a
setup mistake to fix and re-run, not something to silently paper over one
question at a time. Also rejects variant_model already being present in
cached_panel (already baked into the base, or this variant already run
against this exact base file).

Resume: same convention as run_eval.py -- the default --out is keyed by
--experiment (results/<experiment>.jsonl), not a timestamp. On start, any
row already there for a question_id, that did NOT fail, is reused as-is
(the candidate model is NOT re-called for it) -- only new/previously-failed
questions actually make a call. Pass --no-resume to overwrite from scratch.
Run:
  python3 -m eval.run_variant --base-replay eval/results/run_BASE.jsonl \\
      --variant-model xiaomi/mimo-v2.5-pro --out eval/results/variant_mimo.jsonl
"""
import argparse
import json
import sys

from .client import call_api
from .replay_io import (ResumeMismatchError, carry_over_unprocessed, default_out_path,  # noqa: F401
                        ensure_out_dir, load_existing)


class VariantSetupError(Exception):
    """The base run's panel doesn't have the shape this variant run expects
    -- a structural/configuration mistake, not a transient per-question
    failure. Always aborts the whole run (see run()); never caught by the
    per-question try/except that handles ordinary API failures."""


def _validate_variant_setup(cached_panel, variant_model, fixed_count):
    if fixed_count is not None and len(cached_panel) != fixed_count:
        raise VariantSetupError(
            f"cached_panel has {len(cached_panel)} entries, expected exactly {fixed_count} fixed "
            f"panel members (pass fixed_count=None / --fixed-count -1 to skip this check if that's "
            f"intentional). This usually means the base run (run_eval.py) was configured with the "
            f"wrong ZG_PANEL_MODELS -- e.g. a candidate already baked in -- so building variants "
            f"from it would silently accumulate panel members (5 -> 6 -> 7 across variants) instead "
            f"of doing a clean 1-for-1 swap. Panel models found: {[p.get('model') for p in cached_panel]!r}"
        )
    cached_models = [p.get("model") for p in cached_panel]
    if variant_model in cached_models:
        raise VariantSetupError(
            f"variant_model {variant_model!r} is already present in cached_panel {cached_models!r} -- "
            f"either it's already baked into the base run, or this exact variant has already been "
            f"run against this base file."
        )


def run(base_replay_path, fusion_url, fusion_model, variant_model, out_path, limit=None, experiment=None,
        resume=True, fixed_count=4):
    ensure_out_dir(out_path)
    with open(base_replay_path, encoding="utf-8") as f:
        base_rows = [json.loads(line) for line in f if line.strip()]
    if limit:
        base_rows = base_rows[:limit]

    # Fail fast, before making any calls, using the first usable base row --
    # a structural mismatch should stop the run immediately, not surface 198
    # calls later as 198 individual per-question failures.
    first_valid = next((r for r in base_rows if not r.get("failed")), None)
    if first_valid is not None:
        _validate_variant_setup(first_valid["fusion"]["raw_response"]["0g_fusion"]["panel"],
                                 variant_model, fixed_count)

    expected = {r.get("question_id"): (r.get("instruction"), r.get("correct_letter")) for r in base_rows}
    existing = load_existing(out_path, expected) if resume else {}

    skipped = failed = 0
    with open(out_path, "w", encoding="utf-8") as out_f:
        for base_row in base_rows:
            qid = base_row.get("question_id")
            prior = existing.get(qid)
            if prior is not None and not prior.get("failed"):
                skipped += 1
                out_f.write(json.dumps(prior) + "\n")
                continue

            # A base row that itself failed (see run_eval.py) has no
            # fusion.raw_response.0g_fusion.panel to reuse -- skip it the
            # same way, rather than crashing on a missing key.
            if base_row.get("failed"):
                failed += 1
                print(f"eval.run_variant_question_skipped question_id={qid!r} "
                      f"reason='base row itself failed, no cached_panel available'", file=sys.stderr)
                out_f.write(json.dumps({**base_row, "variant_model": variant_model}) + "\n")
                continue
            messages = [{"role": "user", "content": base_row["instruction"]}]
            # The cached_panel lookup, the call AND the success-row
            # construction must all stay inside this one try block: indexing
            # base_row and indexing the response are themselves failure points,
            # so a 200-but-wrong-shape response or a malformed base_row would
            # crash the whole run if either were done outside it.
            try:
                cached_panel = base_row["fusion"]["raw_response"]["0g_fusion"]["panel"]
                _validate_variant_setup(cached_panel, variant_model, fixed_count)
                fusion_resp = call_api(
                    fusion_url, fusion_model, messages,
                    cached_panel=cached_panel, extra_panel_models=[variant_model],
                    experiment=experiment, question_id=qid,
                )
                row = {
                    "schema": "0g.fusion_eval.gpqa.replay.v1",
                    "question_id": qid,
                    "instruction": base_row["instruction"],
                    "correct_letter": base_row["correct_letter"],
                    "fusion": {
                        "model": fusion_model,
                        "content": fusion_resp["choices"][0]["message"]["content"],
                        "reasoning_content": fusion_resp["choices"][0]["message"].get("reasoning_content"),
                        "raw_response": fusion_resp,
                    },
                    "baselines": base_row.get("baselines", []),  # unchanged from the base run -- not re-called
                    "config_id": f"gpqa-v1-variant-{variant_model}-on-{base_row['config_id']}",
                    "variant_of": base_row["config_id"],
                    "cached_panel_models": [p["model"] for p in cached_panel],
                    "variant_model": variant_model,
                }
            except VariantSetupError:
                raise  # structural problem -- must abort, not be absorbed as a per-question failure
            except Exception as e:
                failed += 1
                print(f"eval.run_variant_question_failed question_id={qid!r} error={str(e)!r}", file=sys.stderr)
                row = {
                    "schema": "0g.fusion_eval.gpqa.replay.v1",
                    "question_id": qid,
                    "instruction": base_row["instruction"],
                    "correct_letter": base_row["correct_letter"],
                    # Only the FUSION side failed here; the baselines were
                    # never re-called and are still perfectly good, so they
                    # ride along exactly as on a successful row -- dropping
                    # them would make this file report a lower baseline
                    # accuracy than the base file it was built from.
                    "baselines": base_row.get("baselines", []),
                    "config_id": f"gpqa-v1-variant-{variant_model}-on-{base_row['config_id']}",
                    "variant_of": base_row["config_id"],
                    "variant_model": variant_model,
                    "failed": True,
                    "error": str(e),
                }
            out_f.write(json.dumps(row) + "\n")
        # --limit only: without one, `expected` IS the whole dataset/base file and
        # anything else in --out is stale, not something to preserve.
        carried = carry_over_unprocessed(out_f, existing, expected) if limit else 0
    if skipped:
        print(f"eval.run_variant_resumed skipped={skipped} (already had a successful result at {out_path!r}, "
              f"reused as-is without calling anything)", file=sys.stderr)
    if carried:
        print(f"eval.run_variant_carried_over kept={carried} prior row(s) for questions outside this "
              f"--limit window (they are not re-run, just not thrown away)", file=sys.stderr)
    if failed:
        print(f"eval.run_variant_summary total={len(base_rows)} skipped={skipped} failed={failed}", file=sys.stderr)
    return out_path


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base-replay", required=True, help="Replay JSONL from a prior run_eval.py run.")
    p.add_argument("--fusion-url", default="http://localhost:8000")
    p.add_argument("--fusion-model", default="0g/fusion-preview")
    p.add_argument("--variant-model", required=True, help="The one new panel model to actually call.")
    p.add_argument("--out", default=None,
                    help="Defaults to results/<experiment>.jsonl -- keyed by experiment name, not a "
                         "timestamp, so re-running the same --experiment resumes into the same file.")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--experiment", default=None,
                    help="Defaults to variant-<variant-model>.")
    p.add_argument("--no-resume", action="store_true",
                    help="Ignore any existing rows at --out and overwrite from scratch.")
    p.add_argument("--fixed-count", type=int, default=4,
                    help="Expected number of fixed panel members in the base run (default 4). Aborts "
                         "immediately if the base run's actual panel size doesn't match -- catches a "
                         "base run accidentally configured with a candidate already baked in, which "
                         "would otherwise silently accumulate panel members across variants instead "
                         "of doing a clean 1-for-1 swap. Pass -1 to skip this check.")
    args = p.parse_args()
    experiment = args.experiment or f"variant-{args.variant_model}"
    out = args.out or default_out_path(experiment)
    fixed_count = None if args.fixed_count == -1 else args.fixed_count
    print(run(args.base_replay, args.fusion_url, args.fusion_model, args.variant_model, out, args.limit,
               experiment, resume=not args.no_resume, fixed_count=fixed_count))
