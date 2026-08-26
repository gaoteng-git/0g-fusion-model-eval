"""Run judge+synthesis over N already-computed panel files (see eval.panel.py
-- typically one per fixed panel member plus one candidate, 5 files) for
every question in --input, and write the fused answer.

--judge-model and --synthesis-model are always explicit arguments here, not
global config: this tool has no notion of a single fixed judge/synthesis
model baked in anywhere else.

--input is the single source of truth for question identity and ground
truth (instruction/correct_letter are recomputed fresh from it, the same
deterministic way eval.panel.py did when it built each --panels file) --
the panel files are ONLY consulted for their (model, content, reasoning)
answers, keyed by question_id. If any --panels file is missing (or failed)
a question that's in --input, that ONE question is written as failed
(naming which file was short) and the run continues -- no judge/synthesis
call is made for it, so an incomplete panel costs nothing beyond itself.

No resume, no --limit: every run computes every question in --input, fresh,
and OVERWRITES --out completely.

Run:
  python3 -m eval.fuse --judge-model minimax-m3 --synthesis-model kimi-k3 \\
      --api-url http://localhost:8000 --input eval/samples/first5.jsonl \\
      --panels eval/results/panel-minimax-m3.jsonl,eval/results/panel-kimi-k3.jsonl,\\
eval/results/panel-glm-5.2.jsonl,eval/results/panel-deepseek-v4-pro.jsonl,eval/results/panel-hy3.jsonl \\
      --out eval/results/fuse-hy3.jsonl
"""
import argparse
import json
import os
import sys

from .gpqa_tasks import load_questions, format_question
from .client import call_api
from mock_fusion_api.panel_config import JUDGE_SYSTEM, JUDGE_MODELS_WITHOUT_JSON_MODE, SYNTHESIS_FALLBACK_PROMPT
from mock_fusion_api.pipeline import panel_evidence


def _load_panel_file(path):
    """question_id -> that file's row, for every question it answered
    successfully (a `"failed": true` row is treated as absent -- there's no
    usable answer to fuse for that question from this panel member)."""
    by_id = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if not row.get("failed"):
                by_id[row["question_id"]] = row
    return by_id


def run(api_url, judge_model, synthesis_model, input_path, panel_paths, out_path, experiment=None):
    dupes = sorted({p for p in panel_paths if panel_paths.count(p) > 1})
    if dupes:
        # The same file listed twice would double-weight that panel member's
        # vote in judge/synthesis and still bill for it once per occurrence --
        # never something anyone actually wants, so refuse before any calls.
        raise ValueError(f"--panels lists the same file more than once: {dupes!r}")

    questions = load_questions(input_path)
    panels = [(path, _load_panel_file(path)) for path in panel_paths]
    # eval/results/ is gitignored, so it doesn't exist on a fresh clone -- a
    # bare open() would fail before writing anything.
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    with open(out_path, "w", encoding="utf-8") as f:
        for question_id, row in questions:
            instruction, correct_letter = format_question(row, question_id)
            base = {"question_id": question_id, "instruction": instruction, "correct_letter": correct_letter,
                    "judge_model": judge_model, "synthesis_model": synthesis_model}

            # A panel file is unusable for this question if it's missing the
            # question entirely (or that row failed) -- OR if the row it DOES
            # have disagrees with --input about what the question even is
            # (e.g. --panels built from a different/reordered question file).
            # Either way, fusing it would silently answer the wrong question
            # while looking completely normal, so catch it here using data
            # already in hand (every panel row already carries its own
            # `instruction`) rather than trusting question_id alone.
            bad = []
            for path, by_id in panels:
                prow = by_id.get(question_id)
                if prow is None:
                    bad.append(f"{path!r} (missing/failed)")
                elif prow.get("instruction") is not None and prow["instruction"] != instruction:
                    bad.append(f"{path!r} (different question than --input)")
            if bad:
                print(f"eval.fuse_question_skipped question_id={question_id!r} reason={bad!r}", file=sys.stderr)
                f.write(json.dumps({**base, "failed": True,
                                    "error": f"unusable panel file(s) for this question: {bad}"}) + "\n")
                continue

            panel_rows = [by_id[question_id] for _, by_id in panels]
            panel_models = [r.get("model") for r in panel_rows]
            panel_results = [{"content": r.get("content"), "reasoning": r.get("reasoning")} for r in panel_rows]

            # The judge call, the synthesis call, AND the row construction
            # must all stay inside this one try block: indexing either
            # response is itself a failure point, so a 200-but-wrong-shape
            # response would crash the whole run if the row were built
            # outside it.
            try:
                evidence = panel_evidence(panel_results)
                judge_user = (f"Original request summary:\nUSER: {instruction}\n\nPanel responses:\n"
                               + evidence.split("Panel answers:\n", 1)[-1])
                supports_json_mode = judge_model.strip().lower() not in JUDGE_MODELS_WITHOUT_JSON_MODE
                judge_resp = call_api(
                    api_url, judge_model,
                    [{"role": "system", "content": JUDGE_SYSTEM}, {"role": "user", "content": judge_user}],
                    json_mode=supports_json_mode, reasoning_effort="none", experiment=experiment, role="judge")
                judge_json = judge_resp["choices"][0]["message"]["content"] or "{}"
                try:
                    json.loads(judge_json)
                except json.JSONDecodeError as je:
                    # Visibility only: a malformed judge JSON degrades this
                    # one question's synthesis evidence quality, it must not
                    # abort the run.
                    print(f"eval.fuse_judge_json_invalid question_id={question_id!r} "
                          f"judge_model={judge_model!r} error={str(je)!r}", file=sys.stderr)

                synth_user = f"{SYNTHESIS_FALLBACK_PROMPT}\n\n{evidence}\n\nJudge analysis JSON:\n{judge_json}"
                synth_resp = call_api(api_url, synthesis_model,
                                       [{"role": "user", "content": instruction},
                                        {"role": "user", "content": synth_user}],
                                       reasoning_effort="high", experiment=experiment, role="synthesis")
                synth_message = synth_resp["choices"][0]["message"]
                row_out = {**base, "panel_models": panel_models, "content": synth_message["content"],
                           "reasoning": synth_message.get("reasoning_content"), "judge_json": judge_json}
            except Exception as e:
                print(f"eval.fuse_question_failed question_id={question_id!r} error={str(e)!r}", file=sys.stderr)
                row_out = {**base, "panel_models": panel_models, "failed": True, "error": str(e)}
            f.write(json.dumps(row_out) + "\n")
    return out_path


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--judge-model", required=True)
    p.add_argument("--synthesis-model", required=True)
    p.add_argument("--api-url", default="http://localhost:8000")
    p.add_argument("--input", required=True, help="A question file (see eval.sample.py).")
    p.add_argument("--panels", required=True, help="Comma-separated eval.panel.py output files.")
    p.add_argument("--out", required=True, help="Always overwritten.")
    p.add_argument("--experiment", default=None,
                    help="Names call_logs/<experiment>__judge__<model>.jsonl and __synthesis__<model>.jsonl.")
    args = p.parse_args()
    panel_paths = [s.strip() for s in args.panels.split(",") if s.strip()]
    if not panel_paths:
        p.error("--panels must name at least one file")
    print(run(args.api_url, args.judge_model, args.synthesis_model, args.input, panel_paths, args.out,
              args.experiment))
