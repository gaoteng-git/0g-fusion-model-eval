# 0G Fusion Eval — GPQA round (no tools, thinking on except for the judge)

Five small, single-purpose tools, each a pure function of its explicit
inputs: given a question file and (usually) a model name, compute an answer
and write a complete output file. No resume, no `--limit`, no implicit
state anywhere — every run OVERWRITES its `--out` completely. This is
deliberate: the entire class of bugs a stateful "resume/reuse" design
invites (mismatched schemas, stale cached data, silent truncation on
interrupt, cross-file merge conflicts) simply has nowhere to live when
there's no persistent state to get wrong.

- **`eval.sample`** extracts a subset of a question file by row index into
  a new file, tagging each row with its ORIGINAL absolute index. This is
  the *only* way to control "which questions" anywhere in this harness —
  there is no `--limit` flag on any other tool. Want to debug with 5
  questions, then later run the rest and combine results? Sample 0-4 first;
  once that's paying off, sample the remaining indices, run everything
  again, and `cat`/merge the two result files — they'll never collide,
  because both carry the original index, not a position-in-file counter.
- **`eval.panel`** computes ONE model's answers over a question file.
  Building a 5-member panel means running this 5 times, once per model,
  each with its own `--out`.
- **`eval.fuse`** runs judge+synthesis over N already-built panel files
  (typically 5: the fixed members' files + one candidate's) for every
  question in `--input`. `--judge-model`/`--synthesis-model` are explicit
  arguments here, not global config.
- **`eval.baseline`** computes ONE baseline model's answers over a question
  file — mechanically identical to `eval.panel`, just a different name for
  what the row means.
- **`eval.grade`** scores 1+ files, each completely independently — nothing
  is ever merged across files, so there is nothing for two files to
  disagree about.

Thinking behaviour (see `md_files/0g-router-7模型thinking实测结论.md` for the
real-world probing this is based on): panel, synthesis, and baseline calls
are all made with `reasoning_effort: "high"`; the judge is explicitly called
with `reasoning_effort: "none"` and has any leaked `<think>` block stripped
defensively before its content is parsed as JSON. Judge and synthesis read
each panel member's reasoning *and* final answer (not answers alone) via
`panel_evidence`.

Grading is exact-match on the model's required `Final Answer: <letter>` line
against the dataset's ground truth — no LLM judge needed for GPQA.

## Run self-tests (no network / API keys needed)

```
python3 tests.py
```

## Run for real

```
export ZG_UPSTREAM_BASE_URL=https://your-openai-compatible-provider/v1
export ZG_UPSTREAM_API_KEY=sk-...

python3 -m mock_fusion_api.server 8000 &

# 0. Pick which questions this run covers -- everywhere else in this
#    harness, "which questions" is decided entirely by which file you point
#    --input at. A 5-question smoke test:
python3 -m eval.sample --input /path/to/gpqa_diamond.jsonl --indices 0-4 \
    --out eval/samples/smoke.jsonl

# 1. The 4 fixed panel members, one invocation each -- no judge/synthesis
#    call happens anywhere in this step, it only ever costs the panel
#    models themselves.
for m in minimax-m3 kimi-k3 glm-5.2 deepseek-v4-pro; do
  python3 -m eval.panel --model "$m" --api-url http://localhost:8000 \
      --input eval/samples/smoke.jsonl --out "eval/results/panel-$m.jsonl" \
      --experiment gpqa-panel
done

# 2. A candidate 5th member, same way.
python3 -m eval.panel --model hy3 --api-url http://localhost:8000 \
    --input eval/samples/smoke.jsonl --out eval/results/panel-hy3.jsonl \
    --experiment gpqa-panel

# 3. Fuse the 4 fixed + this one candidate into a real, scored result --
#    the ONLY step that costs judge+synthesis money.
python3 -m eval.fuse --judge-model minimax-m3 --synthesis-model kimi-k3 \
    --api-url http://localhost:8000 --input eval/samples/smoke.jsonl \
    --panels eval/results/panel-minimax-m3.jsonl,eval/results/panel-kimi-k3.jsonl,\
eval/results/panel-glm-5.2.jsonl,eval/results/panel-deepseek-v4-pro.jsonl,eval/results/panel-hy3.jsonl \
    --out eval/results/fuse-hy3.jsonl --experiment gpqa-fuse-hy3

# 4. Baselines are completely independent -- no panel/fusion file needed.
python3 -m eval.baseline --model gpt-5.6-sol --api-url http://localhost:8000 \
    --input eval/samples/smoke.jsonl --out eval/results/baseline-gpt.jsonl \
    --experiment gpqa-baselines

# 5. Grade everything -- each file scored on its own, side by side.
python3 -m eval.grade eval/results/fuse-hy3.jsonl eval/results/baseline-gpt.jsonl
```

To test several candidate 5th members, repeat steps 2-3 once per candidate
(new `--out`/`--experiment` each time) — the 4 fixed panel files from step 1
are read, never recomputed. To run the full question set once the smoke
test looks right, `eval.sample --indices` covering everything (or skip
sampling and pass the real file straight to `--input`) and re-run the same
steps; nothing here resumes, so this is a fresh, complete run, not a
continuation. To extend an existing run with more questions without
re-paying for what's already done, sample the *additional* indices into a
new file, run steps 1-4 against just that file, and append (`cat`) the new
output onto the old one — `eval.panel`/`eval.fuse`/`eval.baseline` never
truncate a file that isn't their own `--out`, and every row carries its own
stable `question_id`, so two files sampled from disjoint index ranges merge
by simple concatenation with no collisions.

Without `ZG_UPSTREAM_BASE_URL` set, every LLM call returns a deterministic
fake response (`FAKE` mode in `llm_client.py`) that reproduces the three
real-world thinking patterns found by probing 0g-router — a model name
containing "minimax" always thinks with thinking inline in `content`
(`<think>...</think>`), one containing "hy3" only thinks when
`reasoning_effort` is set, everything else always thinks via a separate
`reasoning_content` field — useful for wiring/dev, not for real scores.

## Row shapes

`eval.panel`/`eval.baseline` write one row per question:
```json
{"question_id": 0, "instruction": "...", "correct_letter": "B", "model": "minimax-m3", "content": "...", "reasoning": "..."}
```
(or `{"failed": true, "error": "..."}` in place of `content`/`reasoning` on
a call that didn't succeed — the run continues to the next question either
way, never aborting over one failure).

`eval.fuse` writes:
```json
{"question_id": 0, "instruction": "...", "correct_letter": "B",
 "judge_model": "minimax-m3", "synthesis_model": "kimi-k3",
 "panel_models": ["minimax-m3", "kimi-k3", "glm-5.2", "deepseek-v4-pro", "hy3"],
 "content": "...", "reasoning": "...", "judge_json": "{...}"}
```
`instruction`/`correct_letter` are recomputed fresh from `--input` (the same
deterministic way `eval.panel` computed them when it built each `--panels`
file) — `--panels` files are only ever consulted for their `(model,
content, reasoning)` answers, keyed by `question_id`. If any `--panels`
file is missing (or has `"failed": true` for) a question that's in
`--input`, that ONE question is written as failed, naming which file was
short, and the run continues — no judge/synthesis call is made for it, so
an incomplete panel costs nothing beyond itself.

## GPQA task data

`eval/gpqa_tasks.py` expects the official GPQA columns (`Question`,
`Correct Answer`, `Incorrect Answer 1/2/3` — from
https://github.com/idavidrein/gpqa, gated; do not commit real questions to a
public repo, per its own anti-leakage terms). `eval/data/gpqa_sample.jsonl`
is a small set of made-up placeholder questions in the same schema, for
offline testing. Every tool takes an explicit `--input` — there is no
default path anywhere, so pointing at the real download is always a
deliberate choice, never a silent fallback.

`load_questions(path)` returns `(question_id, row)` pairs: `question_id` is
the row's own `question_id` field if present (that's how `eval.sample.py`
tags a row with its original absolute index when extracting a subset), or
its 0-based position in the file otherwise. `format_question(row,
question_id)` seeds the A/B/C/D shuffle by `question_id`, never by file
position — the same question shuffles the same way no matter which file
(the full set, or any sampled subset of it) it's loaded from.

## `mock_fusion_api/` — product-behavior simulation, not part of the eval's own call graph

`mock_fusion_api/pipeline.py` simulates what a genuine `model: "0g/fusion*"`
request would do server-side in the real product (panel → judge →
synthesis, `cached_panel`/`extra_panel_models`/`panel_only` reuse modes) —
useful for testing that assumption ahead of the real endpoint shipping, and
still covered by `tests.py` as such. None of `eval.panel`/`eval.fuse`/
`eval.baseline` send a `"0g/fusion*"` request, though: each calls one
specific real model directly through the same plain, non-fusion passthrough
`--api-url` route (`handle_chat_completion`'s branch for any other model
ID), and `eval.fuse` does its own panel-evidence/judge/synthesis
orchestration client-side (reusing `panel_evidence`/`JUDGE_SYSTEM`/
`JUDGE_MODELS_WITHOUT_JSON_MODE`/`SYNTHESIS_FALLBACK_PROMPT` from
`mock_fusion_api`, since those are the same prompts a real fusion call would
use, but issuing its own HTTP calls rather than sending one and letting the
server orchestrate). `ZG_PANEL_MODELS`/`ZG_JUDGE_MODEL`/`ZG_SYNTHESIS_MODEL`
configure that simulation's own defaults; they have no effect on any
`eval.*` command, all of which take their model names as explicit
arguments.

`allow_tool_call_output` (request field, default `false`) still gates
whether the panel/synthesis stages of that simulation may see caller
`tools` and emit a `tool_call` — unrelated to thinking, and unused by this
GPQA round (no tools are sent, and none of the eval CLI tools accept one).

## Per-call logging

Every call is logged in full — the entire request body and the entire raw
response payload (not just the message: `usage`, `finish_reason`, `id`, the
actually-served model too) — to
`call_logs/<experiment>__<role>__<model>.jsonl`, one JSON line appended per
call. Logging only fires when the caller passes an `experiment` name
(`--experiment` on any of the four tools); calls with no experiment name
(tests, ad-hoc pokes) write nothing. `role` is `"panel"`/`"judge"`/
`"synthesis"`/`"baseline"` depending on which tool made the call — a panel
model, the judge model, the synthesis model, and a baseline model are,
mechanically, all just the same passthrough call with a different `role`.
`call_logs/` is gitignored — for a real run these files contain real GPQA
question text end to end, same anti-leakage handling as `eval/data/*` and
`eval/samples/*`.
