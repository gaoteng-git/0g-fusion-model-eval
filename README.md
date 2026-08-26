# 0G Fusion Eval — GPQA round (no tools, thinking on except for the judge)

Eval code only talks HTTP to the fusion API (`mock_fusion_api/`) — it never
imports panel/judge/synthesis logic directly. When the real 0G product API
exists, only `--fusion-url` changes; no eval code changes.

Thinking behaviour (see `md_files/0g-router-7模型thinking实测结论.md` for the
real-world probing this is based on): panel, synthesis, and the baseline
model are all called with `reasoning_effort: "high"`; the judge is explicitly
called with `reasoning_effort: "none"` and has any leaked `<think>` block
stripped defensively before its content is parsed as JSON. Judge and
synthesis read each panel member's reasoning *and* final answer (not answers
alone) via `panel_evidence`.

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
export ZG_PANEL_MODELS=minimax-m3,kimi-k3,glm-5.2,deepseek-v4-pro   # the 4 fixed members
export ZG_JUDGE_MODEL=minimax-m3
export ZG_SYNTHESIS_MODEL=kimi-k3

python3 -m mock_fusion_api.server 8000 &

# NOTE: `eval.panel`/`eval.fuse` below always pass the panel explicitly via
# `cached_panel`/`extra_panel_models`, so once they're in use ZG_PANEL_MODELS
# above is never actually consulted (it still matters for a bare
# `{"model": "0g/fusion*"}` request, and for tests.py) -- a typo in one of
# these commands' own `--models` gets no protection from it.

# 1. Build the panel. No judge/synthesis call happens here -- this step only
#    ever costs the panel models themselves.
python3 -m eval.panel --fusion-url http://localhost:8000 \
    --models minimax-m3,kimi-k3,glm-5.2,deepseek-v4-pro --experiment gpqa-panel-fixed4

# 2. Fuse it into a real judge+synthesis result whenever you're ready to pay for one.
python3 -m eval.fuse --fusion-url http://localhost:8000 \
    --panel eval/results/gpqa-panel-fixed4.jsonl --experiment gpqa-fuse-fixed4

# 3. Baselines are completely independent -- no panel/fusion file needed at all.
python3 -m eval.baseline --baseline-url http://localhost:8000 \
    --models gpt-5.6-sol,claude-fable-5 --experiment gpqa-baselines

# 4. Grade fusion + baselines together -- they're two separate files on disk.
python3 -m eval.grade eval/results/gpqa-fuse-fixed4.jsonl eval/results/gpqa-baselines.jsonl
```

Four small, single-purpose tools instead of one monolith, composed like Unix
pipeline stages -- each reads/writes a plain JSONL keyed by `question_id`,
nothing hidden between them:

- **`eval.panel`** computes panel-member answers only, nothing else. `--models`
  is always the FULL desired panel for that file, never "add one more to
  whatever's already there" -- a panel is stated completely, every time, so
  there's nothing to accidentally accumulate. `--reuse <file>` pulls
  already-computed answers from a DIFFERENT file for models present there, so
  extending or swapping a panel only calls what's actually new (see below).
  `--reuse` must point at a real file — a typo'd path is refused, not
  silently treated as empty (which would re-call every model at full price).
- **`eval.fuse`** runs judge+synthesis over an already-built panel file --
  the *only* step that costs judge+synthesis money, kept deliberately
  separate so a panel can be built, inspected, and reused as many times as
  you like before paying for a real result against it.
- **`eval.baseline`** computes 1+ baseline models directly against the
  question set, with no dependency on any panel/fusion file. Repeated calls
  targeting the same `--out` accumulate (add a 2nd baseline, retry a failed
  one) rather than replace.
- **`eval.grade`** takes 1+ files and merges them by `question_id` before
  scoring -- a fuse file and a baseline file never need to be combined into
  one file on disk to be graded together. Refuses to merge two files that
  disagree about what a shared `question_id` actually is (different
  `instruction`/`correct_letter`) — that can only mean the files came from
  different question sets, and grading them together would silently depend
  on which file was passed first.

All four default `--out` to `eval/results/<experiment>.jsonl` and resume into
it: re-running the same `--experiment`/`--out` reuses whatever's already
there and only calls for what's missing or previously failed. The reusable
unit differs per tool, on purpose: `eval.fuse` resumes per *row* (one row is
one atomic fusion answer — nothing smaller to reuse); `eval.panel` and
`eval.baseline` both resume per *model* within a row (`--models
a,b,c` after a row already has `a` only calls `b` and `c`; a `--baseline`
model that failed is retried without disturbing the others in the same row).
`--no-resume` discards and starts over. A smaller `--limit` on a later run
only narrows what gets *called* — prior rows outside that window stay in the
file rather than being deleted. Pointing `--out` at a file a *different* one
of these four tools wrote is refused, not silently accepted and rewritten —
each tool only ever adds/replaces its own fields, so treating someone else's
file as "already done" would otherwise drop whatever it already had (and
already cost real money to produce).

Without `ZG_UPSTREAM_BASE_URL` set, every LLM call returns a deterministic
fake response (`FAKE` mode in `llm_client.py`) that reproduces the three
real-world thinking patterns found by probing 0g-router — a model name
containing "minimax" always thinks with thinking inline in `content`
(`<think>...</think>`), one containing "hy3" only thinks when
`reasoning_effort` is set, everything else always thinks via a separate
`reasoning_content` field — useful for wiring/dev, not for real scores.

## GPQA task data

`eval/gpqa_tasks.py` expects the official GPQA columns (`Question`,
`Correct Answer`, `Incorrect Answer 1/2/3` — from
https://github.com/idavidrein/gpqa, gated; do not commit real questions to a
public repo, per its own anti-leakage terms). `eval/data/gpqa_sample.jsonl`
is a small set of made-up placeholder questions in the same schema, for
offline testing only — swap in the real (CSV or same-shaped JSONL) file via
`load_tasks(path=...)` once downloaded.

`allow_tool_call_output` (request field, default `false`) still gates
whether the panel/synthesis stages may see caller `tools` and emit a
`tool_call` — unrelated to thinking, and unused by this GPQA round (no tools
are sent).

## Reusing fixed panel members across variant runs

If you're testing several candidate 5th panel members against the same 4
fixed ones (e.g. 3 MiMo-candidate variants), `eval.panel --reuse` avoids
re-paying for the 4 unchanged panel calls each time — and, just as
importantly, avoids paying for judge+synthesis at all while doing it, since
`eval.panel` never runs them in the first place:

```
# once: the 4 fixed panel members
python3 -m eval.panel --fusion-url http://localhost:8000 \
    --models minimax-m3,kimi-k3,glm-5.2,deepseek-v4-pro --experiment gpqa-panel-fixed4

# one variant per candidate: reuse the 4 fixed ones, call only the new model
python3 -m eval.panel --fusion-url http://localhost:8000 \
    --models minimax-m3,kimi-k3,glm-5.2,deepseek-v4-pro,hy3 \
    --reuse eval/results/gpqa-panel-fixed4.jsonl --experiment gpqa-panel-hy3

# pay for judge+synthesis only once you actually want a scored result
python3 -m eval.fuse --fusion-url http://localhost:8000 \
    --panel eval/results/gpqa-panel-hy3.jsonl --experiment gpqa-fuse-hy3
python3 -m eval.grade eval/results/gpqa-fuse-hy3.jsonl eval/results/gpqa-baselines.jsonl
```

Under the hood this is the same two request fields as before, now issued by
`eval.panel` on the caller's behalf:

- `cached_panel`: a list of already-computed panel entries (the shape
  `run_panel` produces: `model`/`content`/`reasoning`/`tool_calls`), used
  as-is with no LLM call made for them.
- `extra_panel_models`: model IDs actually called fresh this round.

When either is present, `cfg.PANEL_MODELS` is ignored entirely — the caller
fully controls the panel composition for that call. A third field,
`panel_only`, makes `run_fusion` stop right after the panel instead of
continuing into judge+synthesis — the eval-only mechanism `eval.panel` relies
on to build/extend a panel for free (see `pipeline.run_fusion`'s docstring;
no real product caller ever sets it). `eval.fuse` is what runs judge+synthesis
over the merged (cached + fresh) panel once you actually want a scored result.

## Per-call logging

Every `call_llm()` call (panel/judge/synthesis/baseline) is logged in full —
the entire request body and the entire raw response payload (not just the
message: `usage`, `finish_reason`, `id`, the actually-served model too) — to
`call_logs/<experiment>__<role>__<model>.jsonl`, one JSON line appended per
call. Logging only fires when the caller passes an `experiment` name
(`eval.panel`/`eval.fuse`/`eval.baseline`'s `--experiment`, defaulted from
the command's own arguments if not given); calls with no experiment name
(tests, ad-hoc pokes) write nothing.
`call_logs/` is gitignored — for a real run these files contain real GPQA
question text end to end, same anti-leakage handling as `eval/data/*`.
