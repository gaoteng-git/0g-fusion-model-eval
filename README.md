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
python3 -m eval.run_eval --fusion-model 0g/fusion-preview \
    --baseline-model gpt-5.6-sol,claude-fable-5 --experiment gpqa-main
python3 -m eval.gpqa_grade eval/results/gpqa-main.jsonl
```

`--baseline-model` is a comma-separated list of 0+ models, all called for every
question alongside fusion (pass `""` for fusion-only, no baselines). Output
defaults to `eval/results/<experiment>.jsonl` — re-running the same
`--experiment` (e.g. after a `--limit 5` smoke test, now without `--limit`)
resumes into that file: already-succeeded questions are reused, not re-called.
`--no-resume` overwrites from scratch instead. A smaller `--limit` on a later
run only narrows what gets *called* — prior rows outside that window stay in
the file rather than being deleted.

Resume is per-question, not per-baseline: a reused row keeps whatever baseline
entries it already has, so a baseline model that failed (or one added to
`--baseline-model` afterwards) is not filled in by re-running `run_eval.py` —
it says so on stderr and names the model. `eval/run_baseline.py` is what adds
or retries one baseline on an already-completed run without re-paying for
fusion; point its `--out` at the same file to accumulate in place.

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
fixed ones (e.g. 3 MiMo-candidate variants), don't re-pay for the 4 unchanged
panel calls each time. `run_fusion` accepts two extra request fields:

- `cached_panel`: a list of already-computed panel entries (the shape
  `run_panel` produces: `model`/`content`/`reasoning`/`tool_calls`), used
  as-is with no LLM call made for them.
- `extra_panel_models`: model IDs actually called fresh this round.

When either is present, `cfg.PANEL_MODELS` is ignored entirely — the caller
fully controls the panel composition for that call. Judge + synthesis then
run once, over the merged (cached + fresh) panel, same as always.

`eval/run_variant.py` drives this from a prior `run_eval.py` replay file
(whose `fusion.raw_response["0g_fusion"]["panel"]` already holds the full
per-question panel breakdown — nothing extra needs capturing):

```
python3 -m eval.run_eval --out eval/results/base.jsonl        # once, with the 4 fixed panel models
python3 -m eval.run_variant --base-replay eval/results/base.jsonl \
    --variant-model xiaomi/mimo-v2.5-pro --out eval/results/variant_mimo.jsonl --fixed-count 4
python3 -m eval.gpqa_grade eval/results/variant_mimo.jsonl
```
Each variant run only pays for 1 panel call + judge + synthesis, not all 5
panel calls again. The baselines list isn't re-called either — it's carried
over unchanged from the base replay row. `--fixed-count` (default 4) aborts
the run immediately, before any calls, if the base run's panel doesn't have
that many members — guards against the base run having been accidentally
configured with a candidate already baked in, which would otherwise make
variant runs silently accumulate panel members instead of doing a clean swap.

## Per-call logging

Every `call_llm()` call (panel/judge/synthesis/baseline) is logged in full —
the entire request body and the entire raw response payload (not just the
message: `usage`, `finish_reason`, `id`, the actually-served model too) — to
`call_logs/<experiment>__<role>__<model>.jsonl`, one JSON line appended per
call. Logging only fires when the caller passes an `experiment` name
(`run_eval.py`/`run_variant.py`'s `--experiment`, defaulted if not given);
calls with no experiment name (tests, ad-hoc pokes) write nothing.
`call_logs/` is gitignored — for a real run these files contain real GPQA
question text end to end, same anti-leakage handling as `eval/data/*`.
