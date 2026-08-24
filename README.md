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
export ZG_PANEL_MODELS=minimax-m3,kimi-k3,glm-5.2,deepseek-v4-pro,<5th-candidate>
export ZG_JUDGE_MODEL=minimax-m3
export ZG_SYNTHESIS_MODEL=kimi-k3

python3 -m mock_fusion_api.server 8000 &
python3 -m eval.run_eval --fusion-model 0g/fusion-preview --baseline-model gpt-5.6-sol
python3 -m eval.gpqa_grade eval/results/run_<timestamp>.jsonl
```

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
