# 0G Fusion Eval — Stage 1 (AlpacaEval, no tools)

Implements `0G-Fusion产品API与评测代码设计方案-第一阶段-AlpacaEval.md`.

Eval code only talks HTTP to the fusion API (`mock_fusion_api/`) — it never imports
panel/judge/synthesis logic directly. When the real 0G product API exists, only
`--fusion-url` changes; no eval code changes.

## Run self-tests (no network / API keys needed)

```
python3 tests.py
```

## Run for real

```
export ZG_UPSTREAM_BASE_URL=https://your-openai-compatible-provider/v1
export ZG_UPSTREAM_API_KEY=sk-...
export ZG_PANEL_MODELS=model-a,model-b,model-c
export ZG_JUDGE_MODEL=judge-model
export ZG_SYNTHESIS_MODEL=synthesis-model

python3 -m mock_fusion_api.server 8000 &
python3 -m eval.run_eval --fusion-model 0g/fusion-preview --baseline-model some-baseline-model
python3 -m eval.grade eval/results/run_<timestamp>.jsonl
```

Without `ZG_UPSTREAM_BASE_URL` set, every LLM call returns a deterministic fake
response (`FAKE` mode in `pipeline.py`) — useful for wiring/dev, not for real scores.

`allow_tool_call_output` (request field, default `false`) gates whether the panel and
synthesis stages may see caller `tools` and emit a `tool_call`. This eval always sends
`false`. The judge stage never receives tools, regardless of this flag.
