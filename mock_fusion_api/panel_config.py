"""Fusion pipeline config: panel/judge/synthesis model IDs + built-in fallback prompts.
Mirrors fusion.go's fusionPrometheus20Panel + fusionBuiltInPrompts fallback text."""
import os

PANEL_MODELS = [m.strip() for m in os.environ.get("ZG_PANEL_MODELS", "panel-a,panel-b,panel-c").split(",") if m.strip()]
JUDGE_MODEL = os.environ.get("ZG_JUDGE_MODEL", "judge-model")
SYNTHESIS_MODEL = os.environ.get("ZG_SYNTHESIS_MODEL", "synthesis-model")

JUDGE_SYSTEM = (
    "You are the 0G Fusion judge. Compare panel responses and return compact JSON "
    "with keys consensus, contradictions, partial_coverage, unique_insights, "
    "blind_spots, and final_guidance. Do not write the final answer. Return only "
    "JSON; do not include chain-of-thought, hidden reasoning, or <think> blocks."
)
PANEL_FALLBACK_PROMPT = "Answer the request independently and return only the visible answer."
SYNTHESIS_FALLBACK_PROMPT = (
    "Use the panel answers as evidence and the judge analysis as guidance to "
    "answer the original request. Return only the final visible answer."
)
