"""Fusion pipeline config: panel/judge/synthesis model IDs + built-in fallback prompts.
Mirrors fusion.go's fusionPrometheus20Panel + fusionBuiltInPrompts fallback text."""
import os

PANEL_MODELS = [m.strip() for m in os.environ.get("ZG_PANEL_MODELS", "panel-a,panel-b,panel-c").split(",") if m.strip()]
JUDGE_MODEL = os.environ.get("ZG_JUDGE_MODEL", "judge-model")
SYNTHESIS_MODEL = os.environ.get("ZG_SYNTHESIS_MODEL", "synthesis-model")

JUDGE_SYSTEM = (
    "You are the 0G Fusion judge. Compare the panel members' reasoning AND their "
    "final answers, and return compact JSON with keys consensus, contradictions, "
    "partial_coverage, unique_insights, blind_spots, and final_guidance. Do not "
    "write the final answer. Return only JSON; do not include chain-of-thought, "
    "hidden reasoning, or <think> blocks."
)
PANEL_FALLBACK_PROMPT = "Answer the request independently and return only the visible answer."
SYNTHESIS_FALLBACK_PROMPT = (
    "Use the panel members' reasoning and final answers as evidence, and the judge "
    "analysis as guidance, to answer the original request."
)

# GPQA-style multiple-choice format instruction. This is a FORMAT requirement
# (so the final letter can be extracted reliably), not a "please think"
# instruction -- real testing (md_files/0g-router-7模型thinking实测结论.md) found
# thinking itself is already reliably produced via reasoning_effort, not via
# prompt wording, for every one of the 7 candidate models.
FINAL_LETTER_INSTRUCTION = (
    "This is a multiple-choice question. After your reasoning, end your response "
    "with exactly one line in this exact format (nothing after it):\n"
    "Final Answer: <letter>"
)
