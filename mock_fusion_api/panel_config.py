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

# Confirmed via 0g-router's own capability catalog (supported_parameters
# omits response_format for minimax-m3) AND a real 400 model_not_capable
# error hit live against router-api.0g.ai, AND MiniMax's own official docs +
# a GitHub feature-request issue (MiniMax-AI/MiniMax-M2.5#4: response_format
# is silently ignored, JSON output can come out malformed) -- minimax-m3
# does not reliably support response_format at all, on 0G's infra or
# MiniMax's own API. If it's ever configured as JUDGE_MODEL, run_judge must
# NOT send json_mode=True (0g-router hard-rejects response_format for any
# model that doesn't advertise it, regardless of reasoning_effort) -- add a
# model's name here rather than passing json_mode unconditionally.
#
# This is a deliberate choice to match production Prometheus 2.0's actual
# judge model (fusionPresetJudgeModelsForModel: minimax-m3 primary,
# kimi-k3 fallback) as closely as possible given a single fixed judge model
# and no failure-retry list -- production's own fusionJudgeRequest also sets
# response_format unconditionally and presumably reaches minimax-m3 through
# a path that doesn't hard-reject it the way 0g-router does, relying on
# JUDGE_SYSTEM's own "Return only JSON" instruction instead.
# Compared case-insensitively (see run_judge) -- 0g-router itself resolves
# model ids case-insensitively (models.yaml: the "MiniMax-M3" alias "differs
# from canonical only by case"), so an exact-case set here would silently
# fail to protect a differently-cased ZG_JUDGE_MODEL value and let the same
# 400 model_not_capable slip back through.
JUDGE_MODELS_WITHOUT_JSON_MODE = {"minimax-m3", "minimax/minimax-m3"}
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
