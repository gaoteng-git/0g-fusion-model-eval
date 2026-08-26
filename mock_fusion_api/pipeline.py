"""Single-round fusion pipeline: panel (parallel, 1 call each) -> judge (1 call) ->
synthesis (1 call). No tool execution anywhere, no iteration anywhere -- mirrors the
structure of quill-cloud-proxy/enclave-go/cmd/enclave/fusion.go (runFusionPanelObserved /
fusionJudgeRequest / fusionFinalRequest), not its literal code.

allow_tool_call_output (default False) gates whether panel/synthesis may see caller
tools and emit a tool_call. The judge stage never gets tools, regardless of the flag.

Thinking (GPQA round): panel and synthesis are called with reasoning_effort=high
(REASONING_ON); the judge is explicitly called with reasoning_effort=none
(REASONING_OFF) and its content is defensively stripped of any <think> block
before being treated as JSON -- see llm_client.extract_thinking. Panel evidence
fed to judge/synthesis includes each member's reasoning AND its final answer,
not content alone.

Judge JSON validity is not enforced (see JUDGE_MODELS_WITHOUT_JSON_MODE) --
run_judge validates it with json.loads() purely to print a clear stderr
warning naming the question (request["question_id"], if the eval harness set
one) and judge model on failure, then continues into synthesis regardless. A
malformed judge JSON degrades one question's evidence quality; it must not
abort the run.

Cost-saving reuse: a request may pass `cached_panel` (already-computed panel
entries) plus `extra_panel_models` (model IDs to actually call fresh) instead
of `model: "0g/fusion*"` triggering a full fresh panel -- see run_fusion's
docstring and eval/panel.py, which uses this (plus `panel_only` below) to
build/extend a panel file without ever paying for judge+synthesis, and
eval/fuse.py, which runs judge+synthesis against such a file with 0 fresh
panel calls.

Call logging: every call_llm() invocation here is tagged with a `role`
("panel"/"judge"/"synthesis" for the fusion path, "baseline" for the plain
passthrough path) and forwards request["experiment"] (set by the eval
harness, e.g. eval.panel's --experiment) straight through. llm_client writes
the full request+response of each call to
call_logs/<experiment>__<role>__<model>.jsonl -- see llm_client.py's
_log_call. No experiment name -> no log files (keeps tests/dev calls quiet).
"""
import json
import sys
from concurrent.futures import ThreadPoolExecutor

from llm_client import call_llm, extract_thinking, REASONING_ON, REASONING_OFF

from . import panel_config as cfg


def _tools_for(tools, allow):
    return tools if (allow and tools) else None


def _panel_messages(messages, index, allowed_tools):
    # FINAL_LETTER_INSTRUCTION is deliberately NOT re-added here: it's already
    # baked into the caller's own question text (see eval/gpqa_tasks.py), the
    # only place it can live and still reach the baseline model too (which
    # never goes through this function). Re-adding it here would (a) literally
    # duplicate the exact same instruction the model already sees in the user
    # turn, and (b) wrongly hardcode a GPQA-specific "this is multiple choice"
    # assumption into this otherwise task-agnostic pipeline code.
    system = f"You are 0G Fusion panel member {index + 1}.\n\n{cfg.PANEL_FALLBACK_PROMPT}"
    if allowed_tools:
        system += "\n\nIf the next correct step is a provided function call, emit the tool call directly instead of describing it."
    return [{"role": "system", "content": system}] + messages


def run_panel(messages, tools, allow_tool_call_output, experiment=None, models=None, start_index=0):
    """Every panel model is called exactly once, concurrently, with thinking on.
    No iteration, no tool execution.

    `models` overrides cfg.PANEL_MODELS -- used when only a subset of the
    panel needs to be called fresh this round (see run_fusion's
    cached_panel/extra_panel_models handling below). `start_index` offsets
    the "panel member N" system-prompt numbering so a partially-cached call
    still numbers the freshly-called members the way a from-scratch run of
    the full panel would (member numbering is per-model-call context only;
    it does not leak into judge/synthesis, which number panel_evidence
    entries independently by their position in the final merged list)."""
    models = cfg.PANEL_MODELS if models is None else models
    def one(item):
        offset, model = item
        allowed_tools = _tools_for(tools, allow_tool_call_output)
        msg = call_llm(model, _panel_messages(messages, start_index + offset, allowed_tools), allowed_tools,
                        reasoning_effort=REASONING_ON, experiment=experiment, role="panel")
        reasoning, content = extract_thinking(msg)
        return {"model": model, "content": content, "reasoning": reasoning, "tool_calls": msg.get("tool_calls")}
    if not models:
        return []
    with ThreadPoolExecutor(max_workers=len(models)) as pool:
        return list(pool.map(one, enumerate(models)))


def tool_calls_text(tool_calls):
    if not tool_calls:
        return ""
    parts = [f'{tc["function"]["name"]}({tc["function"].get("arguments") or "{}"})' for tc in tool_calls]
    return "Proposed tool call(s): " + ", ".join(parts)


def panel_evidence(panel_results):
    """Formats each panel member's REASONING and FINAL ANSWER as evidence --
    both parts, not content alone -- for the judge and synthesis stages.

    Deliberately anonymized: entries are labeled only by position ([1]/[2]/...),
    never by model name, even though production's fusionPanelEvidence does
    include `model=...`. Production has its own reasons for that (letting
    judge/synthesis weight known per-provider domain strengths); for THIS
    eval's purpose -- cleanly comparing whether a candidate model is a better
    5th panel member -- leaking model identity into the judge/synthesis
    prompt risks a real confound: any brand/provider bias baked into the
    judge or synthesis model's own training would then contaminate the
    comparison, independent of actual answer quality. The real model name is
    still preserved in the returned panel_results / 0g_fusion.panel / call
    logs for debugging -- only the text actually sent to judge/synthesis
    drops it."""
    lines = ["Panel answers:"]
    for i, r in enumerate(panel_results):
        answer = (r["content"] or "").strip()
        tc = tool_calls_text(r.get("tool_calls"))
        if tc:
            answer = f"{answer}\n{tc}" if answer else tc
        reasoning = (r.get("reasoning") or "").strip()
        block = f"Reasoning:\n{reasoning}\n\nFinal answer:\n{answer}" if reasoning else answer
        lines.append(f"\n[{i + 1}]\n{block}\n")
    return "\n".join(lines)


def _messages_text(messages):
    return "\n".join(f"{m['role'].upper()}: {m['content']}" for m in messages if m.get("content"))


def run_judge(messages, panel_results, experiment=None, question_id=None):
    """One call. tools is always None here -- unaffected by allow_tool_call_output.
    Thinking is explicitly requested OFF (reasoning_effort=none); any leaked
    <think> block is stripped defensively before the content is treated as the
    judge's JSON -- a judge that thinks anyway must not have its reasoning
    silently corrupt the JSON payload downstream stages parse.

    json_mode (response_format: json_object) is only sent when JUDGE_MODEL is
    NOT in cfg.JUDGE_MODELS_WITHOUT_JSON_MODE -- 0g-router hard-rejects
    response_format for any model that doesn't advertise it as a supported
    parameter (400 model_not_capable), independent of reasoning_effort or
    anything else in the request. For those models, JSON compliance falls
    back to JUDGE_SYSTEM's own "Return only JSON" instruction alone -- no
    stricter than that, by design (see JUDGE_MODELS_WITHOUT_JSON_MODE's
    comment for why this is an intentional match to production's own
    minimax-m3 judge, not an oversight).

    Because that fallback has no structural guarantee, the output is
    validated with json.loads() purely for VISIBILITY: on failure this prints
    a single clear stderr line naming the question and judge model, but still
    returns the raw text and lets the pipeline continue into synthesis
    unchanged -- a malformed judge JSON degrades that one question's evidence
    quality, it must not abort the run. question_id (set by the eval harness,
    e.g. eval.panel's task["question_id"]) is only used for this message; a
    caller that never sets it (e.g. debug_fusion_call.py) just gets `None` in
    the log line instead of a real id."""
    evidence = panel_evidence(panel_results).split("Panel answers:\n", 1)[-1]
    user = f"Original request summary:\n{_messages_text(messages)}\n\nPanel responses:\n{evidence}"
    supports_json_mode = (cfg.JUDGE_MODEL or "").strip().lower() not in cfg.JUDGE_MODELS_WITHOUT_JSON_MODE
    msg = call_llm(cfg.JUDGE_MODEL,
                    [{"role": "system", "content": cfg.JUDGE_SYSTEM}, {"role": "user", "content": user}],
                    json_mode=supports_json_mode, reasoning_effort=REASONING_OFF, experiment=experiment, role="judge")
    _, clean_content = extract_thinking(msg)
    judge_json = clean_content or "{}"
    try:
        json.loads(judge_json)
    except json.JSONDecodeError as e:
        print(
            f"fusion.judge_json_invalid question_id={question_id!r} judge_model={cfg.JUDGE_MODEL!r} "
            f"error={str(e)!r} raw_len={len(judge_json)}",
            file=sys.stderr,
        )
    return judge_json


def run_synthesis(messages, panel_results, judge_json, tools, allow_tool_call_output, experiment=None):
    """One call, with thinking on. tools passed through only if allow_tool_call_output
    is True. Reads the panel's reasoning AND final answers (via panel_evidence),
    not just their answers."""
    allowed_tools = _tools_for(tools, allow_tool_call_output)
    # Same reasoning as _panel_messages: not re-adding FINAL_LETTER_INSTRUCTION
    # here -- `messages` (appended below) already carries the caller's
    # original question, which already has it baked in.
    instruction = cfg.SYNTHESIS_FALLBACK_PROMPT
    if allowed_tools:
        instruction += ("\n\nIf the next correct action is a provided function call, emit the tool "
                         "call directly instead of describing it in text. Return visible text only "
                         "when no tool call is needed.")
    user = f"{instruction}\n\n{panel_evidence(panel_results)}\n\nJudge analysis JSON:\n{judge_json}"
    msg = call_llm(cfg.SYNTHESIS_MODEL, messages + [{"role": "user", "content": user}], allowed_tools,
                    reasoning_effort=REASONING_ON, experiment=experiment, role="synthesis")
    reasoning, content = extract_thinking(msg)
    return {"content": content, "reasoning": reasoning, "tool_calls": msg.get("tool_calls")}


def _validate_cached_panel(cached_panel):
    for i, entry in enumerate(cached_panel):
        missing = [k for k in ("model", "content") if k not in entry]
        if missing:
            raise ValueError(f"cached_panel[{i}] missing required key(s) {missing}: got keys {list(entry)}")


def run_fusion(request):
    """panel (parallel x1, thinking on) -> judge (x1, thinking off) ->
    synthesis (x1, thinking on). No loops anywhere.

    Partial-reuse mode: if the request carries `cached_panel` and/or
    `extra_panel_models`, cfg.PANEL_MODELS is NOT consulted at all -- the
    caller fully controls this call's panel composition. `cached_panel` is a
    list of already-computed panel entries (same shape run_panel produces:
    model/content/reasoning/tool_calls) that are used as-is, no LLM call
    made for them; `extra_panel_models` are model IDs actually called fresh
    this round. Passing ALL desired members as `cached_panel` with no
    `extra_panel_models` makes 0 panel calls -- see eval/fuse.py, which runs
    judge+synthesis against an already-built eval/panel.py file this way.
    `panel_only` (below) stops before judge+synthesis entirely -- that's
    what eval/panel.py itself uses to build/extend a panel file for free."""
    messages = request["messages"]
    tools = request.get("tools")
    allow = bool(request.get("allow_tool_call_output", False))
    experiment = request.get("experiment")
    question_id = request.get("question_id")
    cached_panel = request.get("cached_panel")
    extra_panel_models = request.get("extra_panel_models")
    if cached_panel is not None or extra_panel_models is not None:
        cached_panel = list(cached_panel or [])
        _validate_cached_panel(cached_panel)
        fresh = run_panel(messages, tools, allow, experiment=experiment,
                           models=extra_panel_models or [], start_index=len(cached_panel))
        panel_results = cached_panel + fresh
    else:
        panel_results = run_panel(messages, tools, allow, experiment=experiment)

    if request.get("panel_only"):
        # Eval-only cost-saving extension, not a real product feature: stop
        # right after the panel, paying nothing for judge+synthesis. A real
        # caller always wants a final answer, so this never fires outside
        # the eval harness -- see eval/panel.py, which is the only caller.
        return {"0g_fusion": {"panel": panel_results}}

    judge_json = run_judge(messages, panel_results, experiment=experiment, question_id=question_id)
    final = run_synthesis(messages, panel_results, judge_json, tools, allow, experiment=experiment)
    tool_calls = final.get("tool_calls")
    message = {"role": "assistant", "content": final.get("content"), "tool_calls": tool_calls}
    if final.get("reasoning"):
        message["reasoning_content"] = final["reasoning"]
    return {
        "choices": [{"message": message, "finish_reason": "tool_calls" if tool_calls else "stop"}],
        "0g_fusion": {"panel": panel_results, "judge_json": judge_json},
    }


def handle_chat_completion(request):
    """Route like isFusionModel() in production: only 0g/fusion* model IDs enter the
    fusion pipeline; anything else is a single plain passthrough call (used for
    baselines). The passthrough forwards the caller's reasoning_effort as-is (the
    eval harness sets it explicitly for baseline calls) and normalizes the
    response the same way the fusion pipeline does -- clean `content` +
    separate `reasoning_content` when the model thinks -- so grading code
    never needs a MiniMax-specific special case."""
    if (request.get("model") or "").startswith("0g/fusion"):
        return run_fusion(request)
    allow = bool(request.get("allow_tool_call_output", False))
    msg = call_llm(request["model"], request["messages"], _tools_for(request.get("tools"), allow),
                    reasoning_effort=request.get("reasoning_effort"),
                    experiment=request.get("experiment"), role="baseline")
    reasoning, content = extract_thinking(msg)
    tool_calls = msg.get("tool_calls")
    message = {"role": "assistant", "content": content, "tool_calls": tool_calls}
    if reasoning:
        message["reasoning_content"] = reasoning
    return {"choices": [{"message": message, "finish_reason": "tool_calls" if tool_calls else "stop"}]}
