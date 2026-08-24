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
"""
from concurrent.futures import ThreadPoolExecutor

from llm_client import call_llm, extract_thinking, REASONING_ON, REASONING_OFF

from . import panel_config as cfg


def _tools_for(tools, allow):
    return tools if (allow and tools) else None


def _panel_messages(messages, index, allowed_tools):
    system = f"You are 0G Fusion panel member {index + 1}.\n\n{cfg.PANEL_FALLBACK_PROMPT}"
    system += "\n\n" + cfg.FINAL_LETTER_INSTRUCTION
    if allowed_tools:
        system += "\n\nIf the next correct step is a provided function call, emit the tool call directly instead of describing it."
    return [{"role": "system", "content": system}] + messages


def run_panel(messages, tools, allow_tool_call_output):
    """Every panel model is called exactly once, concurrently, with thinking on.
    No iteration, no tool execution."""
    def one(item):
        index, model = item
        allowed_tools = _tools_for(tools, allow_tool_call_output)
        msg = call_llm(model, _panel_messages(messages, index, allowed_tools), allowed_tools,
                        reasoning_effort=REASONING_ON)
        reasoning, content = extract_thinking(msg)
        return {"model": model, "content": content, "reasoning": reasoning, "tool_calls": msg.get("tool_calls")}
    with ThreadPoolExecutor(max_workers=len(cfg.PANEL_MODELS)) as pool:
        return list(pool.map(one, enumerate(cfg.PANEL_MODELS)))


def tool_calls_text(tool_calls):
    if not tool_calls:
        return ""
    parts = [f'{tc["function"]["name"]}({tc["function"].get("arguments") or "{}"})' for tc in tool_calls]
    return "Proposed tool call(s): " + ", ".join(parts)


def panel_evidence(panel_results):
    """Formats each panel member's REASONING and FINAL ANSWER as evidence --
    both parts, not content alone -- for the judge and synthesis stages."""
    lines = ["Panel answers:"]
    for i, r in enumerate(panel_results):
        answer = (r["content"] or "").strip()
        tc = tool_calls_text(r.get("tool_calls"))
        if tc:
            answer = f"{answer}\n{tc}" if answer else tc
        reasoning = (r.get("reasoning") or "").strip()
        block = f"Reasoning:\n{reasoning}\n\nFinal answer:\n{answer}" if reasoning else answer
        lines.append(f"\n[{i + 1}] model={r['model']}\n{block}\n")
    return "\n".join(lines)


def _messages_text(messages):
    return "\n".join(f"{m['role'].upper()}: {m['content']}" for m in messages if m.get("content"))


def run_judge(messages, panel_results):
    """One call. tools is always None here -- unaffected by allow_tool_call_output.
    Thinking is explicitly requested OFF (reasoning_effort=none); any leaked
    <think> block is stripped defensively before the content is treated as the
    judge's JSON -- a judge that thinks anyway must not have its reasoning
    silently corrupt the JSON payload downstream stages parse."""
    evidence = panel_evidence(panel_results).split("Panel answers:\n", 1)[-1]
    user = f"Original request summary:\n{_messages_text(messages)}\n\nPanel responses:\n{evidence}"
    msg = call_llm(cfg.JUDGE_MODEL,
                    [{"role": "system", "content": cfg.JUDGE_SYSTEM}, {"role": "user", "content": user}],
                    json_mode=True, reasoning_effort=REASONING_OFF)
    _, clean_content = extract_thinking(msg)
    return clean_content or "{}"


def run_synthesis(messages, panel_results, judge_json, tools, allow_tool_call_output):
    """One call, with thinking on. tools passed through only if allow_tool_call_output
    is True. Reads the panel's reasoning AND final answers (via panel_evidence),
    not just their answers."""
    allowed_tools = _tools_for(tools, allow_tool_call_output)
    instruction = cfg.SYNTHESIS_FALLBACK_PROMPT + "\n\n" + cfg.FINAL_LETTER_INSTRUCTION
    if allowed_tools:
        instruction += ("\n\nIf the next correct action is a provided function call, emit the tool "
                         "call directly instead of describing it in text. Return visible text only "
                         "when no tool call is needed.")
    user = f"{instruction}\n\n{panel_evidence(panel_results)}\n\nJudge analysis JSON:\n{judge_json}"
    msg = call_llm(cfg.SYNTHESIS_MODEL, messages + [{"role": "user", "content": user}], allowed_tools,
                    reasoning_effort=REASONING_ON)
    reasoning, content = extract_thinking(msg)
    return {"content": content, "reasoning": reasoning, "tool_calls": msg.get("tool_calls")}


def run_fusion(request):
    """panel (parallel x1, thinking on) -> judge (x1, thinking off) ->
    synthesis (x1, thinking on). No loops anywhere."""
    messages = request["messages"]
    tools = request.get("tools")
    allow = bool(request.get("allow_tool_call_output", False))
    panel_results = run_panel(messages, tools, allow)
    judge_json = run_judge(messages, panel_results)
    final = run_synthesis(messages, panel_results, judge_json, tools, allow)
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
                    reasoning_effort=request.get("reasoning_effort"))
    reasoning, content = extract_thinking(msg)
    tool_calls = msg.get("tool_calls")
    message = {"role": "assistant", "content": content, "tool_calls": tool_calls}
    if reasoning:
        message["reasoning_content"] = reasoning
    return {"choices": [{"message": message, "finish_reason": "tool_calls" if tool_calls else "stop"}]}
