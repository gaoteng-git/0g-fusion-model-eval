"""Single-shot OpenAI-compatible chat completion call. Shared by mock_fusion_api
(the product pipeline) and eval/grade.py (the grading judge) as a neutral low-level
utility -- neither side imports the other's logic through it."""
import json
import os
import urllib.request

UPSTREAM_BASE_URL = os.environ.get("ZG_UPSTREAM_BASE_URL")
UPSTREAM_API_KEY = os.environ.get("ZG_UPSTREAM_API_KEY", "")
FAKE = not UPSTREAM_BASE_URL  # no upstream configured -> deterministic offline stand-in


def call_llm(model, messages, tools=None, json_mode=False):
    """One single-shot chat completion. No retries, no loop, no tool execution."""
    if FAKE:
        return _fake_llm(model, messages, tools, json_mode)
    body = {"model": model, "messages": messages}
    if tools:
        body["tools"] = tools
    if json_mode:
        body["response_format"] = {"type": "json_object"}
    req = urllib.request.Request(
        UPSTREAM_BASE_URL.rstrip("/") + "/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Authorization": f"Bearer {UPSTREAM_API_KEY}", "Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        payload = json.loads(resp.read())
    return payload["choices"][0]["message"]


def _fake_llm(model, messages, tools, json_mode):
    """Deterministic stand-in so the pipeline runs/tests without network or API keys."""
    user_text = next((m["content"] for m in reversed(messages) if m.get("role") == "user"), "")
    if json_mode:
        content = json.dumps({
            "consensus": "panel members broadly agree", "contradictions": "none noted",
            "partial_coverage": "", "unique_insights": "", "blind_spots": "",
            "final_guidance": "synthesize a direct answer",
        })
        return {"content": content, "tool_calls": None}
    if tools and os.environ.get("ZG_FAKE_FORCE_TOOL_CALL") == model:
        fn = tools[0]["function"]["name"]
        return {"content": None, "tool_calls": [{"id": "call_1", "type": "function",
                                                  "function": {"name": fn, "arguments": "{}"}}]}
    return {"content": f"[fake:{model}] {user_text[:60]}", "tool_calls": None}
