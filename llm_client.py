"""Single-shot OpenAI-compatible chat completion call, plus thinking-content
extraction. Shared by mock_fusion_api (the product pipeline) and eval/*.py
(baseline calls, grading) as a neutral low-level utility -- neither side
imports the other's logic through it.

Thinking/reasoning behaviour is driven by the `reasoning_effort` request field
(0g-router's portable knob -- see docs/design/reasoning-translation.md in
0g-serving-broker; "none"/"minimal" -> off, anything else -> on, absent ->
upstream default). Real-world probing of all 7 candidate panel models (see
md_files/0g-router-7模型thinking实测结论.md) found the reasoning text lands in
one of two places depending on the model:
  - a separate `reasoning_content` field alongside `content` (6/7 models), or
  - inline in `content`, wrapped in <think>...</think> (MiniMax-M3 only).
extract_thinking() below handles both.
"""
import json
import os
import re
import time
import urllib.request

UPSTREAM_BASE_URL = os.environ.get("ZG_UPSTREAM_BASE_URL")
UPSTREAM_API_KEY = os.environ.get("ZG_UPSTREAM_API_KEY", "")
FAKE = not UPSTREAM_BASE_URL  # no upstream configured -> deterministic offline stand-in

REASONING_ON = "high"    # any value other than none/minimal turns thinking on
REASONING_OFF = "none"   # turns thinking off where the model has a native toggle

_THINK_TAG_RE = re.compile(r"<think>(.*?)</think>\s*(.*)", re.DOTALL)

# Every call's full input/output is logged to
#   <LOG_DIR>/<experiment>__<role>__<model>.jsonl   (one JSON line appended per call)
# so a full run can be replayed/audited call-by-call after the fact, not just
# from the final replay row. Only fires when the caller passes an `experiment`
# name (run_eval.py does) -- ad-hoc/offline/test calls that don't pass one
# produce no log files, so `python3 tests.py` and dev pokes stay log-free.
LOG_DIR = os.environ.get("ZG_CALL_LOG_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "call_logs"))
_UNSAFE_FILENAME_CHARS = re.compile(r"[^A-Za-z0-9._-]+")


def _sanitize(name):
    return _UNSAFE_FILENAME_CHARS.sub("-", name or "unknown")


def _log_call(experiment, role, model, request_body, response_message):
    if not experiment:
        return
    os.makedirs(LOG_DIR, exist_ok=True)
    fname = f"{_sanitize(experiment)}__{_sanitize(role)}__{_sanitize(model)}.jsonl"
    record = {
        "ts": time.time(),
        "experiment": experiment,
        "role": role,
        "model": model,
        "request": request_body,
        "response": response_message,
    }
    with open(os.path.join(LOG_DIR, fname), "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def call_llm(model, messages, tools=None, json_mode=False, reasoning_effort=None, experiment=None, role=None):
    """One single-shot chat completion. No retries, no loop, no tool execution.

    `experiment`/`role` (e.g. role="panel"/"judge"/"synthesis"/"baseline") are
    optional and only used to name the call-log file -- they are never sent
    upstream as part of the request body.
    """
    body = {"model": model, "messages": messages}
    if tools:
        body["tools"] = tools
    if json_mode:
        body["response_format"] = {"type": "json_object"}
    if reasoning_effort:
        body["reasoning_effort"] = reasoning_effort

    if FAKE:
        message = _fake_llm(model, messages, tools, json_mode, reasoning_effort)
        # Wrapped the same shape as the real payload below (choices[0].message)
        # so log files have one consistent schema regardless of FAKE mode --
        # just without a real usage block, since FAKE mode has no real token
        # counts to report and fabricating them would be misleading.
        _log_call(experiment, role, model, body, {"choices": [{"message": message}]})
        return message

    req = urllib.request.Request(
        UPSTREAM_BASE_URL.rstrip("/") + "/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Authorization": f"Bearer {UPSTREAM_API_KEY}", "Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        payload = json.loads(resp.read())
    message = payload["choices"][0]["message"]
    # Log the FULL raw response payload, not just the message we use --
    # usage (prompt/completion/reasoning token counts), finish_reason, id,
    # and the actually-served model (may differ from the requested one on a
    # router fallback) all matter for auditing a real run and must not be
    # silently dropped.
    _log_call(experiment, role, model, body, payload)
    return message


def extract_thinking(message):
    """Split a raw provider message into (reasoning_text_or_None, clean_content).

    Handles both patterns confirmed by real testing across the 7 candidate
    models: a separate `reasoning_content` field (the common case), or
    inline <think>...</think> markup in `content` (MiniMax-M3). Returns
    (None, content) when neither is present (e.g. a model with thinking off).
    """
    reasoning_content = message.get("reasoning_content")
    content = message.get("content") or ""
    if reasoning_content:
        return reasoning_content, content
    m = _THINK_TAG_RE.search(content)
    if m:
        return m.group(1).strip(), m.group(2).strip()
    return None, content


def _fake_llm(model, messages, tools, json_mode, reasoning_effort):
    """Deterministic stand-in so the pipeline runs/tests without network or API
    keys. Mirrors the three real behaviour patterns found by probing 0g-router:
      - "minimax" in the model id -> always thinks, inline <think> in content
      - "hy3" in the model id     -> only thinks when reasoning_effort is on
      - everything else          -> always thinks, separate reasoning_content
    """
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
    answer = f"[fake:{model}] {user_text[:60]}"
    thinks = reasoning_effort not in (None, "", "none", "minimal") if "hy3" in model else True
    if not thinks:
        return {"content": answer, "tool_calls": None}
    reasoning = f"[fake reasoning by {model}]"
    if "minimax" in model:
        return {"content": f"<think>{reasoning}</think>\n{answer}", "tool_calls": None}
    return {"content": answer, "reasoning_content": reasoning, "tool_calls": None}
