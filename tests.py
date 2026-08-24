"""Self-test suite, plain asserts (no external test framework). Runs entirely offline
via the FAKE llm stand-in (no ZG_UPSTREAM_BASE_URL set). Covers every requirement from
the design doc: single-round panel/judge/synthesis, no iteration, no tool execution,
allow_tool_call_output gating (panel/synthesis vs judge), eval client -> real HTTP ->
mock server round trip, replay file schema, and grading.
Run: python3 tests.py
"""
import json
import os
import threading
import time
import urllib.request

os.environ.pop("ZG_UPSTREAM_BASE_URL", None)  # force FAKE mode for this whole run

from mock_fusion_api import pipeline, panel_config as cfg  # noqa: E402
from mock_fusion_api.server import Handler  # noqa: E402
from http.server import ThreadingHTTPServer  # noqa: E402

PASS = []


def check(name, cond):
    PASS.append((name, bool(cond)))
    print(("PASS " if cond else "FAIL ") + name)


# --- 1. panel: exactly one call per model, parallel, no tool execution ------------
messages = [{"role": "user", "content": "hello world"}]
panel = pipeline.run_panel(messages, tools=None, allow_tool_call_output=False)
check("panel has one result per configured model", len(panel) == len(cfg.PANEL_MODELS))
check("panel results are distinct per model", len({r["model"] for r in panel}) == len(cfg.PANEL_MODELS))
check("panel results carry no tool_calls when none forced", all(r["tool_calls"] is None for r in panel))

# --- 1b. panel prompt gains the tool-emit instruction only when tools are actually
#         offered (regression: this line used to be missing from the panel prompt
#         entirely, unlike the synthesis prompt) --------------------------------
_captured = []
_real_call_llm = pipeline.call_llm
pipeline.call_llm = lambda model, msgs, tools=None, json_mode=False: (
    _captured.append({"model": model, "system": msgs[0]["content"]}) or
    _real_call_llm(model, msgs, tools, json_mode)
)
sample_tools = [{"type": "function", "function": {"name": "search", "parameters": {}}}]

_captured.clear()
pipeline.run_panel(messages, sample_tools, allow_tool_call_output=True)
check("panel prompt includes tool-emit instruction when tools are allowed",
      all("emit the tool call directly" in c["system"] for c in _captured))

_captured.clear()
pipeline.run_panel(messages, sample_tools, allow_tool_call_output=False)
check("panel prompt omits tool-emit instruction when tools are not allowed",
      all("emit the tool call directly" not in c["system"] for c in _captured))

pipeline.call_llm = _real_call_llm

# --- 2. judge: never receives tools, produces parseable JSON -----------------------
judge_json = pipeline.run_judge(messages, panel)
parsed = json.loads(judge_json)
check("judge output is valid JSON with expected keys",
      set(parsed) >= {"consensus", "contradictions", "final_guidance"})

# --- 3. synthesis: tools stripped when allow_tool_call_output=False ----------------
tools = [{"type": "function", "function": {"name": "search", "parameters": {}}}]
os.environ["ZG_FAKE_FORCE_TOOL_CALL"] = cfg.SYNTHESIS_MODEL
final_no_tools = pipeline.run_synthesis(messages, panel, judge_json, tools, allow_tool_call_output=False)
check("tools stripped -> no tool_call even though fake would force one",
      not final_no_tools.get("tool_calls"))

# --- 4. synthesis: tool_call passes through when allow_tool_call_output=True ------
final_with_tools = pipeline.run_synthesis(messages, panel, judge_json, tools, allow_tool_call_output=True)
check("tools allowed -> tool_call surfaces", bool(final_with_tools.get("tool_calls")))
del os.environ["ZG_FAKE_FORCE_TOOL_CALL"]

# --- 5. run_fusion end-to-end, default switch off ----------------------------------
resp = pipeline.run_fusion({"messages": messages})
msg = resp["choices"][0]["message"]
check("run_fusion returns text answer with switch off by default", bool(msg["content"]) and not msg["tool_calls"])
check("run_fusion finish_reason is stop when no tool_call", resp["choices"][0]["finish_reason"] == "stop")
check("debug field carries panel + judge_json", "panel" in resp["0g_fusion"] and "judge_json" in resp["0g_fusion"])

# --- 6. model routing: non-fusion model id is a single plain passthrough call -----
plain = pipeline.handle_chat_completion({"model": "some-baseline", "messages": messages})
check("non-fusion model bypasses the fusion pipeline",
      plain["choices"][0]["message"]["content"] == f"[fake:some-baseline] {messages[0]['content'][:60]}")

# --- 7. live HTTP round trip: eval.client.call_api -> real socket -> mock server --
server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
port = server.server_address[1]
threading.Thread(target=server.serve_forever, daemon=True).start()
time.sleep(0.2)

from eval.client import call_api  # noqa: E402

base_url = f"http://127.0.0.1:{port}"
live_resp = call_api(base_url, "0g/fusion-preview", messages, allow_tool_call_output=False)
check("live HTTP call to mock server returns fusion answer", bool(live_resp["choices"][0]["message"]["content"]))

# unknown route returns 404
req = urllib.request.Request(base_url + "/v1/unknown", data=b"{}", method="POST")
try:
    urllib.request.urlopen(req)
    check("unknown route returns 404", False)
except urllib.error.HTTPError as e:
    check("unknown route returns 404", e.code == 404)

# --- 8. full run_eval.run() against the live server, writing a replay file --------
from eval.run_eval import run as run_eval  # noqa: E402

out_path = os.path.join(os.path.dirname(__file__), "eval", "results", "test_run.jsonl")
run_eval(base_url, "0g/fusion-preview", base_url, "baseline-model", out_path, limit=2)
with open(out_path) as f:
    rows = [json.loads(line) for line in f]
check("replay file has expected row count", len(rows) == 2)
check("replay rows have required schema keys",
      all({"schema", "instruction", "fusion", "baseline", "config_id"} <= set(r) for r in rows))

# --- 9. grading over the replay file ----------------------------------------------
from eval.grade import grade_replay  # noqa: E402

result = grade_replay(out_path)
check("grade_replay returns a win_rate in [0,1] over all rows",
      0.0 <= result["win_rate"] <= 1.0 and result["n"] == 2)

server.shutdown()
os.remove(out_path)

failed = [name for name, ok in PASS if not ok]
print(f"\n{len(PASS) - len(failed)}/{len(PASS)} passed")
if failed:
    raise SystemExit("FAILED: " + ", ".join(failed))
