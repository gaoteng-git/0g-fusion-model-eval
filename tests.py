"""Self-test suite, plain asserts (no external test framework). Runs entirely
offline via the FAKE llm stand-in (no ZG_UPSTREAM_BASE_URL set). Covers the
GPQA-round requirements: reasoning_effort on for panel/synthesis/baseline, off
for judge (with defensive stripping), panel evidence carrying reasoning AND
content, thinking-extraction for both real-world field patterns, GPQA task
loading + letter extraction/grading, the end-to-end HTTP round trip,
per-call log-file naming/content (call_logs/<experiment>__<role>__<model>.jsonl),
and the cached_panel/extra_panel_models partial-reuse mode (run_variant.py).
Run: python3 tests.py
"""
import glob
import json
import os
import shutil
import threading
import time
import urllib.request

os.environ.pop("ZG_UPSTREAM_BASE_URL", None)  # force FAKE mode for this whole run

import llm_client  # noqa: E402
from llm_client import extract_thinking, REASONING_ON, REASONING_OFF  # noqa: E402
from mock_fusion_api import pipeline, panel_config as cfg  # noqa: E402
from mock_fusion_api.server import Handler  # noqa: E402
from http.server import ThreadingHTTPServer  # noqa: E402

PASS = []


def check(name, cond):
    PASS.append((name, bool(cond)))
    print(("PASS " if cond else "FAIL ") + name)


# --- 1. extract_thinking: both real-world field patterns -------------------
r, c = extract_thinking({"content": "<think>step by step</think>\nfinal answer"})
check("extract_thinking splits inline <think> (MiniMax-style)", r == "step by step" and c == "final answer")
r, c = extract_thinking({"content": "final answer", "reasoning_content": "step by step"})
check("extract_thinking reads separate reasoning_content field", r == "step by step" and c == "final answer")
r, c = extract_thinking({"content": "final answer"})
check("extract_thinking returns None when no thinking present", r is None and c == "final answer")

# --- 2. panel: reasoning_effort=high requested for every member, and each
#        panel result carries both reasoning and content ------------------
messages = [{"role": "user", "content": "Which is a baryon?\nA) Proton\nB) Electron"}]
panel = pipeline.run_panel(messages, tools=None, allow_tool_call_output=False)
check("panel has one result per configured model", len(panel) == len(cfg.PANEL_MODELS))
check("every panel member has non-empty reasoning (fake models always think)",
      all(r["reasoning"] for r in panel))
check("every panel member's content is clean (no leaked <think> tag)",
      all("<think>" not in (r["content"] or "") for r in panel))

evidence = pipeline.panel_evidence(panel)
check("panel_evidence includes each member's reasoning, not just their answer",
      all(r["reasoning"] in evidence for r in panel))
check("panel_evidence includes each member's final answer too",
      all(r["content"] in evidence for r in panel))

# --- 3. judge: called with reasoning_effort=none, output is valid clean JSON -
judge_json = pipeline.run_judge(messages, panel)
parsed = json.loads(judge_json)
check("judge output is valid JSON with expected keys",
      set(parsed) >= {"consensus", "contradictions", "final_guidance"})
check("judge output has no leaked <think> tag", "<think>" not in judge_json)

# --- 4. synthesis: called with reasoning_effort=high, returns both parts ---
final = pipeline.run_synthesis(messages, panel, judge_json, tools=None, allow_tool_call_output=False)
check("synthesis produced a non-empty final answer", bool(final["content"]))
check("synthesis captured its own reasoning (fake models always think)", bool(final["reasoning"]))

# --- 5. run_fusion end-to-end: message carries reasoning_content when present -
resp = pipeline.run_fusion({"messages": messages})
msg = resp["choices"][0]["message"]
check("run_fusion final content has no leaked <think> tag", "<think>" not in (msg["content"] or ""))
check("run_fusion exposes reasoning_content on the wire", bool(msg.get("reasoning_content")))
check("debug field carries panel (with reasoning) + judge_json",
      "panel" in resp["0g_fusion"] and all("reasoning" in p for p in resp["0g_fusion"]["panel"]))

# --- 6. passthrough (baseline) path: forwards reasoning_effort, normalizes
#        reasoning_content the same way as the fusion path ------------------
plain = pipeline.handle_chat_completion({"model": "some-baseline", "messages": messages, "reasoning_effort": "high"})
check("baseline passthrough exposes reasoning_content when thinking is requested",
      bool(plain["choices"][0]["message"].get("reasoning_content")))

minimax_like = pipeline.handle_chat_completion({"model": "minimax-m3", "messages": messages, "reasoning_effort": "high"})
check("baseline passthrough strips inline <think> out of content for MiniMax-style models",
      "<think>" not in (minimax_like["choices"][0]["message"]["content"] or ""))
check("...and surfaces it via reasoning_content instead",
      bool(minimax_like["choices"][0]["message"].get("reasoning_content")))

# --- 7. GPQA task loading: deterministic shuffle, correct_letter matches ---
from eval.gpqa_tasks import load_tasks  # noqa: E402

tasks = load_tasks(limit=3)
check("load_tasks returns the requested number of tasks", len(tasks) == 3)
check("every task has question_id/instruction/correct_letter",
      all({"question_id", "instruction", "correct_letter"} <= set(t) for t in tasks))
check("every task's instruction contains the final-answer format instruction",
      all(cfg.FINAL_LETTER_INSTRUCTION in t["instruction"] for t in tasks))
# re-loading must reproduce the same shuffle/correct_letter (index-seeded, not global RNG)
tasks_again = load_tasks(limit=3)
check("shuffle is deterministic across repeated loads",
      [t["correct_letter"] for t in tasks] == [t["correct_letter"] for t in tasks_again])

# --- 8. gpqa_grade: final-letter extraction -------------------------------
from eval.gpqa_grade import extract_final_letter  # noqa: E402

check("extracts a plain 'Final Answer: B'", extract_final_letter("blah blah\nFinal Answer: B") == "B")
check("extracts through markdown bold", extract_final_letter("**Final Answer: C**") == "C")
check("case-insensitive label", extract_final_letter("final answer: a") == "A")
check("returns None when the format instruction wasn't followed", extract_final_letter("I think it's B.") is None)
check("takes the LAST mention if there are several", extract_final_letter("Final Answer: A\n...\nFinal Answer: D") == "D")

# --- 9. live HTTP round trip + full run_eval + grade ------------------------
server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
port = server.server_address[1]
threading.Thread(target=server.serve_forever, daemon=True).start()
time.sleep(0.2)

from eval.client import call_api  # noqa: E402

base_url = f"http://127.0.0.1:{port}"
live_resp = call_api(base_url, "0g/fusion-preview", messages, allow_tool_call_output=False)
check("live HTTP call to mock server returns fusion answer", bool(live_resp["choices"][0]["message"]["content"]))

req = urllib.request.Request(base_url + "/v1/unknown", data=b"{}", method="POST")
try:
    urllib.request.urlopen(req)
    check("unknown route returns 404", False)
except urllib.error.HTTPError as e:
    check("unknown route returns 404", e.code == 404)

from eval.run_eval import run as run_eval  # noqa: E402

out_path = os.path.join(os.path.dirname(__file__), "eval", "results", "test_run.jsonl")
run_eval(base_url, "0g/fusion-preview", base_url, "baseline-model", out_path, limit=2)
with open(out_path) as f:
    rows = [json.loads(line) for line in f]
check("replay file has expected row count", len(rows) == 2)
check("replay rows have GPQA schema keys (correct_letter, both content+reasoning)",
      all({"schema", "question_id", "instruction", "correct_letter", "fusion", "baseline", "config_id"} <= set(r)
          for r in rows))

from eval.gpqa_grade import grade_replay  # noqa: E402

result = grade_replay(out_path)
check("grade_replay reports fusion + baseline accuracy over all rows",
      result["fusion"]["n"] == 2 and result["baseline"]["n"] == 2)

server.shutdown()
os.remove(out_path)

# --- 10. per-call logging: file named <experiment>__<role>__<model>.jsonl ---
TEST_EXPERIMENT = "unit-test-logging-exp"
_before = set(glob.glob(os.path.join(llm_client.LOG_DIR, f"{TEST_EXPERIMENT}__*")))
for f in _before:
    os.remove(f)

pipeline.run_panel(messages, tools=None, allow_tool_call_output=False, experiment=TEST_EXPERIMENT)
expected_panel_files = [
    os.path.join(llm_client.LOG_DIR, f"{TEST_EXPERIMENT}__panel__{llm_client._sanitize(m)}.jsonl")
    for m in cfg.PANEL_MODELS
]
check("run_panel(experiment=...) writes one log file per panel model",
      all(os.path.exists(p) for p in expected_panel_files))

with open(expected_panel_files[0], encoding="utf-8") as f:
    logged = json.loads(f.readline())
check("logged record has experiment/role/model/request/response",
      {"experiment", "role", "model", "request", "response"} <= set(logged)
      and logged["experiment"] == TEST_EXPERIMENT and logged["role"] == "panel")
check("logged request carries the actual messages sent",
      logged["request"].get("messages") == pipeline._panel_messages(messages, 0, None))
check("logged response is the full raw payload shape (choices[0].message), not just the message",
      bool(logged["response"]["choices"][0]["message"].get("content")))

panel_for_judge = pipeline.run_panel(messages, tools=None, allow_tool_call_output=False)  # no experiment
pipeline.run_judge(messages, panel_for_judge, experiment=TEST_EXPERIMENT)
pipeline.run_synthesis(messages, panel_for_judge, "{}", tools=None, allow_tool_call_output=False,
                        experiment=TEST_EXPERIMENT)
judge_log = os.path.join(llm_client.LOG_DIR, f"{TEST_EXPERIMENT}__judge__{llm_client._sanitize(cfg.JUDGE_MODEL)}.jsonl")
synthesis_log = os.path.join(
    llm_client.LOG_DIR, f"{TEST_EXPERIMENT}__synthesis__{llm_client._sanitize(cfg.SYNTHESIS_MODEL)}.jsonl")
check("run_judge(experiment=...) writes a judge-role log file", os.path.exists(judge_log))
check("run_synthesis(experiment=...) writes a synthesis-role log file", os.path.exists(synthesis_log))

pipeline.handle_chat_completion({"model": "some-baseline", "messages": messages, "reasoning_effort": "high",
                                  "experiment": TEST_EXPERIMENT})
baseline_log = os.path.join(llm_client.LOG_DIR, f"{TEST_EXPERIMENT}__baseline__{llm_client._sanitize('some-baseline')}.jsonl")
check("baseline passthrough with an experiment name writes a baseline-role log file", os.path.exists(baseline_log))

no_exp_before = set(glob.glob(os.path.join(llm_client.LOG_DIR, "None__*")))
pipeline.handle_chat_completion({"model": "some-other-baseline", "messages": messages})  # no experiment field at all
check("no experiment name -> no log file written",
      set(glob.glob(os.path.join(llm_client.LOG_DIR, "None__*"))) == no_exp_before)

for f in glob.glob(os.path.join(llm_client.LOG_DIR, f"{TEST_EXPERIMENT}__*")):
    os.remove(f)
if os.path.isdir(llm_client.LOG_DIR) and not os.listdir(llm_client.LOG_DIR):
    shutil.rmtree(llm_client.LOG_DIR)

# --- 11. logged response is the FULL raw payload, not just choices[0].message,
#         against a real (non-FAKE) HTTP round trip -- usage/id/finish_reason
#         must survive into the log file, not just content/reasoning_content ---
from http.server import BaseHTTPRequestHandler  # noqa: E402

FAKE_UPSTREAM_PAYLOAD = {
    "id": "chatcmpl-realistic-id-123",
    "model": "served-model-name-may-differ-from-requested",
    "usage": {"prompt_tokens": 42, "completion_tokens": 900, "reasoning_tokens": 800, "total_tokens": 942},
    "choices": [{"finish_reason": "stop", "message": {"content": "the real answer", "reasoning_content": "steps"}}],
}


class _FakeUpstreamHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        body = json.dumps(FAKE_UPSTREAM_PAYLOAD).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *a):
        pass


upstream = ThreadingHTTPServer(("127.0.0.1", 0), _FakeUpstreamHandler)
upstream_port = upstream.server_address[1]
threading.Thread(target=upstream.serve_forever, daemon=True).start()
time.sleep(0.1)

_orig_fake, _orig_base_url = llm_client.FAKE, llm_client.UPSTREAM_BASE_URL
llm_client.FAKE = False
llm_client.UPSTREAM_BASE_URL = f"http://127.0.0.1:{upstream_port}"
try:
    msg = llm_client.call_llm("some-model", messages, experiment=TEST_EXPERIMENT, role="realcheck")
finally:
    llm_client.FAKE, llm_client.UPSTREAM_BASE_URL = _orig_fake, _orig_base_url
upstream.shutdown()

check("real HTTP path still returns just the message to the caller (unchanged contract)",
      msg == FAKE_UPSTREAM_PAYLOAD["choices"][0]["message"])

realcheck_log = os.path.join(llm_client.LOG_DIR, f"{TEST_EXPERIMENT}__realcheck__some-model.jsonl")
with open(realcheck_log, encoding="utf-8") as f:
    real_logged = json.loads(f.readline())
check("logged response preserves usage (prompt/completion/reasoning tokens)",
      real_logged["response"].get("usage") == FAKE_UPSTREAM_PAYLOAD["usage"])
check("logged response preserves id + finish_reason + served model, not just message",
      real_logged["response"].get("id") == FAKE_UPSTREAM_PAYLOAD["id"]
      and real_logged["response"]["choices"][0]["finish_reason"] == "stop"
      and real_logged["response"].get("model") == FAKE_UPSTREAM_PAYLOAD["model"])

for f in glob.glob(os.path.join(llm_client.LOG_DIR, f"{TEST_EXPERIMENT}__*")):
    os.remove(f)
if os.path.isdir(llm_client.LOG_DIR) and not os.listdir(llm_client.LOG_DIR):
    shutil.rmtree(llm_client.LOG_DIR)

# --- 12. cached_panel / extra_panel_models: partial-reuse mode -------------
_cached_entry = {"model": "cached-model-x", "content": "CACHED ANSWER\nFinal Answer: A",
                 "reasoning": "cached reasoning, never re-generated", "tool_calls": None}

reused = pipeline.run_fusion({"messages": messages, "cached_panel": [_cached_entry], "extra_panel_models": ["panel-a"]})
reused_panel = reused["0g_fusion"]["panel"]
check("cached_panel entry is passed through byte-for-byte, not re-called", reused_panel[0] == _cached_entry)
check("extra_panel_models entry was actually called fresh (has FAKE-generated reasoning)",
      reused_panel[1]["model"] == "panel-a" and bool(reused_panel[1]["reasoning"])
      and reused_panel[1] != _cached_entry)
check("merged panel length == len(cached_panel) + len(extra_panel_models)", len(reused_panel) == 2)

try:
    pipeline.run_fusion({"messages": messages, "cached_panel": [{"content": "missing model key"}]})
    check("cached_panel entry missing 'model' key raises ValueError", False)
except ValueError:
    check("cached_panel entry missing 'model' key raises ValueError", True)

default_run = pipeline.run_fusion({"messages": messages})
check("no cached_panel/extra_panel_models -> unchanged default behaviour (full cfg.PANEL_MODELS panel)",
      len(default_run["0g_fusion"]["panel"]) == len(cfg.PANEL_MODELS))

# --- 13. run_variant.py end-to-end: reuses a base replay's cached panel,
#         only calls the new variant model, output still gradeable ---------
from eval.run_eval import run as run_eval  # noqa: E402
from eval.run_variant import run as run_variant  # noqa: E402

server2 = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
port2 = server2.server_address[1]
threading.Thread(target=server2.serve_forever, daemon=True).start()
time.sleep(0.2)
base_url2 = f"http://127.0.0.1:{port2}"

base_path = os.path.join(os.path.dirname(__file__), "eval", "results", "test_base.jsonl")
run_eval(base_url2, "0g/fusion-preview", base_url2, "baseline-model", base_path, limit=2)

variant_path = os.path.join(os.path.dirname(__file__), "eval", "results", "test_variant.jsonl")
run_variant(base_path, base_url2, "0g/fusion-preview", "panel-new-candidate", variant_path)

with open(variant_path, encoding="utf-8") as f:
    variant_rows = [json.loads(line) for line in f]
check("run_variant produces one row per base row", len(variant_rows) == 2)
check("variant rows carry the swapped-in model name + which panel models were reused",
      all(r["variant_model"] == "panel-new-candidate" and set(r["cached_panel_models"]) == set(cfg.PANEL_MODELS)
          for r in variant_rows))
check("variant rows carry the base run's baseline unchanged (not re-called)",
      all(r["baseline"]["content"] for r in variant_rows))
check("variant fusion panel = original cfg.PANEL_MODELS (cached) + the 1 new variant model",
      all(len(r["fusion"]["raw_response"]["0g_fusion"]["panel"]) == len(cfg.PANEL_MODELS) + 1 for r in variant_rows))

variant_result = grade_replay(variant_path)
check("grade_replay works on a run_variant output file", variant_result["fusion"]["n"] == 2)

server2.shutdown()
os.remove(base_path)
os.remove(variant_path)

failed = [name for name, ok in PASS if not ok]
print(f"\n{len(PASS) - len(failed)}/{len(PASS)} passed")
if failed:
    raise SystemExit("FAILED: " + ", ".join(failed))
