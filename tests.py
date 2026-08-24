"""Self-test suite, plain asserts (no external test framework). Runs entirely
offline via the FAKE llm stand-in (no ZG_UPSTREAM_BASE_URL set). Covers the
GPQA-round requirements: reasoning_effort on for panel/synthesis/baseline, off
for judge (with defensive stripping), panel evidence carrying reasoning AND
content, thinking-extraction for both real-world field patterns, GPQA task
loading + letter extraction/grading, and the end-to-end HTTP round trip.
Run: python3 tests.py
"""
import json
import os
import threading
import time
import urllib.request

os.environ.pop("ZG_UPSTREAM_BASE_URL", None)  # force FAKE mode for this whole run

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

failed = [name for name, ok in PASS if not ok]
print(f"\n{len(PASS) - len(failed)}/{len(PASS)} passed")
if failed:
    raise SystemExit("FAILED: " + ", ".join(failed))
