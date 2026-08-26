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
import contextlib
import glob
import io
import json
import os
import shutil
import threading
import time
import urllib.request

os.environ.pop("ZG_UPSTREAM_BASE_URL", None)  # force FAKE mode for this whole run

import llm_client  # noqa: E402
llm_client.RETRY_SLEEP_SECONDS = 0  # don't actually wait between retries in tests
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

r, c = extract_thinking({"content": "Let me consider this. <think>step by step</think>\nfinal answer"})
check("extract_thinking keeps text preceding the <think> tag instead of silently dropping it",
      c == "Let me consider this. \nfinal answer")

r, c = extract_thinking({"content": "<think>first</think>mid<think>second</think>final answer"})
check("extract_thinking strips ALL <think> blocks, not just the first",
      "<think>" not in c and "</think>" not in c)
check("extract_thinking joins every <think> block's text into reasoning, not just the first",
      "first" in r and "second" in r)

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
# Note: can't assert the model name string is absent from `evidence` outright --
# FAKE mode's own simulated answer text embeds "[fake:<model>] ..." by design
# (see llm_client._fake_llm), so the model name legitimately shows up as part
# of a panel member's own *answer*. What must be absent is panel_evidence's
# own "model=<name>" attribution tag (production's fusionPanelEvidence has
# this; this eval deliberately drops it -- see panel_evidence's docstring).
check("panel_evidence does NOT add its own model= attribution tag (anonymized by position, unlike production)",
      "model=" not in evidence)

# --- 2b. FINAL_LETTER_INSTRUCTION must not be duplicated: it's already baked
#         into the caller's own question text (gpqa_tasks.py); _panel_messages
#         / run_synthesis must not re-add a second copy on top -------------
_msg_with_instruction = [{"role": "user", "content": f"Some question?\n\n{cfg.FINAL_LETTER_INSTRUCTION}"}]
_orig_call_llm = pipeline.call_llm
_captured = []


def _capture_call_llm(model, msgs, *a, **kw):
    _captured.append(msgs)
    return _orig_call_llm(model, msgs, *a, **kw)


pipeline.call_llm = _capture_call_llm
try:
    _captured.clear()
    pipeline.run_panel(_msg_with_instruction, tools=None, allow_tool_call_output=False)
    check("_panel_messages does not re-add a 2nd copy of FINAL_LETTER_INSTRUCTION",
          all(sum(m.get("content", "").count(cfg.FINAL_LETTER_INSTRUCTION) for m in sent) == 1
              for sent in _captured))

    _captured.clear()
    pipeline.run_synthesis(_msg_with_instruction, panel, "{}", tools=None, allow_tool_call_output=False)
    combined = "\n".join(m.get("content", "") for m in _captured[0])
    check("run_synthesis does not re-add a 2nd copy of FINAL_LETTER_INSTRUCTION",
          combined.count(cfg.FINAL_LETTER_INSTRUCTION) == 1)
finally:
    pipeline.call_llm = _orig_call_llm

# --- 3. judge: called with reasoning_effort=none, output is valid clean JSON -
judge_json = pipeline.run_judge(messages, panel)
parsed = json.loads(judge_json)
check("judge output is valid JSON with expected keys",
      set(parsed) >= {"consensus", "contradictions", "final_guidance"})
check("judge output has no leaked <think> tag", "<think>" not in judge_json)

# --- 3b. json_mode is withheld for judge models in JUDGE_MODELS_WITHOUT_JSON_MODE
#         (0g-router hard-rejects response_format for a model that doesn't
#         advertise it -- confirmed live: a real 400 model_not_capable against
#         minimax-m3) -- and still sent for every other model (regression) ---
_orig_call_llm = pipeline.call_llm
_orig_judge_model = cfg.JUDGE_MODEL
_seen_json_mode = {}


def _capture_json_mode(model, msgs, tools=None, json_mode=False, **kw):
    _seen_json_mode["value"] = json_mode
    return _orig_call_llm(model, msgs, tools, json_mode=json_mode, **kw)


pipeline.call_llm = _capture_json_mode
try:
    cfg.JUDGE_MODEL = "judge-model"  # not in JUDGE_MODELS_WITHOUT_JSON_MODE
    pipeline.run_judge(messages, panel)
    check("json_mode is still sent for a judge model that supports response_format",
          _seen_json_mode["value"] is True)

    cfg.JUDGE_MODEL = "minimax-m3"  # confirmed real 0g-router 400 case
    stderr_capture = io.StringIO()
    with contextlib.redirect_stderr(stderr_capture):
        result = pipeline.run_judge(messages, panel, question_id=42)
    check("json_mode is withheld for minimax-m3 (JUDGE_MODELS_WITHOUT_JSON_MODE)",
          _seen_json_mode["value"] is False)
    check("run_judge with minimax-m3 as judge still returns a usable string, no crash",
          isinstance(result, str) and len(result) > 0)
    warning = stderr_capture.getvalue()
    check("malformed judge JSON (FAKE minimax-m3's non-JSON text) prints a clear stderr warning "
          "naming the question_id and judge model",
          "fusion.judge_json_invalid" in warning and "question_id=42" in warning
          and "judge_model='minimax-m3'" in warning)

    cfg.JUDGE_MODEL = "judge-model"
    stderr_capture2 = io.StringIO()
    with contextlib.redirect_stderr(stderr_capture2):
        pipeline.run_judge(messages, panel, question_id=7)
    check("no warning is printed when the judge's output IS valid JSON (no false positives)",
          stderr_capture2.getvalue() == "")

    # 0g-router resolves model ids case-insensitively (models.yaml: "MiniMax-M3"
    # "differs from canonical only by case") -- the local check must match that,
    # or a differently-cased ZG_JUDGE_MODEL would slip json_mode=True back
    # through and reproduce the exact live 400 this was built to avoid.
    for variant in ("MiniMax-M3", "MINIMAX-M3", "  minimax-m3  ", "minimax/minimax-m3"):
        cfg.JUDGE_MODEL = variant
        pipeline.run_judge(messages, panel)
        check(f"json_mode withheld regardless of casing/whitespace ({variant!r})",
              _seen_json_mode["value"] is False)
finally:
    pipeline.call_llm = _orig_call_llm
    cfg.JUDGE_MODEL = _orig_judge_model

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

# request["question_id"] must reach run_judge (via run_fusion) so a malformed
# judge JSON can actually be traced back to which question caused it
_orig_judge_model2 = cfg.JUDGE_MODEL
cfg.JUDGE_MODEL = "minimax-m3"
try:
    stderr_capture3 = io.StringIO()
    with contextlib.redirect_stderr(stderr_capture3):
        pipeline.run_fusion({"messages": messages, "question_id": 99})
    check("run_fusion forwards request['question_id'] through to run_judge's warning",
          "question_id=99" in stderr_capture3.getvalue())
finally:
    cfg.JUDGE_MODEL = _orig_judge_model2

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

# call_api must surface the server's real error body on a 500, not a bare
# HTTPError with the body silently discarded -- this is exactly what made a
# real 500 from pipeline.run_fusion (e.g. a malformed cached_panel entry)
# undiagnosable from run_eval.py's traceback until this was fixed.
try:
    call_api(base_url, "0g/fusion-preview", messages, cached_panel=[{"content": "missing model key"}])
    check("call_api surfaces the real error body on HTTP 500 instead of a bare HTTPError", False)
except RuntimeError as e:
    check("call_api surfaces the real error body on HTTP 500 instead of a bare HTTPError",
          "500" in str(e) and "cached_panel" in str(e))

from eval.run_eval import run as run_eval  # noqa: E402

out_path = os.path.join(os.path.dirname(__file__), "eval", "results", "test_run.jsonl")
run_eval(base_url, "0g/fusion-preview", base_url, ["baseline-model"], out_path, limit=2)
with open(out_path) as f:
    rows = [json.loads(line) for line in f]
check("replay file has expected row count", len(rows) == 2)
check("replay rows have GPQA schema keys (correct_letter, both content+reasoning)",
      all({"schema", "question_id", "instruction", "correct_letter", "fusion", "baselines", "config_id"} <= set(r)
          for r in rows))
check("each row's baselines list has exactly the 1 requested model",
      all([b["model"] for b in r["baselines"]] == ["baseline-model"] for r in rows))

from eval.gpqa_grade import grade_replay  # noqa: E402

result = grade_replay(out_path)
check("grade_replay reports fusion accuracy over all rows and a per-model baseline score",
      result["fusion"]["n"] == 2 and result["baselines"]["baseline-model"]["n"] == 2)

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


_seen_headers = {}


class _FakeUpstreamHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        _seen_headers["user_agent"] = self.headers.get("User-Agent")
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
check("real HTTP path sends a non-default User-Agent (avoids Cloudflare error 1010 -- "
      "confirmed live against kimi-k3 through router-api.0g.ai, urllib's default UA got 403'd)",
      _seen_headers.get("user_agent") not in (None, "", "Python-urllib/3.10")
      and "python-urllib" not in (_seen_headers.get("user_agent") or "").lower())

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

# --- 11b. a real 403/etc from upstream must surface its actual body, not a
#          bare "HTTP Error 403: Forbidden" with the reason discarded -- this
#          is the exact bug a real run against pc.0g.ai/staging hit. A
#          persistent failure must also be retried MAX_RETRIES+1 times total
#          before finally giving up -----------------------------------------
_403_request_count = {"n": 0}


class _FakeUpstream403Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        _403_request_count["n"] += 1
        body = json.dumps({"error": {"message": "invalid api key for this model tier"}}).encode()
        self.send_response(403)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *a):
        pass


upstream403 = ThreadingHTTPServer(("127.0.0.1", 0), _FakeUpstream403Handler)
upstream403_port = upstream403.server_address[1]
threading.Thread(target=upstream403.serve_forever, daemon=True).start()
time.sleep(0.1)

_orig_fake, _orig_base_url = llm_client.FAKE, llm_client.UPSTREAM_BASE_URL
llm_client.FAKE = False
llm_client.UPSTREAM_BASE_URL = f"http://127.0.0.1:{upstream403_port}"
try:
    stderr_403 = io.StringIO()
    with contextlib.redirect_stderr(stderr_403):
        try:
            llm_client.call_llm("some-model", messages)
            check("call_llm surfaces the real upstream error body on HTTP 403", False)
        except RuntimeError as e:
            check("call_llm surfaces the real upstream error body on HTTP 403",
                  "403" in str(e) and "invalid api key for this model tier" in str(e))
    check("a persistent failure is retried MAX_RETRIES+1 times total, not just tried once",
          _403_request_count["n"] == llm_client.MAX_RETRIES + 1)
    check("each retry (all but the last attempt) prints a clear stderr warning",
          stderr_403.getvalue().count("llm_client.call_llm_retry") == llm_client.MAX_RETRIES)
finally:
    llm_client.FAKE, llm_client.UPSTREAM_BASE_URL = _orig_fake, _orig_base_url
    upstream403.shutdown()

# --- 11c. a call that fails a few times then succeeds must return normally,
#          not raise -- retrying only helps if eventual success is honored --
_flaky_request_count = {"n": 0}
FLAKY_SUCCEEDS_ON_ATTEMPT = 3


class _FlakyThenSucceedsHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        _flaky_request_count["n"] += 1
        if _flaky_request_count["n"] < FLAKY_SUCCEEDS_ON_ATTEMPT:
            body = json.dumps({"error": "temporary overload"}).encode()
            self.send_response(503)
        else:
            body = json.dumps({"choices": [{"message": {"content": "recovered"}}]}).encode()
            self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *a):
        pass


upstream_flaky = ThreadingHTTPServer(("127.0.0.1", 0), _FlakyThenSucceedsHandler)
upstream_flaky_port = upstream_flaky.server_address[1]
threading.Thread(target=upstream_flaky.serve_forever, daemon=True).start()
time.sleep(0.1)

llm_client.FAKE = False
llm_client.UPSTREAM_BASE_URL = f"http://127.0.0.1:{upstream_flaky_port}"
try:
    msg = llm_client.call_llm("some-model", messages)
    check("a call that fails twice then succeeds on the 3rd attempt returns normally (no exception)",
          msg.get("content") == "recovered")
    check("exactly FLAKY_SUCCEEDS_ON_ATTEMPT requests were made -- stops retrying once it succeeds",
          _flaky_request_count["n"] == FLAKY_SUCCEEDS_ON_ATTEMPT)
finally:
    llm_client.FAKE, llm_client.UPSTREAM_BASE_URL = _orig_fake, _orig_base_url
    upstream_flaky.shutdown()

# --- 11d. HTTP 200 with a malformed/wrong-shaped body must ALSO retry and
#          raise a clean RuntimeError -- a bare json.loads()/dict-index would
#          otherwise raise straight out of the retry loop, skipping both the
#          retry and the clear-error wrapping. Verified live, not hypothetical
#          (found while reviewing this same retry feature) ------------------
def _make_fixed_body_server(body_bytes, status=200):
    class _FixedBodyHandler(BaseHTTPRequestHandler):
        def do_POST(self):
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body_bytes)))
            self.end_headers()
            self.wfile.write(body_bytes)

        def log_message(self, *a):
            pass
    srv = ThreadingHTTPServer(("127.0.0.1", 0), _FixedBodyHandler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    time.sleep(0.1)
    return srv


llm_client.FAKE = False
for _label, _body, _expect_in_msg in (
    ("malformed JSON body", b"not valid json{{{", "non-JSON response"),
    ("wrong-shape JSON body (no choices key)", json.dumps({"error": "rate limited"}).encode(), "unexpected response shape"),
):
    srv = _make_fixed_body_server(_body)
    llm_client.UPSTREAM_BASE_URL = f"http://127.0.0.1:{srv.server_address[1]}"
    try:
        llm_client.call_llm("some-model", messages)
        check(f"HTTP 200 + {_label} raises (not silently succeeds)", False)
    except RuntimeError as e:
        check(f"HTTP 200 + {_label} raises a clean RuntimeError (not a bare JSONDecodeError/KeyError)",
              _expect_in_msg in str(e))
    except Exception as e:
        check(f"HTTP 200 + {_label} raises a clean RuntimeError, got {type(e).__name__} instead", False)
    finally:
        srv.shutdown()
llm_client.FAKE, llm_client.UPSTREAM_BASE_URL = _orig_fake, _orig_base_url

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
run_eval(base_url2, "0g/fusion-preview", base_url2, ["baseline-model"], base_path, limit=2)

variant_path = os.path.join(os.path.dirname(__file__), "eval", "results", "test_variant.jsonl")
run_variant(base_path, base_url2, "0g/fusion-preview", "panel-new-candidate", variant_path,
            fixed_count=len(cfg.PANEL_MODELS))

with open(variant_path, encoding="utf-8") as f:
    variant_rows = [json.loads(line) for line in f]
check("run_variant produces one row per base row", len(variant_rows) == 2)
check("variant rows carry the swapped-in model name + which panel models were reused",
      all(r["variant_model"] == "panel-new-candidate" and set(r["cached_panel_models"]) == set(cfg.PANEL_MODELS)
          for r in variant_rows))
check("variant rows carry the base run's baselines unchanged (not re-called)",
      all(r["baselines"][0]["content"] for r in variant_rows))
check("variant fusion panel = original cfg.PANEL_MODELS (cached) + the 1 new variant model",
      all(len(r["fusion"]["raw_response"]["0g_fusion"]["panel"]) == len(cfg.PANEL_MODELS) + 1 for r in variant_rows))

variant_result = grade_replay(variant_path)
check("grade_replay works on a run_variant output file", variant_result["fusion"]["n"] == 2)

# --- 14. run_eval.py: one question's call failing must not abort the run --
#         it's caught, logged, written as a failed row, and the loop moves
#         on to the next question -------------------------------------------
import eval.run_eval as run_eval_module  # noqa: E402
import eval.run_variant as run_variant_module  # noqa: E402
from eval.gpqa_grade import _score, _baseline_content  # noqa: E402

_orig_call_api = run_eval_module.call_api


def _fail_question_1(base_url, model, msgs, **kw):
    if kw.get("question_id") == 1:
        raise RuntimeError("simulated permanent failure for question 1")
    return _orig_call_api(base_url, model, msgs, **kw)


run_eval_module.call_api = _fail_question_1
try:
    catch_path = os.path.join(os.path.dirname(__file__), "eval", "results", "test_catch.jsonl")
    stderr_run = io.StringIO()
    with contextlib.redirect_stderr(stderr_run):
        run_eval_module.run(base_url2, "0g/fusion-preview", base_url2, ["baseline-model"], catch_path, limit=3)
    with open(catch_path, encoding="utf-8") as f:
        catch_rows = [json.loads(line) for line in f]
finally:
    run_eval_module.call_api = _orig_call_api

check("run_eval.py writes one row per task even when one question's call fails "
      "(no missing rows, run did not abort)", len(catch_rows) == 3)
check("the failed question's row is marked failed with the error captured",
      catch_rows[1].get("failed") is True and "simulated permanent failure" in catch_rows[1].get("error", ""))
check("questions before and after the failing one still succeeded normally",
      "failed" not in catch_rows[0] and "failed" not in catch_rows[2]
      and catch_rows[0]["fusion"]["content"] and catch_rows[2]["fusion"]["content"])
run_eval_stderr = stderr_run.getvalue()
check("run_eval.py prints a clear per-question failure warning naming the question_id",
      "eval.run_eval_question_failed" in run_eval_stderr and "question_id=1" in run_eval_stderr)
check("run_eval.py prints a final failed-count summary", "eval.run_eval_summary" in run_eval_stderr
      and "total=3" in run_eval_stderr and "failed=1" in run_eval_stderr)

# --- 14b. a call that returns 200 with an unexpected shape (not an exception
#          from call_api itself) must ALSO be caught -- an earlier version of
#          this fix only wrapped the two call_api() calls in try/except,
#          leaving the success-row construction (which indexes into the
#          response) exposed right after it. Verified live: this crashed the
#          whole run before the fix, confirmed it no longer does -----------
def _malformed_shape_call_api(base_url, model, msgs, **kw):
    if kw.get("question_id") == 1:
        return {"choices": []}  # 200 OK, valid JSON, but empty choices -> IndexError on use
    return _orig_call_api(base_url, model, msgs, **kw)


run_eval_module.call_api = _malformed_shape_call_api
try:
    shape_path = os.path.join(os.path.dirname(__file__), "eval", "results", "test_shape.jsonl")
    run_eval_module.run(base_url2, "0g/fusion-preview", base_url2, ["baseline-model"], shape_path, limit=3)
    with open(shape_path, encoding="utf-8") as f:
        shape_rows = [json.loads(line) for line in f]
finally:
    run_eval_module.call_api = _orig_call_api

check("run_eval.py doesn't crash when a call succeeds but returns a malformed/wrong-shape response",
      len(shape_rows) == 3)
check("the malformed-shape question is caught and marked failed (not left half-built or crashing)",
      shape_rows[1].get("failed") is True and "index" in shape_rows[1].get("error", "").lower())
check("the other 2 questions are unaffected by the malformed-shape one",
      "failed" not in shape_rows[0] and "failed" not in shape_rows[2])
os.remove(shape_path)

# --- 15. gpqa_grade._score: call_failed (never got a response) is counted
#         separately from extraction_failed (got a response, no letter) ----
_fusion_content = lambda r: (r.get("fusion") or {}).get("content")  # noqa: E731
mixed_result = _score(catch_rows, _fusion_content)
check("_score counts the failed row as call_failed, not folded into extraction_failed",
      mixed_result["call_failed"] == 1 and mixed_result["n"] == 3)
check("_score still scores the 2 successful rows normally alongside the failed one",
      mixed_result["correct"] + mixed_result["extraction_failed"] == 2)
grade_replay(catch_path)  # must not raise on a file containing a failed row
os.remove(catch_path)

# --- 16. run_variant.py: same catch-and-continue for a candidate call that
#         fails, plus gracefully skipping a base row that was itself failed -
base_path2 = os.path.join(os.path.dirname(__file__), "eval", "results", "test_base2.jsonl")
run_eval_module.run(base_url2, "0g/fusion-preview", base_url2, ["baseline-model"], base_path2, limit=3)
with open(base_path2, encoding="utf-8") as f:
    base_rows2 = [json.loads(line) for line in f]
base_rows2[1] = {**base_rows2[1], "failed": True, "error": "pretend this base row failed too"}
with open(base_path2, "w", encoding="utf-8") as f:
    for r in base_rows2:
        f.write(json.dumps(r) + "\n")

_orig_call_api_v = run_variant_module.call_api


def _fail_question_2(base_url, model, msgs, **kw):
    if kw.get("question_id") == 2:
        raise RuntimeError("simulated permanent failure for question 2's variant call")
    return _orig_call_api_v(base_url, model, msgs, **kw)


run_variant_module.call_api = _fail_question_2
try:
    variant_catch_path = os.path.join(os.path.dirname(__file__), "eval", "results", "test_variant_catch.jsonl")
    stderr_variant = io.StringIO()
    with contextlib.redirect_stderr(stderr_variant):
        run_variant_module.run(base_path2, base_url2, "0g/fusion-preview", "panel-new-candidate",
                                variant_catch_path, fixed_count=len(cfg.PANEL_MODELS))
    with open(variant_catch_path, encoding="utf-8") as f:
        variant_catch_rows = [json.loads(line) for line in f]
finally:
    run_variant_module.call_api = _orig_call_api_v

check("run_variant.py writes one row per base row even with a failed base row + a failed candidate call",
      len(variant_catch_rows) == 3)
check("a base row that was itself failed is carried forward as failed, not crashed on (missing cached_panel)",
      variant_catch_rows[1].get("failed") is True)
check("a candidate call that fails is caught and marked failed, run continues",
      variant_catch_rows[2].get("failed") is True
      and "simulated permanent failure for question 2" in variant_catch_rows[2].get("error", ""))
check("the one genuinely-successful row (question 0) is unaffected",
      "failed" not in variant_catch_rows[0] and variant_catch_rows[0]["fusion"]["content"])
os.remove(base_path2)
os.remove(variant_catch_path)

# --- 16b. same "200-but-wrong-shape must still be caught" check as 14b,
#          for run_variant.py's own success-row construction --------------
base_path3 = os.path.join(os.path.dirname(__file__), "eval", "results", "test_base3.jsonl")
run_eval_module.run(base_url2, "0g/fusion-preview", base_url2, ["baseline-model"], base_path3, limit=2)


def _malformed_shape_call_api_v(base_url, model, msgs, **kw):
    if kw.get("question_id") == 0:
        return {"choices": []}
    return _orig_call_api_v(base_url, model, msgs, **kw)


run_variant_module.call_api = _malformed_shape_call_api_v
try:
    shape_variant_path = os.path.join(os.path.dirname(__file__), "eval", "results", "test_shape_variant.jsonl")
    run_variant_module.run(base_path3, base_url2, "0g/fusion-preview", "panel-new-candidate", shape_variant_path,
                            fixed_count=len(cfg.PANEL_MODELS))
    with open(shape_variant_path, encoding="utf-8") as f:
        shape_variant_rows = [json.loads(line) for line in f]
finally:
    run_variant_module.call_api = _orig_call_api_v

check("run_variant.py doesn't crash when a call succeeds but returns a malformed/wrong-shape response",
      len(shape_variant_rows) == 2)
check("the malformed-shape question is caught and marked failed",
      shape_variant_rows[0].get("failed") is True and "index" in shape_variant_rows[0].get("error", "").lower())
check("the other question is unaffected", "failed" not in shape_variant_rows[1])
os.remove(base_path3)
os.remove(shape_variant_path)

# --- 17. resume: re-running the same --out with a bigger --limit must reuse
#         already-succeeded rows byte-for-byte and only call for the new/
#         previously-failed ones -- this is the whole point of keying the
#         default output file by experiment name instead of a timestamp ----
resume_path = os.path.join(os.path.dirname(__file__), "eval", "results", "test_resume.jsonl")
if os.path.exists(resume_path):
    os.remove(resume_path)

_call_count = {"n": 0}


def _counting_call_api(base_url, model, msgs, **kw):
    _call_count["n"] += 1
    return _orig_call_api(base_url, model, msgs, **kw)


run_eval_module.call_api = _counting_call_api
try:
    run_eval_module.run(base_url2, "0g/fusion-preview", base_url2, ["baseline-model"], resume_path, limit=3)
    check("first pass (limit=3) makes 2 calls/question x 3 questions", _call_count["n"] == 6)

    _call_count["n"] = 0
    stderr_resume = io.StringIO()
    with contextlib.redirect_stderr(stderr_resume):
        run_eval_module.run(base_url2, "0g/fusion-preview", base_url2, ["baseline-model"], resume_path, limit=6)
    check("resumed pass (limit=6) only calls for the 3 NEW questions, not the 3 already-successful ones",
          _call_count["n"] == 6)  # 3 new questions x 2 calls each, NOT 6 questions x 2 = 12
    check("resume prints a clear 'skipped N already-successful rows' message",
          "eval.run_eval_resumed skipped=3" in stderr_resume.getvalue())
finally:
    run_eval_module.call_api = _orig_call_api

with open(resume_path, encoding="utf-8") as f:
    resumed_rows = [json.loads(line) for line in f]
check("resumed output has all 6 rows (3 reused + 3 newly computed)", len(resumed_rows) == 6)
check("the first 3 rows are reused byte-for-byte from the first pass (same fusion content)",
      all("failed" not in r for r in resumed_rows[:3]))
os.remove(resume_path)

# a previously-FAILED question must be retried, not permanently skipped
retry_path = os.path.join(os.path.dirname(__file__), "eval", "results", "test_resume_retry.jsonl")
if os.path.exists(retry_path):
    os.remove(retry_path)


def _fail_q1_once(base_url, model, msgs, **kw):
    if kw.get("question_id") == 1 and not _fail_q1_once.done:
        _fail_q1_once.done = True
        raise RuntimeError("simulated one-time failure")
    return _orig_call_api(base_url, model, msgs, **kw)


_fail_q1_once.done = False
run_eval_module.call_api = _fail_q1_once
try:
    run_eval_module.run(base_url2, "0g/fusion-preview", base_url2, ["baseline-model"], retry_path, limit=3)
    with open(retry_path, encoding="utf-8") as f:
        first_pass_rows = [json.loads(line) for line in f]
    check("question 1 failed on the first pass (as designed by the test)",
          first_pass_rows[1].get("failed") is True)

    run_eval_module.call_api = _orig_call_api  # this time it will succeed
    run_eval_module.run(base_url2, "0g/fusion-preview", base_url2, ["baseline-model"], retry_path, limit=3)
    with open(retry_path, encoding="utf-8") as f:
        second_pass_rows = [json.loads(line) for line in f]
    check("a previously-failed question is retried on the next resume (not permanently stuck as failed)",
          "failed" not in second_pass_rows[1] and second_pass_rows[1]["fusion"]["content"])
finally:
    run_eval_module.call_api = _orig_call_api
os.remove(retry_path)

# --no-resume equivalent (resume=False) must ignore the existing file entirely
noresume_path = os.path.join(os.path.dirname(__file__), "eval", "results", "test_no_resume.jsonl")
run_eval_module.run(base_url2, "0g/fusion-preview", base_url2, ["baseline-model"], noresume_path, limit=2)
_call_count["n"] = 0
run_eval_module.call_api = _counting_call_api
try:
    run_eval_module.run(base_url2, "0g/fusion-preview", base_url2, ["baseline-model"], noresume_path, limit=2,
                         resume=False)
finally:
    run_eval_module.call_api = _orig_call_api
check("resume=False re-calls everything even though a valid prior result exists",
      _call_count["n"] == 4)  # 2 questions x 2 calls, not 0
os.remove(noresume_path)

# a config_id mismatch (different fusion/baseline model under the same
# out_path) must warn, not silently produce a file mixing two configs
mismatch_path = os.path.join(os.path.dirname(__file__), "eval", "results", "test_mismatch.jsonl")
run_eval_module.run(base_url2, "0g/fusion-preview", base_url2, ["baseline-model"], mismatch_path, limit=1)
stderr_mismatch = io.StringIO()
with contextlib.redirect_stderr(stderr_mismatch):
    run_eval_module.run(base_url2, "0g/fusion-preview", base_url2, ["a-DIFFERENT-baseline-model"], mismatch_path,
                         limit=1)
check("reusing a file written under a different config_id prints a mismatch warning",
      "eval.run_eval_resume_config_mismatch" in stderr_mismatch.getvalue())
os.remove(mismatch_path)

# run_variant.py has its own separate resume implementation -- verify it too,
# not just assume it works because run_eval.py's does
base_path4 = os.path.join(os.path.dirname(__file__), "eval", "results", "test_base4.jsonl")
run_eval_module.run(base_url2, "0g/fusion-preview", base_url2, ["baseline-model"], base_path4, limit=3)
variant_resume_path = os.path.join(os.path.dirname(__file__), "eval", "results", "test_variant_resume.jsonl")
if os.path.exists(variant_resume_path):
    os.remove(variant_resume_path)

_vcall_count = {"n": 0}


def _counting_call_api_v(base_url, model, msgs, **kw):
    _vcall_count["n"] += 1
    return _orig_call_api_v(base_url, model, msgs, **kw)


run_variant_module.call_api = _counting_call_api_v
try:
    run_variant_module.run(base_path4, base_url2, "0g/fusion-preview", "panel-new-candidate",
                            variant_resume_path, limit=2, fixed_count=len(cfg.PANEL_MODELS))
    check("run_variant first pass (limit=2) makes 2 calls", _vcall_count["n"] == 2)

    _vcall_count["n"] = 0
    stderr_vresume = io.StringIO()
    with contextlib.redirect_stderr(stderr_vresume):
        run_variant_module.run(base_path4, base_url2, "0g/fusion-preview", "panel-new-candidate",
                                variant_resume_path, limit=3, fixed_count=len(cfg.PANEL_MODELS))
    check("run_variant resumed pass (limit=3) only calls for the 1 new question", _vcall_count["n"] == 1)
    check("run_variant resume prints its own 'skipped N' message",
          "eval.run_variant_resumed skipped=2" in stderr_vresume.getvalue())
finally:
    run_variant_module.call_api = _orig_call_api_v
os.remove(base_path4)
os.remove(variant_resume_path)

# --- 17b. run_baseline.py: gets a 2nd baseline's answers WITHOUT re-calling
#          fusion -- the whole point of it existing instead of just running
#          run_eval.py twice with a different --baseline-model -----------
import eval.run_baseline as run_baseline_module  # noqa: E402

base_path6 = os.path.join(os.path.dirname(__file__), "eval", "results", "test_base6.jsonl")
run_eval_module.run(base_url2, "0g/fusion-preview", base_url2, ["baseline-model-1"], base_path6, limit=3)

_bcall_count = {"n": 0}
_orig_call_api_b = run_baseline_module.call_api


def _counting_call_api_b(base_url, model, msgs, **kw):
    _bcall_count["n"] += 1
    return _orig_call_api_b(base_url, model, msgs, **kw)


run_baseline_module.call_api = _counting_call_api_b
try:
    baseline2_path = os.path.join(os.path.dirname(__file__), "eval", "results", "test_baseline2.jsonl")
    run_baseline_module.run(base_path6, base_url2, "baseline-model-2", baseline2_path)
    check("run_baseline.py makes exactly 1 call per question (the baseline only, nothing else)",
          _bcall_count["n"] == 3)
    with open(baseline2_path, encoding="utf-8") as f:
        baseline2_rows = [json.loads(line) for line in f]
    with open(base_path6, encoding="utf-8") as f:
        base6_rows = [json.loads(line) for line in f]
    check("run_baseline.py's output has fusion copied through byte-for-byte, not recomputed",
          all(b["fusion"] == m["fusion"] for b, m in zip(base6_rows, baseline2_rows)))
    check("run_baseline.py's output ACCUMULATES: the base run's baseline-model-1 is still there",
          all(any(b["model"] == "baseline-model-1" for b in r["baselines"]) for r in baseline2_rows))
    check("...and the newly requested baseline-model-2 is added alongside it",
          all(any(b["model"] == "baseline-model-2" for b in r["baselines"]) for r in baseline2_rows))
    check("...exactly those 2, no duplicates", all(len(r["baselines"]) == 2 for r in baseline2_rows))

    # resume: re-running for the SAME model must reuse already-succeeded rows
    _bcall_count["n"] = 0
    stderr_b = io.StringIO()
    with contextlib.redirect_stderr(stderr_b):
        run_baseline_module.run(base_path6, base_url2, "baseline-model-2", baseline2_path)
    check("run_baseline.py resume skips already-succeeded questions (0 new calls)", _bcall_count["n"] == 0)
    check("run_baseline.py resume prints its own 'skipped N' message",
          "eval.run_baseline_resumed skipped=3" in stderr_b.getvalue())

    # a THIRD distinct call, same --out: must accumulate a 3rd baseline
    # alongside the 2 already there, not replace or warn about them
    run_baseline_module.run(base_path6, base_url2, "baseline-model-3", baseline2_path)
    with open(baseline2_path, encoding="utf-8") as f:
        baseline3_rows = [json.loads(line) for line in f]
    check("a 2nd run_baseline.py call with a DIFFERENT model accumulates a 3rd entry, "
          "not replace the first 2 or warn",
          all({b["model"] for b in r["baselines"]} == {"baseline-model-1", "baseline-model-2", "baseline-model-3"}
              for r in baseline3_rows))
finally:
    run_baseline_module.call_api = _orig_call_api_b
os.remove(base_path6)
os.remove(baseline2_path)

# --- 18. VariantSetupError guard: catches the exact accumulation bug found
#         earlier (base run misconfigured with a candidate already baked
#         in) instead of silently building a 6/7-member panel -------------
from eval.run_variant import VariantSetupError  # noqa: E402

base_path5 = os.path.join(os.path.dirname(__file__), "eval", "results", "test_base5.jsonl")
run_eval_module.run(base_url2, "0g/fusion-preview", base_url2, ["baseline-model"], base_path5, limit=1)

setup_out = os.path.join(os.path.dirname(__file__), "eval", "results", "test_setup_guard.jsonl")

_vcall_count["n"] = 0
run_variant_module.call_api = _counting_call_api_v
try:
    try:
        run_variant_module.run(base_path5, base_url2, "0g/fusion-preview", "panel-new-candidate",
                                setup_out, fixed_count=4)  # real panel has len(cfg.PANEL_MODELS)=3, not 4
        check("fixed_count mismatch raises VariantSetupError instead of silently proceeding", False)
    except VariantSetupError as e:
        check("fixed_count mismatch raises VariantSetupError instead of silently proceeding",
              "expected exactly 4" in str(e))
    check("the mismatch is caught before making ANY calls (fail fast, no wasted API spend)",
          _vcall_count["n"] == 0)

    # variant_model already present in cached_panel (the exact accumulation
    # scenario: base run already has this "candidate" baked in) must also raise
    try:
        run_variant_module.run(base_path5, base_url2, "0g/fusion-preview", cfg.PANEL_MODELS[0],
                                setup_out, fixed_count=len(cfg.PANEL_MODELS))
        check("variant_model already present in cached_panel raises VariantSetupError", False)
    except VariantSetupError as e:
        check("variant_model already present in cached_panel raises VariantSetupError",
              "already present" in str(e))

    # fixed_count=None must skip the size check entirely -- an explicit opt-out
    run_variant_module.run(base_path5, base_url2, "0g/fusion-preview", "panel-new-candidate",
                            setup_out, fixed_count=None)
    check("fixed_count=None skips the size check and runs normally", os.path.exists(setup_out))
finally:
    run_variant_module.call_api = _orig_call_api_v
os.remove(base_path5)
os.remove(setup_out)

# the actual accumulation reproduction: a base run configured with the
# candidate already in the panel (5 members instead of the real 4-fixed
# plan) must be rejected, not silently produce a 6-member panel
os.environ["ZG_PANEL_MODELS"] = "fixed-1,fixed-2,fixed-3,fixed-4,candidate-1"
import importlib  # noqa: E402
import mock_fusion_api.panel_config as cfg_reloadable  # noqa: E402
importlib.reload(cfg_reloadable)
import mock_fusion_api.pipeline as pipeline_reloadable  # noqa: E402
importlib.reload(pipeline_reloadable)
import mock_fusion_api.server as server_mod_reloadable  # noqa: E402
importlib.reload(server_mod_reloadable)

server_bad = ThreadingHTTPServer(("127.0.0.1", 0), server_mod_reloadable.Handler)
threading.Thread(target=server_bad.serve_forever, daemon=True).start()
time.sleep(0.2)
base_url_bad = f"http://127.0.0.1:{server_bad.server_address[1]}"

bad_base_path = os.path.join(os.path.dirname(__file__), "eval", "results", "test_bad_base.jsonl")
run_eval_module.run(base_url_bad, "0g/fusion-preview", base_url_bad, ["baseline-model"], bad_base_path, limit=1)
bad_variant_out = os.path.join(os.path.dirname(__file__), "eval", "results", "test_bad_variant.jsonl")
try:
    run_variant_module.run(bad_base_path, base_url_bad, "0g/fusion-preview", "candidate-2", bad_variant_out,
                            fixed_count=4)
    check("a base run with 5 panel members (candidate already baked in) is rejected with fixed_count=4",
          False)
except VariantSetupError as e:
    check("a base run with 5 panel members (candidate already baked in) is rejected with fixed_count=4",
          "expected exactly 4" in str(e) and "candidate-1" in str(e))
server_bad.shutdown()
os.remove(bad_base_path)
if os.path.exists(bad_variant_out):
    os.remove(bad_variant_out)

# importlib.reload() above mutated the actual mock_fusion_api.panel_config /
# pipeline / server modules in place -- that's the SAME module object `cfg`
# (etc.) is already bound to everywhere else in this file, not a copy, so
# leaving ZG_PANEL_MODELS unset-but-already-reloaded-with-5-models would
# silently corrupt cfg.PANEL_MODELS for any test added after this point.
# Restore it explicitly rather than relying on nothing else needing it.
os.environ.pop("ZG_PANEL_MODELS", None)
importlib.reload(cfg_reloadable)
importlib.reload(pipeline_reloadable)
importlib.reload(server_mod_reloadable)
check("global module state (cfg.PANEL_MODELS) is restored after the reload-with-5-models probe above",
      cfg.PANEL_MODELS == ["panel-a", "panel-b", "panel-c"])

# --- 19. regressions found in pre-flight review ---------------------------

# 19a. json.loads() on an undecodable body raises UnicodeDecodeError, NOT
#      json.JSONDecodeError (neither is a subclass of the other), so an
#      except clause naming only JSONDecodeError let it escape the retry loop
#      entirely -- no retry, no clear error. Repro: a 200 response whose body
#      is non-UTF-8 bytes (binary/gzip/truncated multi-byte).
llm_client.FAKE = False
srv = _make_fixed_body_server(b"\xff\xfe\x00\x01garbage")
llm_client.UPSTREAM_BASE_URL = f"http://127.0.0.1:{srv.server_address[1]}"
_undecodable_stderr = io.StringIO()
try:
    with contextlib.redirect_stderr(_undecodable_stderr):
        llm_client.call_llm("some-model", messages)
    check("HTTP 200 + undecodable (non-UTF-8) body raises (not silently succeeds)", False)
except RuntimeError as e:
    check("HTTP 200 + undecodable (non-UTF-8) body raises a clean RuntimeError, "
          "not a bare UnicodeDecodeError out of the retry loop", "non-JSON response" in str(e))
except Exception as e:
    check(f"HTTP 200 + undecodable body raises a clean RuntimeError, got {type(e).__name__} instead", False)
finally:
    srv.shutdown()
check("an undecodable body is RETRIED like any other failure, not raised on the first attempt",
      _undecodable_stderr.getvalue().count("call_llm_retry") == llm_client.MAX_RETRIES)
llm_client.FAKE, llm_client.UPSTREAM_BASE_URL = _orig_fake, _orig_base_url

# 19b. `failed` is a per-ROW flag but a row can fail on one side only:
#      run_baseline.py copies a good `fusion` through verbatim and marks the
#      row failed when only the BASELINE call failed. _score keyed off that
#      flag threw those fusion answers away and understated fusion accuracy.
base_path7 = os.path.join(os.path.dirname(__file__), "eval", "results", "test_base7.jsonl")
run_eval_module.run(base_url2, "0g/fusion-preview", base_url2, ["baseline-model-1"], base_path7, limit=3)
onesided_path = os.path.join(os.path.dirname(__file__), "eval", "results", "test_onesided.jsonl")
run_baseline_module.call_api = lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("baseline is down"))
try:
    with contextlib.redirect_stderr(io.StringIO()):
        run_baseline_module.run(base_path7, base_url2, "baseline-model-2", onesided_path)
finally:
    run_baseline_module.call_api = _orig_call_api_b
onesided_rows = [json.loads(l) for l in open(onesided_path, encoding="utf-8")]
check("run_baseline row with only the baseline call failed still carries the base run's fusion, "
      "and does NOT mark the whole row failed (only that one baseline entry is)",
      all(not r.get("failed") and r.get("fusion")
          and any(b.get("model") == "baseline-model-2" and b.get("failed") for b in r.get("baselines", []))
          for r in onesided_rows))
check("_score grades that carried-through fusion instead of discarding it as call_failed",
      _score(onesided_rows, _fusion_content)["call_failed"] == 0
      and _score(onesided_rows, _fusion_content)["n"] == 3)
check("_score still counts the side that really did fail as call_failed",
      _score(onesided_rows, lambda r: _baseline_content(r, "baseline-model-2"))["call_failed"] == 3)
# ...and the both-sides-missing shape (run_eval's own failed row) is unchanged
check("_score still counts a row with NEITHER side as call_failed on both",
      _score([{"failed": True}], _fusion_content)["call_failed"] == 1
      and _score([{"failed": True}], lambda r: _baseline_content(r, "x"))["call_failed"] == 1)

# 19c. grade_replay was the only JSONL reader in the harness that didn't skip
#      blank lines, so a stray trailing newline crashed grading on a file the
#      run_* scripts themselves read back happily.
blank_path = os.path.join(os.path.dirname(__file__), "eval", "results", "test_blank.jsonl")
with open(blank_path, "w", encoding="utf-8") as f:
    f.write(open(base_path7, encoding="utf-8").read() + "\n")
check("grade_replay tolerates a blank trailing line (as every other reader here does)",
      grade_replay(blank_path)["fusion"]["n"] == 3)
os.remove(blank_path)

# 19d. question_id is only a positional index into the dataset, so it is NOT
#      stable across datasets -- and load_tasks() silently switches from
#      gpqa_sample.jsonl to the real gpqa_diamond.jsonl the moment that
#      download appears. Resuming a smoke test into a full run across that
#      switch reused rows answering question A while grading them against
#      question B's correct_letter: silently wrong accuracy, no crash.
import eval.gpqa_tasks as gpqa_tasks_module  # noqa: E402


def _write_dataset(path, tag):
    with open(path, "w", encoding="utf-8") as f:
        for i in range(3):
            f.write(json.dumps({"Question": f"{tag} question {i}", "Correct Answer": f"{tag}-right-{i}",
                                 "Incorrect Answer 1": f"{tag}-w1-{i}", "Incorrect Answer 2": f"{tag}-w2-{i}",
                                 "Incorrect Answer 3": f"{tag}-w3-{i}"}) + "\n")


_results_dir = os.path.join(os.path.dirname(__file__), "eval", "results")
ds_a, ds_b = os.path.join(_results_dir, "test_ds_a.jsonl"), os.path.join(_results_dir, "test_ds_b.jsonl")
_write_dataset(ds_a, "DATASET-A")
_write_dataset(ds_b, "DATASET-B")
swap_path = os.path.join(_results_dir, "test_dsswap.jsonl")
if os.path.exists(swap_path):  # don't resume into a leftover from an aborted earlier run
    os.remove(swap_path)
_orig_real_default = gpqa_tasks_module.REAL_DEFAULT_PATH
try:
    gpqa_tasks_module.REAL_DEFAULT_PATH = ds_a
    run_eval_module.run(base_url2, "0g/fusion-preview", base_url2, ["bl"], swap_path, limit=2)
    # same dataset -> resume must still work exactly as before (no false positive)
    with contextlib.redirect_stderr(io.StringIO()):
        run_eval_module.run(base_url2, "0g/fusion-preview", base_url2, ["bl"], swap_path, limit=3)
    check("resume across an UNCHANGED dataset is unaffected by the identity check",
          sum(1 for _ in open(swap_path, encoding="utf-8")) == 3)
    gpqa_tasks_module.REAL_DEFAULT_PATH = ds_b
    try:
        run_eval_module.run(base_url2, "0g/fusion-preview", base_url2, ["bl"], swap_path, limit=3)
        check("run_eval refuses to resume rows written against a DIFFERENT dataset", False)
    except run_eval_module.ResumeMismatchError:
        check("run_eval refuses to resume rows written against a DIFFERENT dataset", True)
    check("the mismatch aborts before --out is opened for writing, leaving the old rows intact",
          all("DATASET-A" in json.loads(l)["instruction"] for l in open(swap_path, encoding="utf-8")))
    run_eval_module.run(base_url2, "0g/fusion-preview", base_url2, ["bl"], swap_path, limit=3, resume=False)
    check("--no-resume is the documented escape hatch and still recomputes from scratch",
          all("DATASET-B" in json.loads(l)["instruction"] for l in open(swap_path, encoding="utf-8")))
finally:
    gpqa_tasks_module.REAL_DEFAULT_PATH = _orig_real_default

# same guard on the two scripts that key off a --base-replay instead of the dataset
_shifted = []
for _line in open(base_path7, encoding="utf-8"):
    _r = json.loads(_line)
    _r["instruction"] = "A COMPLETELY DIFFERENT QUESTION\n" + _r["instruction"]
    _shifted.append(_r)
shifted_path = os.path.join(_results_dir, "test_shifted.jsonl")
with open(shifted_path, "w", encoding="utf-8") as f:
    for _r in _shifted:
        f.write(json.dumps(_r) + "\n")
for _name, _mod, _call in (
    ("run_variant", run_variant_module,
     lambda out: run_variant_module.run(shifted_path, base_url2, "0g/fusion-preview", "cand-x", out,
                                         fixed_count=None)),
    ("run_baseline", run_baseline_module,
     lambda out: run_baseline_module.run(shifted_path, base_url2, "bl-x", out)),
):
    _out = os.path.join(_results_dir, f"test_{_name}_mismatch.jsonl")
    if _name == "run_variant":
        run_variant_module.run(base_path7, base_url2, "0g/fusion-preview", "cand-x", _out, fixed_count=None)
    else:
        run_baseline_module.run(base_path7, base_url2, "bl-x", _out)
    try:
        _call(_out)
        check(f"{_name} refuses to resume rows written against a different --base-replay's questions", False)
    except _mod.ResumeMismatchError:
        check(f"{_name} refuses to resume rows written against a different --base-replay's questions", True)
    os.remove(_out)
for _p in (ds_a, ds_b, swap_path, shifted_path, onesided_path):
    os.remove(_p)

# --- 20. regressions found in the round-2 review of the multi-baseline
#         (row["baseline"] -> row["baselines"]) change -----------------------

# 20a. each baseline gets its OWN try/except: one of N failing must not cost
#      the row its fusion answer or the OTHER baselines' answers, and must
#      not shift the list (order still matches --baseline-model).
_multi_path = os.path.join(_results_dir, "test_multi_baseline.jsonl")
_orig_call_api_e = run_eval_module.call_api


def _one_baseline_down(url, model, msgs, **kw):
    if model == "bad-baseline":
        raise RuntimeError("this one model is down")
    return _orig_call_api_e(url, model, msgs, **kw)


run_eval_module.call_api = _one_baseline_down
try:
    with contextlib.redirect_stderr(io.StringIO()):
        run_eval_module.run(base_url2, "0g/fusion-preview", base_url2,
                            ["good-a", "bad-baseline", "good-b"], _multi_path, limit=3)
finally:
    run_eval_module.call_api = _orig_call_api_e
_multi_rows = [json.loads(l) for l in open(_multi_path, encoding="utf-8")]
check("one of N baseline models failing leaves the row itself unfailed and fusion intact",
      all(not r.get("failed") and r["fusion"]["content"] for r in _multi_rows))
check("...the other baselines in the same row still have their answers",
      all(r["baselines"][0]["content"] and r["baselines"][2]["content"] for r in _multi_rows))
check("...and only the failing model's own slot is marked failed, in --baseline-model order",
      all([(b["model"], bool(b.get("failed"))) for b in r["baselines"]]
          == [("good-a", False), ("bad-baseline", True), ("good-b", False)] for r in _multi_rows))
check("_score gives each baseline model its own numbers over the same rows",
      _score(_multi_rows, lambda r: _baseline_content(r, "good-a"))["call_failed"] == 0
      and _score(_multi_rows, lambda r: _baseline_content(r, "bad-baseline"))["call_failed"] == 3)

# 20b. resume is per-QUESTION, so a reused row keeps its FAILED baseline entry
#      and re-running run_eval silently leaves the baseline column incomplete.
#      Behaviour is intentional (retrying it would re-pay for fusion too), but
#      it must be said out loud, naming the model -- otherwise the run looks
#      finished when one baseline is missing for N questions.
_resume_stderr = io.StringIO()
with contextlib.redirect_stderr(_resume_stderr):
    run_eval_module.run(base_url2, "0g/fusion-preview", base_url2,
                        ["good-a", "bad-baseline", "good-b"], _multi_path, limit=3)
check("re-running run_eval reports the baseline model still missing from reused rows, by name/count",
      "eval.run_eval_baselines_still_missing" in _resume_stderr.getvalue()
      and "'bad-baseline': 3" in _resume_stderr.getvalue())
check("...and points at run_baseline.py, the tool that can actually fill it in",
      "eval.run_baseline" in _resume_stderr.getvalue())

# 20c. --limit scopes what a run CALLS, not what --out is allowed to keep.
#      Re-running the `--limit 5` smoke test after the full run used to
#      truncate the finished file down to 5 rows, destroying paid-for results.
run_eval_module.run(base_url2, "0g/fusion-preview", base_url2, ["good-a"], _multi_path,
                    limit=1, resume=True)
_kept_rows = [json.loads(l) for l in open(_multi_path, encoding="utf-8")]
check("a smaller --limit re-run keeps the prior rows outside its window instead of deleting them",
      len(_kept_rows) == 3 and [r["question_id"] for r in _kept_rows] == [0, 1, 2])
check("...and --no-resume is still the way to genuinely shrink the file",
      len([json.loads(l) for l in open(
          run_eval_module.run(base_url2, "0g/fusion-preview", base_url2, ["good-a"], _multi_path,
                              limit=1, resume=False), encoding="utf-8")]) == 1)
os.remove(_multi_path)

# 20d. only the FUSION side fails on a variant row; the baselines were never
#      re-called, so dropping them made the variant file report a lower
#      baseline accuracy than the base file it was built from.
_vfail_path = os.path.join(_results_dir, "test_variant_failed_baselines.jsonl")
_orig_call_api_v2 = run_variant_module.call_api
run_variant_module.call_api = lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("variant model down"))
try:
    with contextlib.redirect_stderr(io.StringIO()):
        run_variant_module.run(base_path7, base_url2, "0g/fusion-preview", "cand-y", _vfail_path,
                                fixed_count=None)
finally:
    run_variant_module.call_api = _orig_call_api_v2
_vfail_rows = [json.loads(l) for l in open(_vfail_path, encoding="utf-8")]
_base7_rows = [json.loads(l) for l in open(base_path7, encoding="utf-8")]
check("a variant row whose fusion call failed still carries the base run's baselines unchanged",
      all(r.get("failed") and r.get("baselines") == b["baselines"]
          for r, b in zip(_vfail_rows, _base7_rows)))
check("...so the variant file grades the baseline exactly as its base file does, not lower",
      _score(_vfail_rows, lambda r: _baseline_content(r, "baseline-model-1"))
      == _score(_base7_rows, lambda r: _baseline_content(r, "baseline-model-1")))
os.remove(_vfail_path)

# 20e. a baseline entry left {"failed": true} by an earlier run must be
#      RETRIED, not treated as "already have it" -- otherwise a transient
#      outage permanently poisons that model's column.
_retry_path = os.path.join(_results_dir, "test_baseline_retry.jsonl")
run_baseline_module.call_api = lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("baseline is down"))
try:
    with contextlib.redirect_stderr(io.StringIO()):
        run_baseline_module.run(base_path7, base_url2, "flaky-bl", _retry_path)
finally:
    run_baseline_module.call_api = _orig_call_api_b
check("a failed baseline entry is recorded as failed, not dropped",
      all(any(b["model"] == "flaky-bl" and b.get("failed") for b in r["baselines"])
          for r in (json.loads(l) for l in open(_retry_path, encoding="utf-8"))))
run_baseline_module.run(base_path7, base_url2, "flaky-bl", _retry_path)
_retry_rows = [json.loads(l) for l in open(_retry_path, encoding="utf-8")]
check("re-running run_baseline for that model retries it and replaces the failed entry",
      all([(b["model"], bool(b.get("failed"))) for b in r["baselines"]]
          == [("baseline-model-1", False), ("flaky-bl", False)] for r in _retry_rows))
os.remove(_retry_path)

# 20f. run_baseline promised to log-and-continue on a row with no fusion data,
#      but only checked the row-level "failed" flag -- a row with neither
#      crashed with KeyError('fusion'), AFTER the baseline call was paid for.
_nofusion_path = os.path.join(_results_dir, "test_no_fusion.jsonl")
_nofusion_out = os.path.join(_results_dir, "test_no_fusion_out.jsonl")
with open(_nofusion_path, "w", encoding="utf-8") as f:
    f.write(json.dumps({"question_id": 0, "instruction": "Q", "correct_letter": "A"}) + "\n")
_nofusion_calls = {"n": 0}
run_baseline_module.call_api = lambda *a, **kw: _nofusion_calls.__setitem__("n", _nofusion_calls["n"] + 1)
try:
    with contextlib.redirect_stderr(io.StringIO()):
        run_baseline_module.run(_nofusion_path, base_url2, "bl-z", _nofusion_out)
    check("a base row with no fusion side is skipped, not a KeyError crash", True)
except Exception as _e:
    check(f"a base row with no fusion side is skipped, not a {type(_e).__name__} crash", False)
finally:
    run_baseline_module.call_api = _orig_call_api_b
check("...and it is skipped BEFORE the baseline call, so nothing is paid for it",
      _nofusion_calls["n"] == 0)
os.remove(_nofusion_path)
os.remove(_nofusion_out)

# 20g. grade_replay over every row shape the three scripts can emit, checked
#      against hand-computed numbers -- not just "it didn't crash".
_shapes_path = os.path.join(_results_dir, "test_shapes.jsonl")
with open(_shapes_path, "w", encoding="utf-8") as f:
    for _r in (
        {"question_id": 0, "correct_letter": "A", "fusion": {"content": "Final Answer: A"},
         "baselines": [{"model": "A", "content": "Final Answer: A"},
                       {"model": "B", "content": "Final Answer: C"}]},          # 2 good baselines
        {"question_id": 1, "correct_letter": "B", "fusion": {"content": "Final Answer: B"},
         "baselines": [{"model": "A", "content": "no letter here"},
                       {"model": "B", "failed": True, "error": "boom"}]},       # 1 good + 1 failed
        {"question_id": 2, "correct_letter": "C", "failed": True, "error": "outage"},  # row-level failure
        {"question_id": 3, "correct_letter": "D", "fusion": {"content": "Final Answer: D"},
         "baselines": []},                                                      # no baselines at all
    ):
        f.write(json.dumps(_r) + "\n")
_shape_scores = grade_replay(_shapes_path)
check("grade_replay: fusion scores 3/4 with the row-level failure as call_failed, over all 4 rows",
      _shape_scores["fusion"] == {"accuracy": 0.75, "correct": 3, "extraction_failed": 0,
                                  "call_failed": 1, "n": 4})
check("grade_replay: baseline A = 1 correct, 1 unparseable, 2 rows it was never run for",
      _shape_scores["baselines"]["A"] == {"accuracy": 0.25, "correct": 1, "extraction_failed": 1,
                                          "call_failed": 2, "n": 4})
check("grade_replay: baseline B = answered once (wrong), failed/absent the other 3",
      _shape_scores["baselines"]["B"] == {"accuracy": 0.0, "correct": 0, "extraction_failed": 0,
                                          "call_failed": 3, "n": 4})
check("grade_replay lists exactly the models that appear in any baselines list",
      sorted(_shape_scores["baselines"]) == ["A", "B"])
os.remove(_shapes_path)

# 20h. --baseline-model is parsed in __main__, so exercise the real CLI: a
#      repeated model name must be collapsed, not called (and billed) twice.
_cli_out = os.path.join(_results_dir, "test_cli_dedupe.jsonl")
_cli = __import__("subprocess").run(
    [__import__("sys").executable, "-m", "eval.run_eval", "--fusion-url", base_url2,
     "--baseline-url", base_url2, "--baseline-model", " dup , dup ,other", "--out", _cli_out,
     "--limit", "1"],
    cwd=os.path.dirname(os.path.abspath(__file__)), capture_output=True, text=True)
check("the run_eval CLI runs end to end and writes its --out", _cli.returncode == 0,)
_cli_row = json.loads(open(_cli_out, encoding="utf-8").readline())
check("a repeated --baseline-model name is collapsed to one call, whitespace stripped, order kept",
      [b["model"] for b in _cli_row["baselines"]] == ["dup", "other"])
os.remove(_cli_out)
os.remove(base_path7)

server2.shutdown()
os.remove(base_path)
os.remove(variant_path)

failed = [name for name, ok in PASS if not ok]
print(f"\n{len(PASS) - len(failed)}/{len(PASS)} passed")
if failed:
    raise SystemExit("FAILED: " + ", ".join(failed))
