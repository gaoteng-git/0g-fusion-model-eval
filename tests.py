"""Self-test suite, plain asserts (no external test framework). Runs entirely
offline via the FAKE llm stand-in (no ZG_UPSTREAM_BASE_URL set). Covers the
GPQA-round requirements: reasoning_effort on for panel/synthesis/baseline, off
for judge (with defensive stripping), panel evidence carrying reasoning AND
content, thinking-extraction for both real-world field patterns, GPQA task
loading + letter extraction/grading, the end-to-end HTTP round trip,
per-call log-file naming/content (call_logs/<experiment>__<role>__<model>.jsonl),
the cached_panel/extra_panel_models partial-reuse mode and panel_only (both
used by eval/fuse.py and eval/panel.py), and the eval.panel / eval.fuse /
eval.baseline / eval.grade CLI tools themselves -- including the concrete
regression test for the reason eval.panel exists at all: building/extending a
panel must make ZERO judge/synthesis calls.
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

import eval.gpqa_tasks as gpqa_tasks_module  # noqa: E402
# load_tasks() silently switches its default from the fake sample set to the
# real, gated GPQA file the moment that file exists on disk one directory
# above this repo (see gpqa_tasks.py's docstring) -- and in this environment
# it does. Every test below that calls load_tasks() (directly or via
# eval.panel/eval.baseline) does so with no explicit path, so leaving that
# switch live would risk loading -- and, on any assertion failure, PRINTING --
# real gated question text. Force the sample set unconditionally for this
# whole run; the resume-safety tests further down restore/override it
# themselves, on purpose, to exercise the dataset-swap guard.
_orig_real_default = gpqa_tasks_module.REAL_DEFAULT_PATH
gpqa_tasks_module.REAL_DEFAULT_PATH = gpqa_tasks_module.SAMPLE_PATH

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

# --- 8. grade: final-letter extraction ------------------------------------
from eval.grade import extract_final_letter  # noqa: E402

check("extracts a plain 'Final Answer: B'", extract_final_letter("blah blah\nFinal Answer: B") == "B")
check("extracts through markdown bold", extract_final_letter("**Final Answer: C**") == "C")
check("case-insensitive label", extract_final_letter("final answer: a") == "A")
check("returns None when the format instruction wasn't followed", extract_final_letter("I think it's B.") is None)
check("takes the LAST mention if there are several", extract_final_letter("Final Answer: A\n...\nFinal Answer: D") == "D")

# --- 9. live HTTP round trip: the server started here stays up for every
#        eval.panel/eval.fuse/eval.baseline test further down -------------
server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
port = server.server_address[1]
threading.Thread(target=server.serve_forever, daemon=True).start()
time.sleep(0.2)
base_url = f"http://127.0.0.1:{port}"

from eval.client import call_api  # noqa: E402

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
# undiagnosable until this was fixed.
try:
    call_api(base_url, "0g/fusion-preview", messages, cached_panel=[{"content": "missing model key"}])
    check("call_api surfaces the real error body on HTTP 500 instead of a bare HTTPError", False)
except RuntimeError as e:
    check("call_api surfaces the real error body on HTTP 500 instead of a bare HTTPError",
          "500" in str(e) and "cached_panel" in str(e))

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

# --- 13. panel_only: pipeline.run_fusion stops after the panel, no judge or
#         synthesis call at all -- the mechanism eval.panel relies on to
#         build/extend a panel for free (see run_fusion's docstring). This is
#         the direct fix for the $67.23/198-question wasted judge+synthesis
#         cost a 4-fixed-panel-only base run used to pay for unconditionally --
_orig_run_judge, _orig_run_synthesis = pipeline.run_judge, pipeline.run_synthesis


def _must_not_be_called(*a, **kw):
    raise AssertionError("must not be called when panel_only is set")


pipeline.run_judge = _must_not_be_called
pipeline.run_synthesis = _must_not_be_called
try:
    panel_only_resp = pipeline.run_fusion({"messages": messages, "panel_only": True})
    check("panel_only response has no 'choices' key (no final answer produced)",
          "choices" not in panel_only_resp)
    check("panel_only response carries exactly the panel, nothing else, under 0g_fusion",
          set(panel_only_resp["0g_fusion"]) == {"panel"}
          and len(panel_only_resp["0g_fusion"]["panel"]) == len(cfg.PANEL_MODELS))

    # combined with cached_panel/extra_panel_models: same zero-cost guarantee,
    # and cached+fresh still merge exactly like the non-panel_only path
    combo_resp = pipeline.run_fusion({"messages": messages, "panel_only": True,
                                       "cached_panel": [_cached_entry], "extra_panel_models": ["panel-a"]})
    check("panel_only + cached_panel/extra_panel_models still makes no judge/synthesis call",
          "choices" not in combo_resp and len(combo_resp["0g_fusion"]["panel"]) == 2
          and combo_resp["0g_fusion"]["panel"][0] == _cached_entry)
finally:
    pipeline.run_judge, pipeline.run_synthesis = _orig_run_judge, _orig_run_synthesis

# panel_only being absent/falsy must be unchanged from before (regression guard)
check("panel_only defaulting to falsy is unchanged -- still runs judge+synthesis",
      "choices" in pipeline.run_fusion({"messages": messages}))

# --- 14. eval.panel: build a panel file -- resume, growing --limit, and (the
#          whole point) ZERO judge/synthesis calls made along the way -------
from eval import panel as panel_module  # noqa: E402
from eval import fuse as fuse_module  # noqa: E402
from eval import baseline as baseline_module  # noqa: E402
from eval.grade import grade_replay, load_rows  # noqa: E402
from eval.replay_io import ResumeMismatchError, run_replay  # noqa: E402

_results_dir = os.path.join(os.path.dirname(__file__), "eval", "results")


def _cli(*args):
    return __import__("subprocess").run(
        [__import__("sys").executable, "-m", *args],
        cwd=os.path.dirname(os.path.abspath(__file__)), capture_output=True, text=True)


panel_path = os.path.join(_results_dir, "test_panel.jsonl")
PANEL_EXP = "test-panel-exp"
for f in glob.glob(os.path.join(llm_client.LOG_DIR, f"{PANEL_EXP}__*")):
    os.remove(f)

panel_module.run(base_url, "0g/fusion-preview", list(cfg.PANEL_MODELS), panel_path, limit=3, experiment=PANEL_EXP)
with open(panel_path, encoding="utf-8") as f:
    panel_rows_out = [json.loads(l) for l in f]
check("eval.panel writes one row per question", len(panel_rows_out) == 3)
check("every panel row has the full requested panel, correct schema",
      all(r["schema"] == "0g.fusion_eval.gpqa.panel.v1"
          and {p["model"] for p in r["panel"]} == set(cfg.PANEL_MODELS) for r in panel_rows_out))

panel_logs = glob.glob(os.path.join(llm_client.LOG_DIR, f"{PANEL_EXP}__panel__*"))
judge_logs = glob.glob(os.path.join(llm_client.LOG_DIR, f"{PANEL_EXP}__judge__*"))
synthesis_logs = glob.glob(os.path.join(llm_client.LOG_DIR, f"{PANEL_EXP}__synthesis__*"))
check("eval.panel makes real panel calls (log files exist)", len(panel_logs) == len(cfg.PANEL_MODELS))
check("*** the $67.23/198q fix: eval.panel makes ZERO judge calls ***", judge_logs == [])
check("*** eval.panel makes ZERO synthesis calls ***", synthesis_logs == [])
for f in panel_logs:
    os.remove(f)

_orig_call_api_p = panel_module.call_api
_pcalls = {"n": 0}


def _counting_call_api_p(*a, **kw):
    _pcalls["n"] += 1
    return _orig_call_api_p(*a, **kw)


panel_module.call_api = _counting_call_api_p
try:
    stderr_p = io.StringIO()
    with contextlib.redirect_stderr(stderr_p):
        panel_module.run(base_url, "0g/fusion-preview", list(cfg.PANEL_MODELS), panel_path, limit=3,
                          experiment=PANEL_EXP)
    check("re-running eval.panel for the exact same models makes zero fresh calls (fully resumed)",
          _pcalls["n"] == 0)
    check("resume prints a clear skipped= message", "eval.panel_resumed skipped=3" in stderr_p.getvalue())

    # growing the window (limit 3 -> 5) must only call for the 2 NEW questions
    _pcalls["n"] = 0
    panel_module.run(base_url, "0g/fusion-preview", list(cfg.PANEL_MODELS), panel_path, limit=5,
                      experiment=PANEL_EXP)
    check("growing --limit only calls for the newly-included questions", _pcalls["n"] == 2)
    with open(panel_path, encoding="utf-8") as f:
        grown_rows = [json.loads(l) for l in f]
    check("growing --limit keeps the earlier rows and adds the new ones, in order",
          len(grown_rows) == 5 and [r["question_id"] for r in grown_rows] == [0, 1, 2, 3, 4])
finally:
    panel_module.call_api = _orig_call_api_p
os.remove(panel_path)

# --- 15. eval.panel --reuse: pull already-computed answers from a DIFFERENT
#          file, call only what's genuinely missing, including the zero-HTTP
#          path when nothing is missing at all -------------------------------
base_panel_path = os.path.join(_results_dir, "test_panel_base.jsonl")
panel_module.run(base_url, "0g/fusion-preview", list(cfg.PANEL_MODELS), base_panel_path, limit=3,
                  experiment="test-panel-base")

variant_panel_path = os.path.join(_results_dir, "test_panel_variant.jsonl")
_pcalls["n"] = 0
panel_module.call_api = _counting_call_api_p
try:
    panel_module.run(base_url, "0g/fusion-preview", list(cfg.PANEL_MODELS) + ["candidate-x"],
                      variant_panel_path, limit=3, experiment="test-panel-variant", reuse_path=base_panel_path)
    # one call_api round trip per QUESTION (each carries only the still-missing
    # models as extra_panel_models) -- 3 questions, so 3 calls, not 3x(models+1)
    check("--reuse calls once per question, carrying only the ONE genuinely new model each time",
          _pcalls["n"] == 3)
finally:
    panel_module.call_api = _orig_call_api_p

with open(variant_panel_path, encoding="utf-8") as f:
    variant_panel_rows = [json.loads(l) for l in f]
with open(base_panel_path, encoding="utf-8") as f:
    base_panel_rows = [json.loads(l) for l in f]
check("the reused models' entries are copied byte-for-byte from --reuse, not recomputed",
      all({p["model"]: p for p in vr["panel"] if p["model"] in cfg.PANEL_MODELS} == {p["model"]: p for p in br["panel"]}
          for vr, br in zip(variant_panel_rows, base_panel_rows)))
check("the new candidate model is present alongside the reused ones",
      all(any(p["model"] == "candidate-x" for p in r["panel"]) for r in variant_panel_rows))
check("variant panel has exactly len(PANEL_MODELS)+1 members",
      all(len(r["panel"]) == len(cfg.PANEL_MODELS) + 1 for r in variant_panel_rows))

# zero-HTTP-call optimization: re-running the same --reuse variant makes no calls
_pcalls["n"] = 0
panel_module.call_api = _counting_call_api_p
try:
    panel_module.run(base_url, "0g/fusion-preview", list(cfg.PANEL_MODELS) + ["candidate-x"],
                      variant_panel_path, limit=3, experiment="test-panel-variant", reuse_path=base_panel_path)
    check("re-running the same --reuse variant makes zero calls (everything already at --out)",
          _pcalls["n"] == 0)
finally:
    panel_module.call_api = _orig_call_api_p

# a brand-new --out whose every model is covered by --reuse alone (nothing at
# --out yet) must also make zero calls -- the cached-but-no-existing-row path
fully_reused_path = os.path.join(_results_dir, "test_panel_fully_reused.jsonl")
_pcalls["n"] = 0
panel_module.call_api = _counting_call_api_p
try:
    panel_module.run(base_url, "0g/fusion-preview", list(cfg.PANEL_MODELS), fully_reused_path, limit=3,
                      experiment="test-panel-fully-reused", reuse_path=base_panel_path)
    check("a brand-new --out fully covered by --reuse makes zero HTTP calls", _pcalls["n"] == 0)
finally:
    panel_module.call_api = _orig_call_api_p
with open(fully_reused_path, encoding="utf-8") as f:
    fully_reused_rows = [json.loads(l) for l in f]
check("...and still produces a correct, complete panel file",
      all({p["model"] for p in r["panel"]} == set(cfg.PANEL_MODELS) for r in fully_reused_rows))
os.remove(fully_reused_path)

# failure handling: a model call failing must be caught, row marked failed,
# other questions unaffected, run does not abort
def _fail_candidate(url, model, msgs, **kw):
    if "candidate-y" in (kw.get("extra_panel_models") or []):
        raise RuntimeError("simulated candidate failure")
    return _orig_call_api_p(url, model, msgs, **kw)


panel_fail_path = os.path.join(_results_dir, "test_panel_fail.jsonl")
panel_module.call_api = _fail_candidate
try:
    stderr_pf = io.StringIO()
    with contextlib.redirect_stderr(stderr_pf):
        panel_module.run(base_url, "0g/fusion-preview", list(cfg.PANEL_MODELS) + ["candidate-y"],
                          panel_fail_path, limit=3, experiment="test-panel-fail", reuse_path=base_panel_path)
finally:
    panel_module.call_api = _orig_call_api_p
with open(panel_fail_path, encoding="utf-8") as f:
    panel_fail_rows = [json.loads(l) for l in f]
check("eval.panel writes one row per question even when a model call fails (run doesn't abort)",
      len(panel_fail_rows) == 3)
check("the failed question's row is marked failed with the error captured",
      all(r.get("failed") is True and "simulated candidate failure" in r.get("error", "") for r in panel_fail_rows))
check("eval.panel prints a clear per-question failure warning naming the question_id",
      "eval.panel_question_failed" in stderr_pf.getvalue())
os.remove(panel_fail_path)

# a 200-but-wrong-shape response (not an exception) must ALSO be caught -- the
# row construction sits in the SAME try block as the call, on purpose
def _malformed_shape_panel(url, model, msgs, **kw):
    if "candidate-shape" in (kw.get("extra_panel_models") or []):
        return {"0g_fusion": {}}  # 200-ok, valid JSON, but missing "panel" -> KeyError on use
    return _orig_call_api_p(url, model, msgs, **kw)


panel_shape_path = os.path.join(_results_dir, "test_panel_shape.jsonl")
panel_module.call_api = _malformed_shape_panel
try:
    panel_module.run(base_url, "0g/fusion-preview", list(cfg.PANEL_MODELS) + ["candidate-shape"],
                      panel_shape_path, limit=2, experiment="test-panel-shape", reuse_path=base_panel_path)
finally:
    panel_module.call_api = _orig_call_api_p
with open(panel_shape_path, encoding="utf-8") as f:
    panel_shape_rows = [json.loads(l) for l in f]
check("eval.panel doesn't crash when a call succeeds but returns a malformed/wrong-shape response",
      len(panel_shape_rows) == 2)
check("the malformed-shape question is caught and marked failed (KeyError caught by the same try)",
      all(r.get("failed") is True for r in panel_shape_rows))
os.remove(panel_shape_path)
os.remove(base_panel_path)
os.remove(variant_panel_path)

# --- 16. eval.fuse: judge+synthesis over a panel file, with ZERO fresh panel
#          calls -- the panel is used exactly as given ----------------------
fuse_panel_path = os.path.join(_results_dir, "test_fuse_panel.jsonl")
panel_module.run(base_url, "0g/fusion-preview", list(cfg.PANEL_MODELS), fuse_panel_path, limit=3,
                  experiment="test-fuse-panel-src")

fuse_out_path = os.path.join(_results_dir, "test_fuse_out.jsonl")
FUSE_EXP = "test-fuse-exp"
for f in glob.glob(os.path.join(llm_client.LOG_DIR, f"{FUSE_EXP}__*")):
    os.remove(f)
fuse_module.run(base_url, "0g/fusion-preview", fuse_panel_path, fuse_out_path, experiment=FUSE_EXP)

with open(fuse_out_path, encoding="utf-8") as f:
    fuse_rows = [json.loads(l) for l in f]
with open(fuse_panel_path, encoding="utf-8") as f:
    fuse_panel_rows = [json.loads(l) for l in f]
check("eval.fuse writes one row per panel-file row", len(fuse_rows) == 3)
check("every row has schema + a real fusion answer + the SAME panel it was given",
      all(r["schema"] == "0g.fusion_eval.gpqa.replay.v1" and r["fusion"]["content"] and r["panel"] == pr["panel"]
          for r, pr in zip(fuse_rows, fuse_panel_rows)))

fuse_panel_logs = glob.glob(os.path.join(llm_client.LOG_DIR, f"{FUSE_EXP}__panel__*"))
fuse_judge_logs = glob.glob(os.path.join(llm_client.LOG_DIR, f"{FUSE_EXP}__judge__*"))
fuse_synth_logs = glob.glob(os.path.join(llm_client.LOG_DIR, f"{FUSE_EXP}__synthesis__*"))
check("eval.fuse makes ZERO fresh panel calls (the panel is used exactly as given)", fuse_panel_logs == [])
check("eval.fuse DOES make the judge call -- this is the one step meant to cost it", len(fuse_judge_logs) == 1)
check("eval.fuse DOES make the synthesis call", len(fuse_synth_logs) == 1)
for f in fuse_judge_logs + fuse_synth_logs:
    os.remove(f)

_orig_call_api_f = fuse_module.call_api
_fcalls = {"n": 0}


def _counting_call_api_f(*a, **kw):
    _fcalls["n"] += 1
    return _orig_call_api_f(*a, **kw)


fuse_module.call_api = _counting_call_api_f
try:
    stderr_f = io.StringIO()
    with contextlib.redirect_stderr(stderr_f):
        fuse_module.run(base_url, "0g/fusion-preview", fuse_panel_path, fuse_out_path, experiment=FUSE_EXP)
    check("re-running eval.fuse against the same panel/out makes zero fresh calls (fully resumed)",
          _fcalls["n"] == 0)
    check("resume prints a clear skipped= message", "eval.fuse_resumed skipped=3" in stderr_f.getvalue())
finally:
    fuse_module.call_api = _orig_call_api_f

# a panel row that itself failed (or has no panel) must be skipped, not crash,
# and marked failed in the fuse output -- no call attempted for it
panel_rows_for_fail = list(fuse_panel_rows)
panel_rows_for_fail[1] = {**panel_rows_for_fail[1], "failed": True, "error": "pretend panel build failed", "panel": None}
failed_panel_path = os.path.join(_results_dir, "test_fuse_panel_failed.jsonl")
with open(failed_panel_path, "w", encoding="utf-8") as f:
    for r in panel_rows_for_fail:
        f.write(json.dumps(r) + "\n")

fuse_from_failed_path = os.path.join(_results_dir, "test_fuse_from_failed.jsonl")
_fcalls["n"] = 0
fuse_module.call_api = _counting_call_api_f
try:
    stderr_ff = io.StringIO()
    with contextlib.redirect_stderr(stderr_ff):
        fuse_module.run(base_url, "0g/fusion-preview", failed_panel_path, fuse_from_failed_path)
    check("a failed/missing panel row costs zero calls (skipped before any HTTP happens)", _fcalls["n"] == 2)
finally:
    fuse_module.call_api = _orig_call_api_f
with open(fuse_from_failed_path, encoding="utf-8") as f:
    fuse_from_failed_rows = [json.loads(l) for l in f]
check("the row whose panel had failed is carried through as failed, not crashed on",
      fuse_from_failed_rows[1].get("failed") is True)
check("the other 2 rows fused normally", all(fuse_from_failed_rows[i]["fusion"]["content"] for i in (0, 2)))
check("eval.fuse prints a clear 'no panel available' skip message naming the question_id",
      "eval.fuse_question_skipped" in stderr_ff.getvalue())
os.remove(failed_panel_path)
os.remove(fuse_from_failed_path)

# a call that fails must be caught and marked, not abort the run
def _fail_fuse_q1(url, model, msgs, **kw):
    if kw.get("question_id") == 1:
        raise RuntimeError("simulated fuse failure for question 1")
    return _orig_call_api_f(url, model, msgs, **kw)


fuse_catch_path = os.path.join(_results_dir, "test_fuse_catch.jsonl")
fuse_module.call_api = _fail_fuse_q1
try:
    fuse_module.run(base_url, "0g/fusion-preview", fuse_panel_path, fuse_catch_path)
finally:
    fuse_module.call_api = _orig_call_api_f
with open(fuse_catch_path, encoding="utf-8") as f:
    fuse_catch_rows = [json.loads(l) for l in f]
check("eval.fuse writes one row per question even when one question's call fails", len(fuse_catch_rows) == 3)
check("the failed question is marked failed with the error captured, others unaffected",
      fuse_catch_rows[1].get("failed") is True and "simulated fuse failure" in fuse_catch_rows[1].get("error", "")
      and fuse_catch_rows[0]["fusion"]["content"] and fuse_catch_rows[2]["fusion"]["content"])
os.remove(fuse_catch_path)

# 200-but-wrong-shape must ALSO be caught (same try-block-scope lesson)
def _malformed_shape_fuse(url, model, msgs, **kw):
    return {"choices": []}  # 200-ok, valid JSON, but empty choices -> IndexError on use


fuse_shape_path = os.path.join(_results_dir, "test_fuse_shape.jsonl")
fuse_module.call_api = _malformed_shape_fuse
try:
    fuse_module.run(base_url, "0g/fusion-preview", fuse_panel_path, fuse_shape_path)
finally:
    fuse_module.call_api = _orig_call_api_f
with open(fuse_shape_path, encoding="utf-8") as f:
    fuse_shape_rows = [json.loads(l) for l in f]
check("eval.fuse doesn't crash when a call succeeds but returns a malformed/wrong-shape response",
      len(fuse_shape_rows) == 3 and all(r.get("failed") is True for r in fuse_shape_rows))
os.remove(fuse_shape_path)
os.remove(fuse_out_path)
os.remove(fuse_panel_path)

# --- 17. eval.baseline: independent of any panel/fusion file, accumulates
#          across repeated calls into the same --out -------------------------
baseline_out_path = os.path.join(_results_dir, "test_baseline_out.jsonl")
baseline_module.run(base_url, ["baseline-a"], baseline_out_path, limit=3, experiment="test-baseline-a")
with open(baseline_out_path, encoding="utf-8") as f:
    baseline_rows1 = [json.loads(l) for l in f]
check("eval.baseline writes one row per question, needs no panel/fusion file at all",
      len(baseline_rows1) == 3 and all(r["schema"] == "0g.fusion_eval.gpqa.baselines.v1" for r in baseline_rows1))
check("each row has exactly the 1 requested baseline model",
      all([b["model"] for b in r["baselines"]] == ["baseline-a"] for r in baseline_rows1))

baseline_module.run(base_url, ["baseline-b"], baseline_out_path, limit=3, experiment="test-baseline-b")
with open(baseline_out_path, encoding="utf-8") as f:
    baseline_rows2 = [json.loads(l) for l in f]
check("a 2nd eval.baseline call for a different model accumulates alongside the first, doesn't replace it",
      all({b["model"] for b in r["baselines"]} == {"baseline-a", "baseline-b"} for r in baseline_rows2))

_orig_call_api_b = baseline_module.call_api
_bcalls = {"n": 0}


def _counting_call_api_b(*a, **kw):
    _bcalls["n"] += 1
    return _orig_call_api_b(*a, **kw)


baseline_module.call_api = _counting_call_api_b
try:
    stderr_b = io.StringIO()
    with contextlib.redirect_stderr(stderr_b):
        baseline_module.run(base_url, ["baseline-a", "baseline-b"], baseline_out_path, limit=3,
                             experiment="test-baseline-resume")
    check("re-running eval.baseline for models already present makes zero new calls", _bcalls["n"] == 0)
    check("resume prints a clear skipped= message", "eval.baseline_resumed skipped=3" in stderr_b.getvalue())
finally:
    baseline_module.call_api = _orig_call_api_b

# a previously-failed model entry must be retried, not permanently poisoned
baseline_module.call_api = lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("baseline down"))
try:
    with contextlib.redirect_stderr(io.StringIO()):
        baseline_module.run(base_url, ["flaky"], baseline_out_path, limit=3, experiment="test-baseline-flaky")
finally:
    baseline_module.call_api = _orig_call_api_b
with open(baseline_out_path, encoding="utf-8") as f:
    flaky_rows = [json.loads(l) for l in f]
check("a failed baseline model is recorded as failed, not silently dropped",
      all(any(b["model"] == "flaky" and b.get("failed") for b in r["baselines"]) for r in flaky_rows))

baseline_module.run(base_url, ["flaky"], baseline_out_path, limit=3, experiment="test-baseline-flaky-retry")
with open(baseline_out_path, encoding="utf-8") as f:
    flaky_retry_rows = [json.loads(l) for l in f]
check("re-running eval.baseline for that model retries and replaces the failed entry",
      all(any(b["model"] == "flaky" and not b.get("failed") for b in r["baselines"]) for r in flaky_retry_rows))
check("...and the other 2 models are untouched",
      all({"baseline-a", "baseline-b"} <= {b["model"] for b in r["baselines"]} for r in flaky_retry_rows))
os.remove(baseline_out_path)

# a call raising must not abort the run -- caught, marked failed, others continue
_bfail_calls = {"n": 0}


def _fail_2nd_baseline_call(url, model, msgs, **kw):
    _bfail_calls["n"] += 1
    if _bfail_calls["n"] == 2:
        raise RuntimeError("simulated baseline failure for the 2nd question")
    return _orig_call_api_b(url, model, msgs, **kw)


baseline_fail_path = os.path.join(_results_dir, "test_baseline_fail.jsonl")
baseline_module.call_api = _fail_2nd_baseline_call
try:
    baseline_module.run(base_url, ["baseline-c"], baseline_fail_path, limit=3, experiment="test-baseline-fail")
finally:
    baseline_module.call_api = _orig_call_api_b
with open(baseline_fail_path, encoding="utf-8") as f:
    baseline_fail_rows = [json.loads(l) for l in f]
check("eval.baseline writes one row per question even when one question's call fails",
      len(baseline_fail_rows) == 3)
check("exactly one row's baseline entry is marked failed, the others succeeded",
      sum(1 for r in baseline_fail_rows for b in r["baselines"] if b.get("failed")) == 1
      and sum(1 for r in baseline_fail_rows for b in r["baselines"] if not b.get("failed")) == 2)
os.remove(baseline_fail_path)

# 200-but-wrong-shape must ALSO be caught (same try-block-scope lesson)
def _malformed_shape_baseline(url, model, msgs, **kw):
    return {"choices": []}


baseline_shape_path = os.path.join(_results_dir, "test_baseline_shape.jsonl")
baseline_module.call_api = _malformed_shape_baseline
try:
    baseline_module.run(base_url, ["shape-model"], baseline_shape_path, limit=2, experiment="test-baseline-shape")
finally:
    baseline_module.call_api = _orig_call_api_b
with open(baseline_shape_path, encoding="utf-8") as f:
    baseline_shape_rows = [json.loads(l) for l in f]
check("eval.baseline doesn't crash on a malformed/wrong-shape response, marks that entry failed",
      len(baseline_shape_rows) == 2
      and all(any(b["model"] == "shape-model" and b.get("failed") for b in r["baselines"])
              for r in baseline_shape_rows))
os.remove(baseline_shape_path)

# duplicate model names in --models must be deduped (exercised at the real CLI)
dedupe_out = os.path.join(_results_dir, "test_baseline_dedupe.jsonl")
_cli_dedupe = _cli("eval.baseline", "--baseline-url", base_url, "--models", " dup , dup ,other",
                    "--out", dedupe_out, "--limit", "1", "--experiment", "test-baseline-dedupe")
check("the eval.baseline CLI runs end to end and writes its --out", _cli_dedupe.returncode == 0)
_dedupe_row = json.loads(open(dedupe_out, encoding="utf-8").readline())
check("a repeated --models name is collapsed to one call, whitespace stripped, order kept",
      [b["model"] for b in _dedupe_row["baselines"]] == ["dup", "other"])
os.remove(dedupe_out)

# --- 18. eval.grade: merges 1+ files by question_id before scoring ---------
grade_fuse_path = os.path.join(_results_dir, "test_grade_fuse.jsonl")
grade_panel_path = os.path.join(_results_dir, "test_grade_panel.jsonl")
panel_module.run(base_url, "0g/fusion-preview", list(cfg.PANEL_MODELS), grade_panel_path, limit=3,
                  experiment="test-grade-panel")
fuse_module.run(base_url, "0g/fusion-preview", grade_panel_path, grade_fuse_path, experiment="test-grade-fuse")

grade_baseline_path = os.path.join(_results_dir, "test_grade_baseline.jsonl")
baseline_module.run(base_url, ["bl-x", "bl-y"], grade_baseline_path, limit=3, experiment="test-grade-baseline")

single_result = grade_replay(grade_fuse_path)
check("grade_replay on a single fuse file scores fusion, no baselines present",
      single_result["fusion"]["n"] == 3 and single_result["baselines"] == {})

merged_result = grade_replay(grade_fuse_path, grade_baseline_path)
check("grade_replay merges a fuse file + a baseline file by question_id into one scoreboard",
      merged_result["fusion"]["n"] == 3 and set(merged_result["baselines"]) == {"bl-x", "bl-y"}
      and merged_result["baselines"]["bl-x"]["n"] == 3)

merged_rows = load_rows([grade_fuse_path, grade_baseline_path])
check("load_rows produces exactly one merged row per question_id, carrying both fusion and baselines",
      len(merged_rows) == 3 and all("fusion" in r and "baselines" in r for r in merged_rows))

_cli_grade = _cli("eval.grade", grade_fuse_path, grade_baseline_path)
check("the eval.grade CLI accepts multiple files and runs end to end", _cli_grade.returncode == 0)
check("the CLI's output matches grade_replay's own return value",
      json.loads(_cli_grade.stdout) == merged_result)

os.remove(grade_fuse_path)
os.remove(grade_panel_path)
os.remove(grade_baseline_path)

# blank trailing line tolerance
blank_path = os.path.join(_results_dir, "test_grade_blank.jsonl")
with open(blank_path, "w", encoding="utf-8") as f:
    f.write(json.dumps({"question_id": 0, "correct_letter": "A", "fusion": {"content": "Final Answer: A"}}) + "\n\n")
check("grade_replay tolerates a blank trailing line", grade_replay(blank_path)["fusion"]["n"] == 1)
os.remove(blank_path)

# hand-computed scoring check across every row shape eval.fuse/eval.baseline
# can emit, checked against hand-computed numbers -- not just "it didn't crash"
shapes_path = os.path.join(_results_dir, "test_grade_shapes.jsonl")
with open(shapes_path, "w", encoding="utf-8") as f:
    for _r in (
        {"question_id": 0, "correct_letter": "A", "fusion": {"content": "Final Answer: A"},
         "baselines": [{"model": "A", "content": "Final Answer: A"}, {"model": "B", "content": "Final Answer: C"}]},
        {"question_id": 1, "correct_letter": "B", "fusion": {"content": "Final Answer: B"},
         "baselines": [{"model": "A", "content": "no letter here"}, {"model": "B", "failed": True, "error": "boom"}]},
        {"question_id": 2, "correct_letter": "C", "failed": True, "error": "outage"},
        {"question_id": 3, "correct_letter": "D", "fusion": {"content": "Final Answer: D"}, "baselines": []},
    ):
        f.write(json.dumps(_r) + "\n")
shape_scores = grade_replay(shapes_path)
check("grade_replay: fusion scores 3/4 with the row-level failure as call_failed",
      shape_scores["fusion"] == {"accuracy": 0.75, "correct": 3, "extraction_failed": 0, "call_failed": 1,
                                  "no_ground_truth": 0, "n": 4})
check("grade_replay: baseline A = 1 correct, 1 unparseable, 2 rows it was never run for",
      shape_scores["baselines"]["A"] == {"accuracy": 0.25, "correct": 1, "extraction_failed": 1, "call_failed": 2,
                                          "no_ground_truth": 0, "n": 4})
check("grade_replay: baseline B = answered once (wrong), failed/absent the other 3",
      shape_scores["baselines"]["B"] == {"accuracy": 0.0, "correct": 0, "extraction_failed": 0, "call_failed": 3,
                                          "no_ground_truth": 0, "n": 4})
check("grade_replay lists exactly the models that appear in any baselines list",
      sorted(shape_scores["baselines"]) == ["A", "B"])
os.remove(shapes_path)

# --- 19. resume-safety: refusing to reuse rows written against a DIFFERENT
#          dataset/question set under the same --out (eval.panel, eval.baseline,
#          and eval.fuse keyed off its --panel file's own question set) -----
def _write_dataset(path, tag, n=3):
    with open(path, "w", encoding="utf-8") as f:
        for i in range(n):
            f.write(json.dumps({"Question": f"{tag} question {i}", "Correct Answer": f"{tag}-right-{i}",
                                 "Incorrect Answer 1": f"{tag}-w1-{i}", "Incorrect Answer 2": f"{tag}-w2-{i}",
                                 "Incorrect Answer 3": f"{tag}-w3-{i}"}) + "\n")


_safe_default = gpqa_tasks_module.REAL_DEFAULT_PATH  # the forced-sample default set up at the top of this file
ds_a = os.path.join(_results_dir, "test_ds_a.jsonl")
ds_b = os.path.join(_results_dir, "test_ds_b.jsonl")
_write_dataset(ds_a, "DATASET-A")
_write_dataset(ds_b, "DATASET-B")

swap_panel_path = os.path.join(_results_dir, "test_swap_panel.jsonl")
if os.path.exists(swap_panel_path):
    os.remove(swap_panel_path)
try:
    gpqa_tasks_module.REAL_DEFAULT_PATH = ds_a
    panel_module.run(base_url, "0g/fusion-preview", list(cfg.PANEL_MODELS), swap_panel_path, limit=3,
                      experiment="test-swap")
    with contextlib.redirect_stderr(io.StringIO()):
        panel_module.run(base_url, "0g/fusion-preview", list(cfg.PANEL_MODELS), swap_panel_path, limit=3,
                          experiment="test-swap")
    check("resume across an UNCHANGED dataset is unaffected by the identity check",
          sum(1 for _ in open(swap_panel_path, encoding="utf-8")) == 3)

    gpqa_tasks_module.REAL_DEFAULT_PATH = ds_b
    try:
        panel_module.run(base_url, "0g/fusion-preview", list(cfg.PANEL_MODELS), swap_panel_path, limit=3,
                          experiment="test-swap")
        check("eval.panel refuses to resume rows written against a DIFFERENT dataset", False)
    except ResumeMismatchError:
        check("eval.panel refuses to resume rows written against a DIFFERENT dataset", True)
    check("the mismatch aborts before --out is opened for writing, leaving the old rows intact",
          all("DATASET-A" in json.loads(l)["instruction"] for l in open(swap_panel_path, encoding="utf-8")))

    panel_module.run(base_url, "0g/fusion-preview", list(cfg.PANEL_MODELS), swap_panel_path, limit=3,
                      experiment="test-swap", resume=False)
    check("--no-resume is still the documented escape hatch and recomputes from scratch",
          all("DATASET-B" in json.loads(l)["instruction"] for l in open(swap_panel_path, encoding="utf-8")))
finally:
    gpqa_tasks_module.REAL_DEFAULT_PATH = _safe_default
os.remove(swap_panel_path)

swap_baseline_path = os.path.join(_results_dir, "test_swap_baseline.jsonl")
if os.path.exists(swap_baseline_path):
    os.remove(swap_baseline_path)
try:
    gpqa_tasks_module.REAL_DEFAULT_PATH = ds_a
    baseline_module.run(base_url, ["bl"], swap_baseline_path, limit=3, experiment="test-swap-bl")
    gpqa_tasks_module.REAL_DEFAULT_PATH = ds_b
    try:
        baseline_module.run(base_url, ["bl"], swap_baseline_path, limit=3, experiment="test-swap-bl")
        check("eval.baseline refuses to resume rows written against a DIFFERENT dataset", False)
    except ResumeMismatchError:
        check("eval.baseline refuses to resume rows written against a DIFFERENT dataset", True)
finally:
    gpqa_tasks_module.REAL_DEFAULT_PATH = _safe_default
os.remove(swap_baseline_path)

# eval.fuse doesn't call load_tasks at all -- its resume-safety is keyed off
# the --panel FILE's own question set, so build two panels over different
# question sets directly and swap the --panel file under the same --out
fuse_swap_panel_a = os.path.join(_results_dir, "test_fuse_swap_a.jsonl")
fuse_swap_panel_b = os.path.join(_results_dir, "test_fuse_swap_b.jsonl")
try:
    gpqa_tasks_module.REAL_DEFAULT_PATH = ds_a
    panel_module.run(base_url, "0g/fusion-preview", list(cfg.PANEL_MODELS), fuse_swap_panel_a, limit=3,
                      experiment="test-fuse-swap-a")
    gpqa_tasks_module.REAL_DEFAULT_PATH = ds_b
    panel_module.run(base_url, "0g/fusion-preview", list(cfg.PANEL_MODELS), fuse_swap_panel_b, limit=3,
                      experiment="test-fuse-swap-b")
finally:
    gpqa_tasks_module.REAL_DEFAULT_PATH = _safe_default
fuse_swap_out = os.path.join(_results_dir, "test_fuse_swap_out.jsonl")
if os.path.exists(fuse_swap_out):
    os.remove(fuse_swap_out)
fuse_module.run(base_url, "0g/fusion-preview", fuse_swap_panel_a, fuse_swap_out, experiment="test-fuse-swap")
try:
    fuse_module.run(base_url, "0g/fusion-preview", fuse_swap_panel_b, fuse_swap_out, experiment="test-fuse-swap")
    check("eval.fuse refuses to resume rows written against a different --panel file's questions", False)
except ResumeMismatchError:
    check("eval.fuse refuses to resume rows written against a different --panel file's questions", True)
os.remove(fuse_swap_panel_a)
os.remove(fuse_swap_panel_b)
os.remove(fuse_swap_out)
os.remove(ds_a)
os.remove(ds_b)

# --- 20. full command-chain smoke test via the actual CLIs (subprocess),
#          exactly the sequence intended for a real run: build the fixed
#          panel once, fuse it, build a --reuse variant panel, fuse that too,
#          add baselines completely independently, grade fusion+baselines
#          together -- and restate the cost guarantee at the CLI level -----
cli_panel_out = os.path.join(_results_dir, "cli-panel-fixed.jsonl")
r1 = _cli("eval.panel", "--fusion-url", base_url, "--models", ",".join(cfg.PANEL_MODELS),
          "--out", cli_panel_out, "--limit", "2", "--experiment", "cli-panel-fixed")
check("CLI: eval.panel runs end to end", r1.returncode == 0 and r1.stderr == "")

cli_fuse_out = os.path.join(_results_dir, "cli-fuse-fixed.jsonl")
r2 = _cli("eval.fuse", "--fusion-url", base_url, "--panel", cli_panel_out,
          "--out", cli_fuse_out, "--experiment", "cli-fuse-fixed")
check("CLI: eval.fuse runs end to end on the panel it just built", r2.returncode == 0)

cli_variant_out = os.path.join(_results_dir, "cli-panel-variant.jsonl")
r3 = _cli("eval.panel", "--fusion-url", base_url, "--models", ",".join(cfg.PANEL_MODELS) + ",candidate-cli",
          "--reuse", cli_panel_out, "--out", cli_variant_out, "--limit", "2", "--experiment", "cli-panel-variant")
check("CLI: eval.panel --reuse builds a variant panel end to end", r3.returncode == 0)
with open(cli_variant_out, encoding="utf-8") as f:
    cli_variant_rows = [json.loads(l) for l in f]
check("CLI-built variant panel reused the fixed models and added the candidate",
      all(len(r["panel"]) == len(cfg.PANEL_MODELS) + 1 for r in cli_variant_rows))

cli_baseline_out = os.path.join(_results_dir, "cli-baselines.jsonl")
r4 = _cli("eval.baseline", "--baseline-url", base_url, "--models", "gpt-baseline-cli,claude-baseline-cli",
          "--out", cli_baseline_out, "--limit", "2", "--experiment", "cli-baselines")
check("CLI: eval.baseline runs end to end, independent of any panel/fusion file", r4.returncode == 0)

cli_variant_fuse_out = os.path.join(_results_dir, "cli-fuse-variant.jsonl")
r5 = _cli("eval.fuse", "--fusion-url", base_url, "--panel", cli_variant_out,
          "--out", cli_variant_fuse_out, "--experiment", "cli-fuse-variant")
check("CLI: eval.fuse on the variant panel runs end to end", r5.returncode == 0)

r6 = _cli("eval.grade", cli_variant_fuse_out, cli_baseline_out)
check("CLI: eval.grade merges the variant's fusion result with the independent baselines", r6.returncode == 0)
cli_grade_result = json.loads(r6.stdout)
check("the merged CLI grade result has both the fusion score and both baseline models scored",
      cli_grade_result["fusion"]["n"] == 2 and set(cli_grade_result["baselines"]) == {"gpt-baseline-cli", "claude-baseline-cli"})

check("*** end-to-end via the real CLI: building/reusing panels made zero judge/synthesis calls ***",
      glob.glob(os.path.join(llm_client.LOG_DIR, "cli-panel-fixed__judge__*")) == []
      and glob.glob(os.path.join(llm_client.LOG_DIR, "cli-panel-fixed__synthesis__*")) == []
      and glob.glob(os.path.join(llm_client.LOG_DIR, "cli-panel-variant__judge__*")) == []
      and glob.glob(os.path.join(llm_client.LOG_DIR, "cli-panel-variant__synthesis__*")) == [])

for f in glob.glob(os.path.join(llm_client.LOG_DIR, "cli-*")):
    os.remove(f)
for p in (cli_panel_out, cli_fuse_out, cli_variant_out, cli_baseline_out, cli_variant_fuse_out):
    os.remove(p)

# --- 21. regressions found in the adversarial review of this redesign ------

# 21a. eval.grade must refuse to merge two files that disagree about what a
#      shared question_id actually IS -- otherwise the score depends on
#      argument ORDER (whichever file is seen last wins the dict-merge).
from eval.grade import GradeMergeError  # noqa: E402

_conflict_a = os.path.join(_results_dir, "test_conflict_a.jsonl")
_conflict_b = os.path.join(_results_dir, "test_conflict_b.jsonl")
with open(_conflict_a, "w", encoding="utf-8") as f:
    f.write(json.dumps({"question_id": 0, "instruction": "question A", "correct_letter": "A",
                         "fusion": {"content": "Final Answer: A"}}) + "\n")
with open(_conflict_b, "w", encoding="utf-8") as f:
    f.write(json.dumps({"question_id": 0, "instruction": "a completely different question",
                         "correct_letter": "B", "baselines": [{"model": "x", "content": "Final Answer: B"}]}) + "\n")
for _order in ((_conflict_a, _conflict_b), (_conflict_b, _conflict_a)):
    try:
        grade_replay(*_order)
        check(f"eval.grade refuses to merge files that disagree on question_id 0 (order={_order})", False)
    except GradeMergeError:
        check(f"eval.grade refuses to merge files that disagree on question_id 0 (order={_order})", True)
# a row-level failure with no `instruction` at all carries no ground truth to
# conflict with -- must NOT be flagged as a false-positive conflict
_conflict_c = os.path.join(_results_dir, "test_conflict_c.jsonl")
with open(_conflict_c, "w", encoding="utf-8") as f:
    f.write(json.dumps({"question_id": 0, "correct_letter": "A", "failed": True, "error": "outage"}) + "\n")
check("a row-level failure (no instruction) merges fine alongside a real row for the same question_id",
      grade_replay(_conflict_a, _conflict_c)["fusion"]["n"] == 1)
for _p in (_conflict_a, _conflict_b, _conflict_c):
    os.remove(_p)

# 21b. schema guard: pointing --out (or eval.panel's --reuse) at a file
#      written by a DIFFERENT tool must refuse, not silently rebuild the row
#      and drop whatever that other tool already paid for.
_schema_src = os.path.join(_results_dir, "test_schema_src.jsonl")
panel_module.run(base_url, "0g/fusion-preview", list(cfg.PANEL_MODELS), _schema_src, limit=2,
                  experiment="test-schema-src")

_schema_baseline_out = os.path.join(_results_dir, "test_schema_baseline.jsonl")
import shutil as _shutil  # noqa: E402
_shutil.copy(_schema_src, _schema_baseline_out)
try:
    baseline_module.run(base_url, ["some-model"], _schema_baseline_out, limit=2, experiment="test-schema-b")
    check("eval.baseline refuses to reuse an --out written by eval.panel (different schema)", False)
except ResumeMismatchError:
    check("eval.baseline refuses to reuse an --out written by eval.panel (different schema)", True)
os.remove(_schema_baseline_out)

_schema_fuse_out = os.path.join(_results_dir, "test_schema_fuse.jsonl")
baseline_module.run(base_url, ["some-model"], _schema_fuse_out, limit=2, experiment="test-schema-f")
try:
    fuse_module.run(base_url, "0g/fusion-preview", _schema_src, _schema_fuse_out, experiment="test-schema-f2")
    check("eval.fuse refuses to reuse an --out written by eval.baseline (different schema)", False)
except ResumeMismatchError:
    check("eval.fuse refuses to reuse an --out written by eval.baseline (different schema)", True)
os.remove(_schema_fuse_out)

# the self-referential case: fuse.py's own --out pointed at a --panel-shaped
# file (e.g. --experiment accidentally collides with the panel's own name)
_schema_selffuse = os.path.join(_results_dir, "test_schema_selffuse.jsonl")
_shutil.copy(_schema_src, _schema_selffuse)
try:
    fuse_module.run(base_url, "0g/fusion-preview", _schema_src, _schema_selffuse, experiment="test-schema-self")
    check("eval.fuse refuses to treat a --panel-shaped --out as already-fused", False)
except ResumeMismatchError:
    check("eval.fuse refuses to treat a --panel-shaped --out as already-fused", True)
os.remove(_schema_selffuse)

_schema_panel_out = os.path.join(_results_dir, "test_schema_panel.jsonl")
baseline_module.run(base_url, ["some-model"], _schema_panel_out, limit=2, experiment="test-schema-p")
try:
    panel_module.run(base_url, "0g/fusion-preview", list(cfg.PANEL_MODELS), _schema_panel_out, limit=2,
                      experiment="test-schema-p2")
    check("eval.panel refuses to reuse an --out written by eval.baseline (different schema)", False)
except ResumeMismatchError:
    check("eval.panel refuses to reuse an --out written by eval.baseline (different schema)", True)
try:
    panel_module.run(base_url, "0g/fusion-preview", list(cfg.PANEL_MODELS), _schema_src, limit=2,
                      experiment="test-schema-p3", reuse_path=_schema_panel_out)
    check("eval.panel's --reuse refuses a file written by eval.baseline (different schema)", False)
except ResumeMismatchError:
    check("eval.panel's --reuse refuses a file written by eval.baseline (different schema)", True)
os.remove(_schema_panel_out)
os.remove(_schema_src)

# 21c. eval.fuse must carry forward rows outside the CURRENT --panel file's
#      question set even with no --limit -- that window can shrink for
#      reasons that have nothing to do with --limit (an interrupted
#      eval.panel run, a hand-edited/concatenated panel file).
_shrink_panel = os.path.join(_results_dir, "test_shrink_panel.jsonl")
panel_module.run(base_url, "0g/fusion-preview", list(cfg.PANEL_MODELS), _shrink_panel, limit=3,
                  experiment="test-shrink-panel")
_shrink_fuse = os.path.join(_results_dir, "test_shrink_fuse.jsonl")
fuse_module.run(base_url, "0g/fusion-preview", _shrink_panel, _shrink_fuse, experiment="test-shrink-fuse")
check("all 3 rows fused before the panel file shrinks", sum(1 for _ in open(_shrink_fuse, encoding="utf-8")) == 3)
with open(_shrink_panel, encoding="utf-8") as f:
    _first_row_only = f.readline()
with open(_shrink_panel, "w", encoding="utf-8") as f:
    f.write(_first_row_only)  # simulate an interrupted/hand-edited panel file: 3 rows -> 1
fuse_module.run(base_url, "0g/fusion-preview", _shrink_panel, _shrink_fuse, experiment="test-shrink-fuse")
check("re-running eval.fuse against a SHRUNK --panel file, with no --limit, keeps the other "
      "already-paid fusion rows instead of deleting them",
      sum(1 for _ in open(_shrink_fuse, encoding="utf-8")) == 3)
os.remove(_shrink_panel)
os.remove(_shrink_fuse)

# 21d. eval.panel's --reuse must hard-fail on a path that doesn't exist,
#      instead of silently treating it as empty and re-calling everything.
_missing_reuse = os.path.join(_results_dir, "does_not_exist_reuse.jsonl")
_reuse_fail_out = os.path.join(_results_dir, "test_reuse_missing.jsonl")
try:
    panel_module.run(base_url, "0g/fusion-preview", list(cfg.PANEL_MODELS), _reuse_fail_out, limit=2,
                      experiment="test-reuse-missing", reuse_path=_missing_reuse)
    check("eval.panel --reuse raises on a nonexistent path instead of silently re-calling everything", False)
except FileNotFoundError:
    check("eval.panel --reuse raises on a nonexistent path instead of silently re-calling everything", True)
check("...and nothing was called before the check failed", not os.path.exists(_reuse_fail_out))

# 21e. eval.panel's --models must dedupe (matches eval.baseline's existing
#      behavior) -- a repeated model name must be called once, not twice,
#      and must not double-weight that model's vote in judge/synthesis.
_dedupe_panel_out = os.path.join(_results_dir, "test_panel_dedupe.jsonl")
_cli_panel_dedupe = _cli("eval.panel", "--fusion-url", base_url, "--models", " dup-panel , dup-panel ,other-panel",
                          "--out", _dedupe_panel_out, "--limit", "1", "--experiment", "test-panel-dedupe")
check("the eval.panel CLI runs end to end", _cli_panel_dedupe.returncode == 0)
_dedupe_panel_row = json.loads(open(_dedupe_panel_out, encoding="utf-8").readline())
check("a repeated --models name is collapsed to one call, whitespace stripped, order kept",
      [p["model"] for p in _dedupe_panel_row["panel"]] == ["dup-panel", "other-panel"])
os.remove(_dedupe_panel_out)

# eval.panel/eval.baseline must reject an empty --models instead of silently
# "succeeding" with 0 models and a misleading "already had everything" message
for _mod_name in ("eval.panel", "eval.baseline"):
    _empty_out = os.path.join(_results_dir, "test_empty_models.jsonl")
    _flag = "--models"
    _url_flag = "--fusion-url" if _mod_name == "eval.panel" else "--baseline-url"
    _r = _cli(_mod_name, _url_flag, base_url, _flag, "", "--out", _empty_out, "--limit", "1")
    check(f"{_mod_name} CLI rejects an empty --models instead of silently succeeding with 0 models",
          _r.returncode != 0 and not os.path.exists(_empty_out))

# 21f. the unified "everything already cached" skip branch: (1) a previously
#      FAILED row still carries whatever was already cached BEFORE that
#      failing call (not thrown away), so a retry doesn't re-pay for it;
#      (2) a row with EXTRA members beyond the current --models gets trimmed
#      to exactly --models, not kept verbatim; (3) a row that's now fully
#      covered never carries a stale "failed": true forward.
#      (Note: a single call batching several NEW models together is still
#      all-or-nothing if one of them fails -- that failure mode lives in
#      pipeline.run_panel's ThreadPoolExecutor, not in eval.panel, and isn't
#      addressed here. What IS fixed: members cached from an EARLIER run/
#      --reuse must survive a LATER call's failure, instead of being
#      silently dropped from the row alongside it.)
_retry_panel_path = os.path.join(_results_dir, "test_panel_retry_cached.jsonl")
panel_module.run(base_url, "0g/fusion-preview", ["good-member"], _retry_panel_path, limit=1,
                  experiment="test-retry-cached")


def _fail_one_model(url, model, msgs, **kw):
    if "flaky-member" in (kw.get("extra_panel_models") or []):
        raise RuntimeError("simulated flaky-member failure")
    return _orig_call_api_p(url, model, msgs, **kw)


panel_module.call_api = _fail_one_model
try:
    panel_module.run(base_url, "0g/fusion-preview", ["good-member", "flaky-member"], _retry_panel_path,
                      limit=1, experiment="test-retry-cached")
finally:
    panel_module.call_api = _orig_call_api_p
with open(_retry_panel_path, encoding="utf-8") as f:
    _first_fail_row = json.loads(f.readline())
check("a batch failure's row still carries the member that was ALREADY cached before this call, "
      "not just the error",
      _first_fail_row.get("failed") is True
      and [p["model"] for p in _first_fail_row.get("panel", [])] == ["good-member"])

_pcalls["n"] = 0
panel_module.call_api = _counting_call_api_p
try:
    panel_module.run(base_url, "0g/fusion-preview", ["good-member", "flaky-member"], _retry_panel_path,
                      limit=1, experiment="test-retry-cached")
    check("retrying only re-calls the model that actually failed, not the one already cached",
          _pcalls["n"] == 1)
finally:
    panel_module.call_api = _orig_call_api_p
with open(_retry_panel_path, encoding="utf-8") as f:
    _retried_row = json.loads(f.readline())
check("after a successful retry, the row is no longer marked failed and has both members",
      "failed" not in _retried_row and {p["model"] for p in _retried_row["panel"]} == {"good-member", "flaky-member"})

# now ask for a SMALLER panel than what's already cached (extra member from
# an earlier request) -- must trim to exactly --models, not keep the extra
panel_module.run(base_url, "0g/fusion-preview", ["good-member"], _retry_panel_path, limit=1,
                  experiment="test-retry-cached")
with open(_retry_panel_path, encoding="utf-8") as f:
    _trimmed_row = json.loads(f.readline())
check("requesting a SMALLER --models than what's cached trims the row to exactly --models "
      "(never silently keeps a stale extra member)",
      [p["model"] for p in _trimmed_row["panel"]] == ["good-member"] and "failed" not in _trimmed_row)
os.remove(_retry_panel_path)

# 21g. eval.fuse must refuse a --panel file with duplicate question_ids
#      instead of silently paying for judge+synthesis twice for that question.
_dup_qid_panel = os.path.join(_results_dir, "test_dup_qid_panel.jsonl")
panel_module.run(base_url, "0g/fusion-preview", list(cfg.PANEL_MODELS), _dup_qid_panel, limit=2,
                  experiment="test-dup-qid-panel")
with open(_dup_qid_panel, encoding="utf-8") as f:
    _dup_qid_rows = [json.loads(l) for l in f]
with open(_dup_qid_panel, "w", encoding="utf-8") as f:
    for r in _dup_qid_rows + [_dup_qid_rows[0]]:  # duplicate question_id 0
        f.write(json.dumps(r) + "\n")
_dup_qid_fuse_out = os.path.join(_results_dir, "test_dup_qid_fuse.jsonl")
try:
    fuse_module.run(base_url, "0g/fusion-preview", _dup_qid_panel, _dup_qid_fuse_out, experiment="test-dup-qid-fuse")
    check("eval.fuse refuses a --panel file with a duplicate question_id", False)
except ValueError as e:
    check("eval.fuse refuses a --panel file with a duplicate question_id", "duplicate question_id" in str(e))
check("...and nothing was called before the check failed", not os.path.exists(_dup_qid_fuse_out))
os.remove(_dup_qid_panel)

# 21h. a --panel row missing correct_letter (but with everything else) must
#      NOT crash even when the call SUCCEEDS -- it's ungradeable, not a call
#      failure, so eval.fuse must write it through normally (with
#      correct_letter: null) and let eval.grade's no_ground_truth bucket
#      flag it, instead of the row itself pretending nothing's wrong.
_no_letter_panel_path = os.path.join(_results_dir, "test_no_letter_panel.jsonl")
with open(_no_letter_panel_path, "w", encoding="utf-8") as f:
    f.write(json.dumps({"schema": panel_module.SCHEMA, "question_id": 0, "instruction": "Q?",
                         # correct_letter deliberately missing
                         "panel": [{"model": m, "content": "x", "reasoning": "y"} for m in cfg.PANEL_MODELS]}) + "\n")
_no_letter_fuse_out = os.path.join(_results_dir, "test_no_letter_fuse_out.jsonl")
fuse_module.run(base_url, "0g/fusion-preview", _no_letter_panel_path, _no_letter_fuse_out,
                experiment="test-no-letter-row")
with open(_no_letter_fuse_out, encoding="utf-8") as f:
    _no_letter_rows = [json.loads(l) for l in f]
check("a panel row missing correct_letter fuses successfully (not a call failure), correct_letter stays null",
      len(_no_letter_rows) == 1 and "failed" not in _no_letter_rows[0]
      and _no_letter_rows[0]["correct_letter"] is None and _no_letter_rows[0]["fusion"]["content"])
os.remove(_no_letter_panel_path)
os.remove(_no_letter_fuse_out)

# FAKE mode's synthesis never actually emits "Final Answer: X" (it's a
# canned stand-in, not instruction-following), so grading the bucketing
# itself needs a hand-built row with a real parseable answer alongside the
# missing ground truth -- exercises grade._score directly, not eval.fuse.
_no_letter_graded_path = os.path.join(_results_dir, "test_no_letter_graded.jsonl")
with open(_no_letter_graded_path, "w", encoding="utf-8") as f:
    f.write(json.dumps({"question_id": 0, "fusion": {"content": "Final Answer: A"}}) + "\n")  # no correct_letter
_no_letter_score = grade_replay(_no_letter_graded_path)["fusion"]
check("eval.grade buckets a missing correct_letter as no_ground_truth, not as a wrong answer",
      _no_letter_score["no_ground_truth"] == 1 and _no_letter_score["correct"] == 0
      and _no_letter_score["call_failed"] == 0 and _no_letter_score["extraction_failed"] == 0)
os.remove(_no_letter_graded_path)

# 21i. a --panel row missing `instruction` entirely DOES crash the try block
#      (there's no message to send) -- must be caught cleanly by the except
#      handler's .get()-based failure row, not crash the whole run (the
#      original try-block-scope lesson, now exercised via a field the
#      except branch doesn't already special-case).
_no_instruction_panel_path = os.path.join(_results_dir, "test_no_instruction_panel.jsonl")
with open(_no_instruction_panel_path, "w", encoding="utf-8") as f:
    f.write(json.dumps({"schema": panel_module.SCHEMA, "question_id": 0, "correct_letter": "A",
                         # instruction deliberately missing
                         "panel": [{"model": m, "content": "x", "reasoning": "y"} for m in cfg.PANEL_MODELS]}) + "\n")
_no_instruction_fuse_out = os.path.join(_results_dir, "test_no_instruction_fuse_out.jsonl")
fuse_module.run(base_url, "0g/fusion-preview", _no_instruction_panel_path, _no_instruction_fuse_out,
                experiment="test-no-instruction-row")
with open(_no_instruction_fuse_out, encoding="utf-8") as f:
    _no_instruction_rows = [json.loads(l) for l in f]
check("a panel row missing instruction is caught cleanly (KeyError inside try, handled by the except "
      "branch's .get()-based row, not a 2nd crash from the except handler itself)",
      len(_no_instruction_rows) == 1 and _no_instruction_rows[0].get("failed") is True
      and "instruction" in _no_instruction_rows[0].get("error", ""))
os.remove(_no_instruction_panel_path)
os.remove(_no_instruction_fuse_out)

# --- 22. regressions found in the SECOND independent review round ----------

# 22a. eval.panel/eval.baseline must carry forward already-paid rows when the
#      DATASET shrinks between runs, not just when --limit is set -- the
#      window they must protect is `expected` (whatever load_tasks() returns
#      right now), which can shrink for reasons that have nothing to do with
#      this run's own --limit (a re-download using ITS OWN --limit, writing
#      over the same default path a full download used).
_ds_full = os.path.join(_results_dir, "test_ds_full.jsonl")
_ds_shrunk = os.path.join(_results_dir, "test_ds_shrunk.jsonl")
_write_dataset(_ds_full, "FULL", n=5)
_write_dataset(_ds_shrunk, "FULL", n=2)  # same questions, just fewer of them -- like a smaller re-download

_shrink_panel_path = os.path.join(_results_dir, "test_dataset_shrink_panel.jsonl")
_shrink_baseline_path = os.path.join(_results_dir, "test_dataset_shrink_baseline.jsonl")
try:
    gpqa_tasks_module.REAL_DEFAULT_PATH = _ds_full
    panel_module.run(base_url, "0g/fusion-preview", ["m-a"], _shrink_panel_path, experiment="test-ds-shrink-panel")
    baseline_module.run(base_url, ["b-a"], _shrink_baseline_path, experiment="test-ds-shrink-baseline")
    check("5 rows written before the dataset shrinks", sum(1 for _ in open(_shrink_panel_path)) == 5)

    gpqa_tasks_module.REAL_DEFAULT_PATH = _ds_shrunk
    stderr_shrink_p = io.StringIO()
    with contextlib.redirect_stderr(stderr_shrink_p):
        panel_module.run(base_url, "0g/fusion-preview", ["m-a"], _shrink_panel_path, experiment="test-ds-shrink-panel")
    check("eval.panel keeps rows outside a SHRUNK dataset (no --limit involved), doesn't delete them",
          sum(1 for _ in open(_shrink_panel_path)) == 5)
    check("eval.panel reports the carried-over count by name", "eval.panel_carried_over=3" in stderr_shrink_p.getvalue())

    stderr_shrink_b = io.StringIO()
    with contextlib.redirect_stderr(stderr_shrink_b):
        baseline_module.run(base_url, ["b-a"], _shrink_baseline_path, experiment="test-ds-shrink-baseline")
    check("eval.baseline keeps rows outside a SHRUNK dataset too",
          sum(1 for _ in open(_shrink_baseline_path)) == 5)
    check("eval.baseline reports the carried-over count by name",
          "eval.baseline_carried_over=3" in stderr_shrink_b.getvalue())
finally:
    gpqa_tasks_module.REAL_DEFAULT_PATH = _safe_default
for _p in (_ds_full, _ds_shrunk, _shrink_panel_path, _shrink_baseline_path):
    os.remove(_p)

# 22b. eval.panel must say out loud when --models drops a model that was
#      already cached -- silent trimming is correct-by-design (--models is
#      the full desired panel, always) but invisible trimming next to
#      eval.baseline's near-identical, ACCUMULATING --models is a trap.
_drop_path = os.path.join(_results_dir, "test_drop_models.jsonl")
panel_module.run(base_url, "0g/fusion-preview", ["m-a", "m-b"], _drop_path, limit=2, experiment="test-drop")
stderr_drop = io.StringIO()
with contextlib.redirect_stderr(stderr_drop):
    panel_module.run(base_url, "0g/fusion-preview", ["m-a"], _drop_path, limit=2, experiment="test-drop")
check("dropping a cached model via a smaller --models prints a clear warning naming it",
      "eval.panel_dropped_models" in stderr_drop.getvalue() and "'m-b': 2" in stderr_drop.getvalue())
os.remove(_drop_path)

# 22c. eval.fuse must refuse to resume a row that was fused from a DIFFERENT
#      panel than the --panel file given now -- the "forgot to change
#      --experiment along with --panel" trap. Must fail BEFORE --out is
#      opened for writing (leaves any prior good rows intact).
_panel_v1 = os.path.join(_results_dir, "test_fuse_configid_v1.jsonl")
_panel_v2 = os.path.join(_results_dir, "test_fuse_configid_v2.jsonl")
panel_module.run(base_url, "0g/fusion-preview", ["m-a"], _panel_v1, limit=2, experiment="test-configid-v1")
panel_module.run(base_url, "0g/fusion-preview", ["m-a", "m-b"], _panel_v2, limit=2, experiment="test-configid-v2")
_configid_out = os.path.join(_results_dir, "test_fuse_configid_out.jsonl")
fuse_module.run(base_url, "0g/fusion-preview", _panel_v1, _configid_out, experiment="test-configid-reused")
try:
    fuse_module.run(base_url, "0g/fusion-preview", _panel_v2, _configid_out, experiment="test-configid-reused")
    check("eval.fuse refuses to resume a row fused from a DIFFERENT panel than --panel gives now", False)
except ResumeMismatchError:
    check("eval.fuse refuses to resume a row fused from a DIFFERENT panel than --panel gives now", True)
check("the mismatch aborts before --out is opened for writing, leaving the old (v1) rows intact",
      all("m-a+m-b" not in json.loads(l)["config_id"] for l in open(_configid_out, encoding="utf-8")))
for _p in (_panel_v1, _panel_v2, _configid_out):
    os.remove(_p)

# 22d. eval.grade must refuse to blend two DIFFERENT fusion results for the
#      same question_id into one score -- e.g. gluing two variant fuse files
#      together (a glob like `gpqa-fuse-*.jsonl`) instead of grading one at
#      a time.
_two_fuse_a = os.path.join(_results_dir, "test_two_fuse_a.jsonl")
_two_fuse_b = os.path.join(_results_dir, "test_two_fuse_b.jsonl")
panel_module.run(base_url, "0g/fusion-preview", ["m-a"], _two_fuse_a, limit=1, experiment="test-two-fuse-a-panel")
panel_module.run(base_url, "0g/fusion-preview", ["m-a", "m-b"], _two_fuse_b, limit=1, experiment="test-two-fuse-b-panel")
_two_fuse_a_out = os.path.join(_results_dir, "test_two_fuse_a_out.jsonl")
_two_fuse_b_out = os.path.join(_results_dir, "test_two_fuse_b_out.jsonl")
fuse_module.run(base_url, "0g/fusion-preview", _two_fuse_a, _two_fuse_a_out, experiment="test-two-fuse-a")
fuse_module.run(base_url, "0g/fusion-preview", _two_fuse_b, _two_fuse_b_out, experiment="test-two-fuse-b")
try:
    grade_replay(_two_fuse_a_out, _two_fuse_b_out)
    check("eval.grade refuses to merge two different fusion results for the same question_id", False)
except GradeMergeError:
    check("eval.grade refuses to merge two different fusion results for the same question_id", True)
for _p in (_two_fuse_a, _two_fuse_b, _two_fuse_a_out, _two_fuse_b_out):
    os.remove(_p)

# 22e. eval.panel must refuse a --fusion-model that would never reach the
#      panel_only path (anything not starting with "0g/fusion") -- BEFORE
#      any calls, instead of billing a real call per question that then
#      fails with an opaque KeyError.
_bad_model_out = os.path.join(_results_dir, "test_bad_fusion_model.jsonl")
panel_module.call_api = _counting_call_api_p
try:
    _pcalls["n"] = 0
    try:
        panel_module.run(base_url, "gpt-5.6-sol", ["m-a"], _bad_model_out, limit=2, experiment="test-bad-model")
        check("eval.panel refuses a --fusion-model that doesn't start with '0g/fusion'", False)
    except ValueError as e:
        check("eval.panel refuses a --fusion-model that doesn't start with '0g/fusion'", "0g/fusion" in str(e))
    check("...and refuses it BEFORE making any calls", _pcalls["n"] == 0)
finally:
    panel_module.call_api = _orig_call_api_p
check("...and --out was never created", not os.path.exists(_bad_model_out))

# --- 23. regressions found reviewing the shared run_replay() extraction ----

# 23a. an exception escaping `process` (a real Ctrl-C is exactly this: a
#      BaseException the per-item try/except inside each tool can't catch)
#      must NOT skip carrying forward out-of-window rows -- those are the
#      ones already paid for, and losing them at the exact moment a run is
#      being interrupted is the worst time to lose them.
_carry_repro_path = os.path.join(_results_dir, "test_carry_on_exception.jsonl")
_repro_existing = {i: {"question_id": i, "val": f"old-{i}"} for i in range(6)}
_repro_expected = {i: (f"q{i}", "A") for i in range(3)}  # this run's window is qids 0-2


def _raise_on_item_1(item, prior):
    if item == 1:
        raise RuntimeError("simulated interruption mid-run")
    return {"question_id": item, "val": f"new-{item}"}, {}


try:
    run_replay([0, 1, 2], lambda x: x, _raise_on_item_1, _carry_repro_path, _repro_existing, _repro_expected)
    check("run_replay propagates an exception from `process` instead of swallowing it", False)
except RuntimeError:
    check("run_replay propagates an exception from `process` instead of swallowing it", True)
with open(_carry_repro_path, encoding="utf-8") as f:
    _carry_repro_rows = [json.loads(l) for l in f]
check("...but still carries forward every out-of-window row before re-raising, even though the "
      "loop never finished",
      sorted(r["question_id"] for r in _carry_repro_rows if r["question_id"] >= 3) == [3, 4, 5])
check("...and still wrote whatever succeeded before the exception (question 0)",
      any(r["question_id"] == 0 and r["val"] == "new-0" for r in _carry_repro_rows))
os.remove(_carry_repro_path)

server.shutdown()
gpqa_tasks_module.REAL_DEFAULT_PATH = _orig_real_default

# call_logs/ is gitignored and explicitly documented as reproducible-from-a-
# rerun -- sweep up every log file any "test-*"/"cli-*" experiment name used
# above left behind (individual sections only clean up their own, narrowly-
# scoped experiment; this catches the rest) so repeated runs don't accumulate
# litter.
for f in glob.glob(os.path.join(llm_client.LOG_DIR, "test-*")) + glob.glob(os.path.join(llm_client.LOG_DIR, "cli-*")):
    os.remove(f)
if os.path.isdir(llm_client.LOG_DIR) and not os.listdir(llm_client.LOG_DIR):
    shutil.rmtree(llm_client.LOG_DIR)

failed = [name for name, ok in PASS if not ok]
print(f"\n{len(PASS) - len(failed)}/{len(PASS)} passed")
if failed:
    raise SystemExit("FAILED: " + ", ".join(failed))
