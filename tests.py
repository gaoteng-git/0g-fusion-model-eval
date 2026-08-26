"""Self-test suite, plain asserts (no external test framework). Runs entirely
offline via the FAKE llm stand-in (no ZG_UPSTREAM_BASE_URL set). Covers the
core product-simulation pipeline (mock_fusion_api: reasoning_effort on for
panel/synthesis/baseline, off for judge with defensive <think>-stripping,
panel evidence carrying reasoning AND content, thinking-extraction for both
real-world field patterns, the cached_panel/extra_panel_models/panel_only
mechanisms) -- none of which the eval CLI actually calls into anymore, but
all of which still simulate the real product and are still tested as such --
plus the eval CLI itself: eval.sample (extract a subset of questions, tagged
with their original index so later results can be merged), eval.panel /
eval.fuse / eval.baseline (each a pure function of its explicit --input:
no resume, no --limit, no implicit state, every run overwrites --out
completely), and eval.grade (scores each given file independently, no
merging, so there is nothing for two files to disagree about).
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


def _cli(*args):
    return __import__("subprocess").run(
        [__import__("sys").executable, "-m", *args],
        cwd=os.path.dirname(os.path.abspath(__file__)), capture_output=True, text=True)


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

# --- 6. passthrough path: forwards reasoning_effort, normalizes reasoning_content
#        the same way as the fusion path, and honors an explicit `role` -----
plain = pipeline.handle_chat_completion({"model": "some-baseline", "messages": messages, "reasoning_effort": "high"})
check("baseline passthrough exposes reasoning_content when thinking is requested",
      bool(plain["choices"][0]["message"].get("reasoning_content")))

minimax_like = pipeline.handle_chat_completion({"model": "minimax-m3", "messages": messages, "reasoning_effort": "high"})
check("baseline passthrough strips inline <think> out of content for MiniMax-style models",
      "<think>" not in (minimax_like["choices"][0]["message"]["content"] or ""))
check("...and surfaces it via reasoning_content instead",
      bool(minimax_like["choices"][0]["message"].get("reasoning_content")))

# eval.panel/eval.fuse/eval.baseline are ALL just this passthrough with a
# different `role`, for clearer call-log naming -- an explicit role in the
# request must be honored, falling back to "baseline" only when absent
# (unchanged default, regression guard).
TEST_EXPERIMENT = "unit-test-logging-exp"
pipeline.handle_chat_completion({"model": "some-panel-model", "messages": messages, "role": "panel",
                                  "experiment": TEST_EXPERIMENT})
_role_log = os.path.join(llm_client.LOG_DIR, f"{TEST_EXPERIMENT}__panel__{llm_client._sanitize('some-panel-model')}.jsonl")
check("plain passthrough honors an explicit `role` in the request instead of hardcoding 'baseline'",
      os.path.exists(_role_log))
pipeline.handle_chat_completion({"model": "some-other-model", "messages": messages, "experiment": TEST_EXPERIMENT})
_default_role_log = os.path.join(llm_client.LOG_DIR,
                                   f"{TEST_EXPERIMENT}__baseline__{llm_client._sanitize('some-other-model')}.jsonl")
check("...and still defaults to role='baseline' when the request doesn't set one (unchanged default)",
      os.path.exists(_default_role_log))
for f in glob.glob(os.path.join(llm_client.LOG_DIR, f"{TEST_EXPERIMENT}__*")):
    os.remove(f)

# --- 7. gpqa_tasks: load_questions/format_question -------------------------
from eval.gpqa_tasks import load_questions, format_question, SAMPLE_PATH  # noqa: E402

questions = load_questions(SAMPLE_PATH)
check("load_questions returns one (question_id, row) pair per line", len(questions) == 5)
check("question_id defaults to position when the row has no explicit question_id field",
      [qid for qid, _ in questions] == [0, 1, 2, 3, 4])

_instruction0, _correct_letter0 = format_question(questions[0][1], questions[0][0])
check("format_question's instruction contains the final-answer format instruction",
      cfg.FINAL_LETTER_INSTRUCTION in _instruction0)
check("format_question returns one of A/B/C/D", _correct_letter0 in "ABCD")

questions_again = load_questions(SAMPLE_PATH)
check("shuffle is deterministic across repeated loads (question_id-seeded, not global RNG)",
      [format_question(row, qid)[1] for qid, row in questions]
      == [format_question(row, qid)[1] for qid, row in questions_again])

_results_dir = os.path.join(os.path.dirname(__file__), "eval", "results")
os.makedirs(_results_dir, exist_ok=True)

# an explicit question_id field (as eval.sample.py writes) must be honored,
# not overridden by position -- this is what keeps a merge across sampled
# files correct
_qid_test_path = os.path.join(_results_dir, "test_explicit_qid.jsonl")
with open(_qid_test_path, "w", encoding="utf-8") as f:
    f.write(json.dumps({**questions[2][1], "question_id": 99}) + "\n")
_tagged = load_questions(_qid_test_path)
check("an explicit question_id field in the row is honored, not overridden by file position",
      _tagged[0][0] == 99)

# format_question must seed the shuffle by question_id, not by the row's
# position in whatever file happens to be open -- put question 2's row at
# position 0 in a standalone file, tagged with its ORIGINAL id (2, not its
# new position); it must shuffle exactly the same way it did loaded from the
# full 5-row file at position 2.
_same_id_path = os.path.join(_results_dir, "test_same_id_diff_position.jsonl")
with open(_same_id_path, "w", encoding="utf-8") as f:
    f.write(json.dumps({**questions[2][1], "question_id": 2}) + "\n")
_relocated = load_questions(_same_id_path)
check("format_question seeds the shuffle by question_id, not by file position -- the SAME "
      "question (row content), tagged with the SAME id, shuffles identically whether it's at "
      "position 2 in the full file or position 0 in a standalone one-row file",
      format_question(_relocated[0][1], _relocated[0][0]) == format_question(questions[2][1], questions[2][0]))
os.remove(_same_id_path)
os.remove(_qid_test_path)

_bad_path = os.path.join(_results_dir, "test_missing_column.jsonl")
with open(_bad_path, "w", encoding="utf-8") as f:
    f.write(json.dumps({"Question": "Q?", "Correct Answer": "A"}) + "\n")  # missing 3 required columns
try:
    load_questions(_bad_path)
    check("load_questions raises a clear error when a required column is missing", False)
except ValueError as e:
    check("load_questions raises a clear error when a required column is missing",
          "Incorrect Answer 1" in str(e))
os.remove(_bad_path)

# --- 8. grade: final-letter extraction --------------------------------------
from eval.grade import extract_final_letter, grade_file  # noqa: E402

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
#         synthesis call at all. Still a real, tested feature of the product
#         simulation even though the eval CLI doesn't use it anymore -------
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

    combo_resp = pipeline.run_fusion({"messages": messages, "panel_only": True,
                                       "cached_panel": [_cached_entry], "extra_panel_models": ["panel-a"]})
    check("panel_only + cached_panel/extra_panel_models still makes no judge/synthesis call",
          "choices" not in combo_resp and len(combo_resp["0g_fusion"]["panel"]) == 2
          and combo_resp["0g_fusion"]["panel"][0] == _cached_entry)
finally:
    pipeline.run_judge, pipeline.run_synthesis = _orig_run_judge, _orig_run_synthesis

check("panel_only defaulting to falsy is unchanged -- still runs judge+synthesis",
      "choices" in pipeline.run_fusion({"messages": messages}))

# --- 14. eval.sample: extract a subset of a question file by index, tagged
#          with each row's ORIGINAL absolute index as question_id ----------
from eval.sample import parse_indices, run as sample_run  # noqa: E402

check("parse_indices: comma list", parse_indices("0,2,4") == [0, 2, 4])
check("parse_indices: a range is inclusive on both ends", parse_indices("0-4") == [0, 1, 2, 3, 4])
check("parse_indices: mixed ranges + singles, deduped and sorted",
      parse_indices("2,0-1,4,3") == [0, 1, 2, 3, 4])
try:
    parse_indices("")
    check("parse_indices rejects an empty --indices", False)
except SystemExit:
    check("parse_indices rejects an empty --indices", True)

_sample_path = os.path.join(_results_dir, "test_sample_first3.jsonl")
sample_run(SAMPLE_PATH, [0, 1, 2], _sample_path)
_sampled = load_questions(_sample_path)
check("eval.sample extracts exactly the requested rows, tagged with their original index",
      [qid for qid, _ in _sampled] == [0, 1, 2])
check("the extracted rows' actual content matches the source file's rows at those indices",
      all(_sampled[i][1]["Question"] == questions[i][1]["Question"] for i in range(3)))

_sample_path2 = os.path.join(_results_dir, "test_sample_middle_of_sample.jsonl")
sample_run(_sample_path, [1], _sample_path2)  # position 1 within the 3-row sample == original index 1
_resampled = load_questions(_sample_path2)
check("sub-sampling an already-sampled file preserves the ORIGINAL absolute question_id "
      "(not renumbered from the sample's own position) -- what makes 'sample the rest later, "
      "merge the results' actually work",
      _resampled[0][0] == 1)

_bad_sample_path = os.path.join(_results_dir, "test_sample_bad.jsonl")
try:
    sample_run(SAMPLE_PATH, [0, 99], _bad_sample_path)
    check("eval.sample rejects an out-of-range index", False)
except ValueError as e:
    check("eval.sample rejects an out-of-range index", "99" in str(e))
check("...and nothing was written for the bad request", not os.path.exists(_bad_sample_path))

sample_run(SAMPLE_PATH, [3, 4], _sample_path)  # same --out as before, different indices
check("re-running eval.sample against the same --out overwrites it completely",
      [qid for qid, _ in load_questions(_sample_path)] == [3, 4])

for _p in (_sample_path, _sample_path2):
    os.remove(_p)

_cli_sample_out = os.path.join(_results_dir, "test_sample_cli.jsonl")
_r = _cli("eval.sample", "--input", SAMPLE_PATH, "--indices", "0-1", "--out", _cli_sample_out)
check("the eval.sample CLI runs end to end", _r.returncode == 0)
check("...and writes the right rows", [qid for qid, _ in load_questions(_cli_sample_out)] == [0, 1])
os.remove(_cli_sample_out)

# --- 15. eval.panel: one model, one input file, one output file, no resume -
from eval import panel as panel_module  # noqa: E402

_panel_input = os.path.join(_results_dir, "test_panel_input.jsonl")
sample_run(SAMPLE_PATH, [0, 1, 2], _panel_input)

_panel_out = os.path.join(_results_dir, "test_panel_out.jsonl")
panel_module.run(base_url, "panel-a", _panel_input, _panel_out, experiment="test-panel-exp")
_panel_rows = [json.loads(l) for l in open(_panel_out, encoding="utf-8")]
check("eval.panel writes exactly one row per input question", len(_panel_rows) == 3)
check("every row has question_id/instruction/correct_letter/model/content/reasoning",
      all({"question_id", "instruction", "correct_letter", "model", "content", "reasoning"} <= set(r)
          for r in _panel_rows))
check("every row's model is the one requested", all(r["model"] == "panel-a" for r in _panel_rows))
check("question_ids match the input file's, in order", [r["question_id"] for r in _panel_rows] == [0, 1, 2])
check("eval.panel logs its calls under role=panel",
      os.path.exists(os.path.join(llm_client.LOG_DIR, "test-panel-exp__panel__panel-a.jsonl")))

_orig_call_api_panel = panel_module.call_api
_pcount = {"n": 0}


def _counting_panel_call(url, model, msgs, **kw):
    _pcount["n"] += 1
    return _orig_call_api_panel(url, model, msgs, **kw)


panel_module.call_api = _counting_panel_call
try:
    panel_module.run(base_url, "panel-a", _panel_input, _panel_out, experiment="test-panel-exp")
    check("eval.panel makes a fresh call for every question every run, regardless of any prior "
          "--out content -- no resume/skip logic exists at all",
          _pcount["n"] == 3)
finally:
    panel_module.call_api = _orig_call_api_panel

_call_count = {"n": 0}


def _fail_2nd_call(url, model, msgs, **kw):
    _call_count["n"] += 1
    if _call_count["n"] == 2:
        raise RuntimeError("simulated failure for the 2nd question")
    return _orig_call_api_panel(url, model, msgs, **kw)


panel_module.call_api = _fail_2nd_call
try:
    _panel_fail_out = os.path.join(_results_dir, "test_panel_fail.jsonl")
    panel_module.run(base_url, "panel-b", _panel_input, _panel_fail_out, experiment="test-panel-fail")
finally:
    panel_module.call_api = _orig_call_api_panel
_fail_rows = [json.loads(l) for l in open(_panel_fail_out, encoding="utf-8")]
check("eval.panel writes one row per question even when one question's call fails",
      len(_fail_rows) == 3)
check("the failed question is marked failed with the error captured, others succeeded",
      _fail_rows[1].get("failed") is True and "simulated failure" in _fail_rows[1].get("error", "")
      and "failed" not in _fail_rows[0] and "failed" not in _fail_rows[2])
os.remove(_panel_fail_out)


def _malformed_shape_panel(url, model, msgs, **kw):
    return {"choices": []}  # 200-ok, valid JSON, but empty choices -> IndexError on use


panel_module.call_api = _malformed_shape_panel
try:
    _panel_shape_out = os.path.join(_results_dir, "test_panel_shape.jsonl")
    panel_module.run(base_url, "panel-c", _panel_input, _panel_shape_out, experiment="test-panel-shape")
finally:
    panel_module.call_api = _orig_call_api_panel
_shape_rows = [json.loads(l) for l in open(_panel_shape_out, encoding="utf-8")]
check("eval.panel doesn't crash on a malformed/wrong-shape 200 response, marks it failed instead",
      len(_shape_rows) == 3 and all(r.get("failed") is True for r in _shape_rows))
os.remove(_panel_shape_out)

_cli_panel_out = os.path.join(_results_dir, "test_panel_cli.jsonl")
_r = _cli("eval.panel", "--model", "panel-cli", "--api-url", base_url, "--input", _panel_input,
          "--out", _cli_panel_out, "--experiment", "test-panel-cli")
check("the eval.panel CLI runs end to end", _r.returncode == 0)
check("...output has the right row count", len(open(_cli_panel_out, encoding="utf-8").readlines()) == 3)
os.remove(_cli_panel_out)
os.remove(_panel_out)

for f in glob.glob(os.path.join(llm_client.LOG_DIR, "test-panel*")):
    os.remove(f)

# --- 16. eval.baseline: same shape as eval.panel, role=baseline ------------
from eval import baseline as baseline_module  # noqa: E402

_baseline_out = os.path.join(_results_dir, "test_baseline_out.jsonl")
baseline_module.run(base_url, "gpt-5.6-sol", _panel_input, _baseline_out, experiment="test-baseline-exp")
_baseline_rows = [json.loads(l) for l in open(_baseline_out, encoding="utf-8")]
check("eval.baseline writes exactly one row per input question", len(_baseline_rows) == 3)
check("every row has the same shape as eval.panel's rows",
      all({"question_id", "instruction", "correct_letter", "model", "content", "reasoning"} <= set(r)
          for r in _baseline_rows))
check("eval.baseline logs its calls under role=baseline",
      os.path.exists(os.path.join(llm_client.LOG_DIR, "test-baseline-exp__baseline__gpt-5.6-sol.jsonl")))

_orig_call_api_baseline = baseline_module.call_api


def _fail_1st_baseline_call(url, model, msgs, **kw):
    raise RuntimeError("simulated baseline failure")


baseline_module.call_api = _fail_1st_baseline_call
try:
    _baseline_fail_out = os.path.join(_results_dir, "test_baseline_fail.jsonl")
    baseline_module.run(base_url, "claude-fable-5", _panel_input, _baseline_fail_out, experiment="test-baseline-fail")
finally:
    baseline_module.call_api = _orig_call_api_baseline
_baseline_fail_rows = [json.loads(l) for l in open(_baseline_fail_out, encoding="utf-8")]
check("eval.baseline writes one row per question even when every call fails, doesn't abort",
      len(_baseline_fail_rows) == 3 and all(r.get("failed") is True for r in _baseline_fail_rows))
os.remove(_baseline_fail_out)

_cli_baseline_out = os.path.join(_results_dir, "test_baseline_cli.jsonl")
_r = _cli("eval.baseline", "--model", "baseline-cli", "--api-url", base_url, "--input", _panel_input,
          "--out", _cli_baseline_out, "--experiment", "test-baseline-cli")
check("the eval.baseline CLI runs end to end", _r.returncode == 0)
check("...output has the right row count", len(open(_cli_baseline_out, encoding="utf-8").readlines()) == 3)
os.remove(_cli_baseline_out)

for f in glob.glob(os.path.join(llm_client.LOG_DIR, "test-baseline*")):
    os.remove(f)
os.remove(_baseline_out)

# --- 17. eval.fuse: judge+synthesis over N already-built panel files -------
from eval import fuse as fuse_module  # noqa: E402

_panel_a_path = os.path.join(_results_dir, "fp_a.jsonl")
_panel_b_path = os.path.join(_results_dir, "fp_b.jsonl")
panel_module.run(base_url, "fuse-panel-a", _panel_input, _panel_a_path, experiment="test-fuse-panels")
panel_module.run(base_url, "fuse-panel-b", _panel_input, _panel_b_path, experiment="test-fuse-panels")
check("building the 2 panel files made zero judge/synthesis calls",
      glob.glob(os.path.join(llm_client.LOG_DIR, "test-fuse-panels__judge__*")) == []
      and glob.glob(os.path.join(llm_client.LOG_DIR, "test-fuse-panels__synthesis__*")) == [])
for f in glob.glob(os.path.join(llm_client.LOG_DIR, "test-fuse-panels__*")):
    os.remove(f)

_fuse_out = os.path.join(_results_dir, "test_fuse_out.jsonl")
fuse_module.run(base_url, "judge-x", "synthesis-y", _panel_input, [_panel_a_path, _panel_b_path], _fuse_out,
                experiment="test-fuse-exp")
_fuse_rows = [json.loads(l) for l in open(_fuse_out, encoding="utf-8")]
check("eval.fuse writes one row per input question", len(_fuse_rows) == 3)
check("every row has the fusion answer + judge_json + panel_models list",
      all({"question_id", "instruction", "correct_letter", "judge_model", "synthesis_model",
           "panel_models", "content", "reasoning", "judge_json"} <= set(r) for r in _fuse_rows))
check("panel_models records exactly the models that were fused, in order",
      all(r["panel_models"] == ["fuse-panel-a", "fuse-panel-b"] for r in _fuse_rows))
check("*** eval.fuse makes zero fresh panel calls (only reads the already-built panel files) ***",
      glob.glob(os.path.join(llm_client.LOG_DIR, "test-fuse-exp__panel__*")) == [])
check("eval.fuse makes exactly the judge call",
      len(glob.glob(os.path.join(llm_client.LOG_DIR, "test-fuse-exp__judge__*"))) == 1)
check("eval.fuse makes exactly the synthesis call",
      len(glob.glob(os.path.join(llm_client.LOG_DIR, "test-fuse-exp__synthesis__*"))) == 1)
for f in glob.glob(os.path.join(llm_client.LOG_DIR, "test-fuse-exp__*")):
    os.remove(f)

# a question missing (or failed) in one panel file must be skipped -- marked
# failed, naming which file was short -- not abort the run, and must cost
# zero judge/synthesis calls for exactly that question
_panel_b_rows = [json.loads(l) for l in open(_panel_b_path, encoding="utf-8")]
_panel_b_missing_path = os.path.join(_results_dir, "fp_b_missing_q1.jsonl")
with open(_panel_b_missing_path, "w", encoding="utf-8") as f:
    for r in _panel_b_rows:
        if r["question_id"] != 1:
            f.write(json.dumps(r) + "\n")

_orig_call_api_fuse = fuse_module.call_api
_fcount = {"n": 0}


def _counting_fuse_call(url, model, msgs, **kw):
    _fcount["n"] += 1
    return _orig_call_api_fuse(url, model, msgs, **kw)


fuse_module.call_api = _counting_fuse_call
try:
    _fuse_missing_out = os.path.join(_results_dir, "test_fuse_missing.jsonl")
    stderr_missing = io.StringIO()
    with contextlib.redirect_stderr(stderr_missing):
        fuse_module.run(base_url, "judge-x", "synthesis-y", _panel_input, [_panel_a_path, _panel_b_missing_path],
                         _fuse_missing_out, experiment="test-fuse-missing")
    check("only 2 questions' worth of judge+synthesis calls were made (2 questions x 2 calls = 4), "
          "not 3 x 2 = 6 -- the missing question cost nothing", _fcount["n"] == 4)
finally:
    fuse_module.call_api = _orig_call_api_fuse
_missing_rows = [json.loads(l) for l in open(_fuse_missing_out, encoding="utf-8")]
check("eval.fuse writes one row per question even when a panel file is missing one",
      len(_missing_rows) == 3)
check("the question missing from a panel file is marked failed, naming which file was short",
      _missing_rows[1].get("failed") is True and "fp_b_missing_q1.jsonl" in _missing_rows[1].get("error", ""))
check("the other 2 questions fused normally", "failed" not in _missing_rows[0] and "failed" not in _missing_rows[2])
check("eval.fuse prints a clear skip message naming the question_id",
      "eval.fuse_question_skipped" in stderr_missing.getvalue())
os.remove(_panel_b_missing_path)
os.remove(_fuse_missing_out)
for f in glob.glob(os.path.join(llm_client.LOG_DIR, "test-fuse-missing__*")):
    os.remove(f)

# JUDGE_MODELS_WITHOUT_JSON_MODE quirk must still be honored (0g-router hard-
# rejects response_format for a judge model that doesn't advertise it)
_seen_json_mode_fuse = {}


def _capture_json_mode_fuse(url, model, msgs, **kw):
    if "json_mode" in kw:
        _seen_json_mode_fuse["value"] = kw["json_mode"]
    return _orig_call_api_fuse(url, model, msgs, **kw)


fuse_module.call_api = _capture_json_mode_fuse
try:
    fuse_module.run(base_url, "minimax-m3", "synthesis-y", _panel_input, [_panel_a_path, _panel_b_path],
                     _fuse_out, experiment="test-fuse-jsonmode")
    check("eval.fuse withholds json_mode for a judge model in JUDGE_MODELS_WITHOUT_JSON_MODE",
          _seen_json_mode_fuse.get("value") is False)
finally:
    fuse_module.call_api = _orig_call_api_fuse
os.remove(_fuse_out)
for f in glob.glob(os.path.join(llm_client.LOG_DIR, "test-fuse-jsonmode__*")):
    os.remove(f)

# judge succeeding but synthesis failing must still mark the question failed
# cleanly, not crash the run (the call+row-construction try-block-scope lesson)
def _synthesis_fails(url, model, msgs, **kw):
    if model == "synthesis-y":
        raise RuntimeError("simulated synthesis failure")
    return _orig_call_api_fuse(url, model, msgs, **kw)


fuse_module.call_api = _synthesis_fails
try:
    _fuse_synthfail_out = os.path.join(_results_dir, "test_fuse_synthfail.jsonl")
    fuse_module.run(base_url, "judge-x", "synthesis-y", _panel_input, [_panel_a_path, _panel_b_path],
                     _fuse_synthfail_out, experiment="test-fuse-synthfail")
finally:
    fuse_module.call_api = _orig_call_api_fuse
_synthfail_rows = [json.loads(l) for l in open(_fuse_synthfail_out, encoding="utf-8")]
check("eval.fuse marks a question failed if synthesis fails (even though judge succeeded first), "
      "doesn't crash the run",
      len(_synthfail_rows) == 3
      and all(r.get("failed") is True and "simulated synthesis failure" in r.get("error", "")
              for r in _synthfail_rows))
os.remove(_fuse_synthfail_out)
for f in glob.glob(os.path.join(llm_client.LOG_DIR, "test-fuse-synthfail__*")):
    os.remove(f)

# a malformed judge JSON must warn but not abort the run (minimax-m3 as judge
# with json_mode withheld -- FAKE mode's non-JSON text for it)
stderr_judge_invalid = io.StringIO()
with contextlib.redirect_stderr(stderr_judge_invalid):
    fuse_module.run(base_url, "minimax-m3", "kimi-k3", _panel_input, [_panel_a_path, _panel_b_path],
                     _fuse_out, experiment="test-fuse-judgeinvalid")
check("a malformed judge JSON prints a clear warning naming the question and judge model, "
      "and does not abort the run",
      "eval.fuse_judge_json_invalid" in stderr_judge_invalid.getvalue()
      and "judge_model='minimax-m3'" in stderr_judge_invalid.getvalue())
_judgeinvalid_rows = [json.loads(l) for l in open(_fuse_out, encoding="utf-8")]
check("...and every question still fused successfully despite the warning",
      len(_judgeinvalid_rows) == 3 and all("failed" not in r for r in _judgeinvalid_rows))
os.remove(_fuse_out)
for f in glob.glob(os.path.join(llm_client.LOG_DIR, "test-fuse-judgeinvalid__*")):
    os.remove(f)

_cli_fuse_out = os.path.join(_results_dir, "test_fuse_cli.jsonl")
_r = _cli("eval.fuse", "--judge-model", "judge-x", "--synthesis-model", "synthesis-y", "--api-url", base_url,
          "--input", _panel_input, "--panels", f"{_panel_a_path},{_panel_b_path}", "--out", _cli_fuse_out,
          "--experiment", "test-fuse-cli")
check("the eval.fuse CLI runs end to end", _r.returncode == 0)
check("...output has the right row count", len(open(_cli_fuse_out, encoding="utf-8").readlines()) == 3)
os.remove(_cli_fuse_out)
for f in glob.glob(os.path.join(llm_client.LOG_DIR, "test-fuse-cli__*")):
    os.remove(f)

os.remove(_panel_a_path)
os.remove(_panel_b_path)

# --- 18. eval.grade: scores each given file independently, never merges ----
_grade_path = os.path.join(_results_dir, "test_grade_shapes.jsonl")
with open(_grade_path, "w", encoding="utf-8") as f:
    for r in (
        {"question_id": 0, "correct_letter": "A", "content": "Final Answer: A"},     # correct
        {"question_id": 1, "correct_letter": "B", "content": "Final Answer: C"},     # wrong
        {"question_id": 2, "correct_letter": "A", "content": "no letter here"},      # extraction_failed
        {"question_id": 3, "correct_letter": "A", "failed": True, "error": "boom"},  # call_failed
        {"question_id": 4, "content": "Final Answer: A"},                            # no_ground_truth
    ):
        f.write(json.dumps(r) + "\n")
_grade_result = grade_file(_grade_path)
check("grade_file: hand-computed scoring across every row shape",
      _grade_result == {"accuracy": 0.2, "correct": 1, "extraction_failed": 1, "call_failed": 1,
                         "no_ground_truth": 1, "duplicate_question_ids": 0, "n": 5})
os.remove(_grade_path)

_grade_path_a = os.path.join(_results_dir, "test_grade_a.jsonl")
_grade_path_b = os.path.join(_results_dir, "test_grade_b.jsonl")
with open(_grade_path_a, "w", encoding="utf-8") as f:
    f.write(json.dumps({"question_id": 0, "correct_letter": "A", "content": "Final Answer: A"}) + "\n")
with open(_grade_path_b, "w", encoding="utf-8") as f:
    f.write(json.dumps({"question_id": 0, "correct_letter": "Z", "content": "Final Answer: Q"}) + "\n")
_result_a = grade_file(_grade_path_a)
_result_b = grade_file(_grade_path_b)
check("grading two files that disagree about question_id 0 causes no error and no interaction -- "
      "each file's own score is exactly its own, because nothing ever merges them",
      _result_a["accuracy"] == 1.0 and _result_b["accuracy"] == 0.0)

_r = _cli("eval.grade", _grade_path_a, _grade_path_b)
check("the eval.grade CLI accepts multiple files and runs end to end", _r.returncode == 0)
_cli_grade_result = json.loads(_r.stdout)
check("the CLI reports each file's score separately, keyed by its own path",
      set(_cli_grade_result) == {_grade_path_a, _grade_path_b}
      and _cli_grade_result[_grade_path_a]["accuracy"] == 1.0
      and _cli_grade_result[_grade_path_b]["accuracy"] == 0.0)
os.remove(_grade_path_a)
os.remove(_grade_path_b)

_r = _cli("eval.grade")
check("the eval.grade CLI rejects being run with no files at all", _r.returncode != 0)

os.remove(_panel_input)

# --- 19. full workflow, end to end: sample -> panel (x2 models) -> fuse ->
#          baseline -> grade, then extend with "sample the rest, merge" -----
_e2e_sample1 = os.path.join(_results_dir, "test_e2e_sample_0-2.jsonl")
sample_run(SAMPLE_PATH, [0, 1, 2], _e2e_sample1)

_e2e_panel1 = os.path.join(_results_dir, "test_e2e_panel_m1.jsonl")
_e2e_panel2 = os.path.join(_results_dir, "test_e2e_panel_m2.jsonl")
panel_module.run(base_url, "e2e-m1", _e2e_sample1, _e2e_panel1, experiment="test-e2e")
panel_module.run(base_url, "e2e-m2", _e2e_sample1, _e2e_panel2, experiment="test-e2e")
check("end-to-end: building 2 panel files made zero judge/synthesis calls",
      glob.glob(os.path.join(llm_client.LOG_DIR, "test-e2e__judge__*")) == []
      and glob.glob(os.path.join(llm_client.LOG_DIR, "test-e2e__synthesis__*")) == [])

_e2e_fuse = os.path.join(_results_dir, "test_e2e_fuse.jsonl")
fuse_module.run(base_url, "e2e-judge", "e2e-synth", _e2e_sample1, [_e2e_panel1, _e2e_panel2], _e2e_fuse,
                 experiment="test-e2e")
check("end-to-end: fuse made exactly 1 judge-log and 1 synthesis-log file, 3 lines each "
      "(one per question, no more)",
      len(open(os.path.join(llm_client.LOG_DIR, "test-e2e__judge__e2e-judge.jsonl"), encoding="utf-8")
          .readlines()) == 3
      and len(open(os.path.join(llm_client.LOG_DIR, "test-e2e__synthesis__e2e-synth.jsonl"), encoding="utf-8")
              .readlines()) == 3)

_e2e_baseline = os.path.join(_results_dir, "test_e2e_baseline.jsonl")
baseline_module.run(base_url, "e2e-baseline", _e2e_sample1, _e2e_baseline, experiment="test-e2e")

check("end-to-end: fuse result over the 3-question sample grades cleanly", grade_file(_e2e_fuse)["n"] == 3)

for f in glob.glob(os.path.join(llm_client.LOG_DIR, "test-e2e__*")):
    os.remove(f)

# sample the REMAINING 2 questions, run the same model over them, then merge
# the two output files by plain concatenation -- exactly the workflow this
# whole redesign exists to support
_e2e_sample2 = os.path.join(_results_dir, "test_e2e_sample_3-4.jsonl")
sample_run(SAMPLE_PATH, [3, 4], _e2e_sample2)
_e2e_panel1_rest = os.path.join(_results_dir, "test_e2e_panel_m1_rest.jsonl")
panel_module.run(base_url, "e2e-m1", _e2e_sample2, _e2e_panel1_rest, experiment="test-e2e-rest")

_e2e_panel1_merged = os.path.join(_results_dir, "test_e2e_panel_m1_merged.jsonl")
with open(_e2e_panel1_merged, "w", encoding="utf-8") as out_f:
    for path in (_e2e_panel1, _e2e_panel1_rest):
        with open(path, encoding="utf-8") as in_f:
            out_f.write(in_f.read())
_merged_rows = [json.loads(l) for l in open(_e2e_panel1_merged, encoding="utf-8")]
check("merging a first-batch panel file with a later 'rest of the questions' panel file (plain "
      "concatenation) produces exactly the 5 original questions, each with the right question_id, "
      "no collisions",
      sorted(r["question_id"] for r in _merged_rows) == [0, 1, 2, 3, 4])

for f in glob.glob(os.path.join(llm_client.LOG_DIR, "test-e2e-rest__*")):
    os.remove(f)
for p in (_e2e_sample1, _e2e_sample2, _e2e_panel1, _e2e_panel2, _e2e_fuse, _e2e_baseline,
          _e2e_panel1_rest, _e2e_panel1_merged):
    os.remove(p)

# --- 20. regressions found in the 4th independent review round -----------

# 20a. --out's directory must be created if it doesn't exist yet --
#      eval/samples/ and eval/results/ are both gitignored, so neither
#      exists on a fresh clone; a bare open() used to fail before writing
#      anything, on the very first documented command.
_json_input = os.path.join(_results_dir, "test_json_input.jsonl")
sample_run(SAMPLE_PATH, [0], _json_input)

_nodir_sample = os.path.join(_results_dir, "nodir_a", "out.jsonl")
sample_run(SAMPLE_PATH, [0], _nodir_sample)
check("eval.sample creates --out's directory if it doesn't exist yet", os.path.exists(_nodir_sample))
shutil.rmtree(os.path.dirname(_nodir_sample))

_nodir_panel = os.path.join(_results_dir, "nodir_b", "out.jsonl")
panel_module.run(base_url, "m", _json_input, _nodir_panel)
check("eval.panel creates --out's directory if it doesn't exist yet", os.path.exists(_nodir_panel))
shutil.rmtree(os.path.dirname(_nodir_panel))

_nodir_baseline = os.path.join(_results_dir, "nodir_c", "out.jsonl")
baseline_module.run(base_url, "m", _json_input, _nodir_baseline)
check("eval.baseline creates --out's directory if it doesn't exist yet", os.path.exists(_nodir_baseline))
shutil.rmtree(os.path.dirname(_nodir_baseline))

_json_panel_a = os.path.join(_results_dir, "test_json_panel_a.jsonl")
_json_panel_b = os.path.join(_results_dir, "test_json_panel_b.jsonl")
panel_module.run(base_url, "json-panel-a", _json_input, _json_panel_a, experiment="test-json-mode-panels")
panel_module.run(base_url, "json-panel-b", _json_input, _json_panel_b, experiment="test-json-mode-panels")

_nodir_fuse = os.path.join(_results_dir, "nodir_d", "out.jsonl")
fuse_module.run(base_url, "judge-x", "synth-x", _json_input, [_json_panel_a, _json_panel_b], _nodir_fuse)
check("eval.fuse creates --out's directory if it doesn't exist yet", os.path.exists(_nodir_fuse))
shutil.rmtree(os.path.dirname(_nodir_fuse))
for f in glob.glob(os.path.join(llm_client.LOG_DIR, "test-json-mode-panels__*")):
    os.remove(f)

# 20b. json_mode must actually reach the real upstream call through the
#      plain passthrough -- it used to be silently dropped by
#      handle_chat_completion (which forwarded reasoning_effort/role but not
#      json_mode), so the judge NEVER got JSON mode regardless of model,
#      making JUDGE_MODELS_WITHOUT_JSON_MODE's whole purpose moot on every
#      paid run. Checking the call_api() call-site kwarg alone (as an
#      earlier test did) can't catch this -- the drop happens one hop
#      further downstream, inside the server.
_json_fuse_out = os.path.join(_results_dir, "test_json_fuse_out.jsonl")
stderr_json = io.StringIO()
with contextlib.redirect_stderr(stderr_json):
    fuse_module.run(base_url, "judge-supports-json-mode", "synth-x", _json_input, [_json_panel_a, _json_panel_b],
                     _json_fuse_out, experiment="test-json-mode")
check("json_mode reaches the real upstream call -- a judge model NOT in "
      "JUDGE_MODELS_WITHOUT_JSON_MODE gets a clean JSON response with no 'invalid' warning",
      "eval.fuse_judge_json_invalid" not in stderr_json.getvalue())
_json_row = json.loads(open(_json_fuse_out, encoding="utf-8").readline())
check("...and judge_json is actually FAKE mode's json_mode response shape, proving json_mode "
      "reached llm_client._fake_llm itself, not just the call_api() call site",
      json.loads(_json_row["judge_json"]).get("consensus") == "panel members broadly agree")
os.remove(_json_fuse_out)
for f in glob.glob(os.path.join(llm_client.LOG_DIR, "test-json-mode__*")):
    os.remove(f)

# 20c. --panels listing the same file twice must be refused before any
#      calls -- it would double-weight that panel member's vote in
#      judge/synthesis and bill for it once per occurrence.
_dup_panels_out = os.path.join(_results_dir, "test_dup_panels_out.jsonl")
try:
    fuse_module.run(base_url, "judge-x", "synth-x", _json_input, [_json_panel_a, _json_panel_a], _dup_panels_out)
    check("eval.fuse refuses --panels listing the same file more than once", False)
except ValueError as e:
    check("eval.fuse refuses --panels listing the same file more than once", _json_panel_a in str(e))
check("...and --out was never created", not os.path.exists(_dup_panels_out))

# 20d. a panel row that disagrees with --input about what the question even
#      IS (e.g. built from a different/reordered source file) must be
#      treated as unusable for that question, not silently fused as if it
#      matched -- every panel row already carries its own `instruction`,
#      which is exactly the data needed to catch this for free.
_json_panel_wrong = os.path.join(_results_dir, "test_json_panel_wrong.jsonl")
with open(_json_panel_wrong, "w", encoding="utf-8") as wf:
    wf.write(json.dumps({"question_id": 0, "instruction": "a completely different question text",
                          "correct_letter": "A", "model": "wrong-model", "content": "x", "reasoning": "y"}) + "\n")
_mismatch_out = os.path.join(_results_dir, "test_mismatch_out.jsonl")
stderr_mismatch = io.StringIO()
with contextlib.redirect_stderr(stderr_mismatch):
    fuse_module.run(base_url, "judge-x", "synth-x", _json_input, [_json_panel_a, _json_panel_wrong], _mismatch_out,
                     experiment="test-mismatch")
_mismatch_rows = [json.loads(l) for l in open(_mismatch_out, encoding="utf-8")]
check("a panel file whose row disagrees with --input about the question text is treated as "
      "unusable (marked failed), not silently fused as if it matched",
      _mismatch_rows[0].get("failed") is True and "different question" in _mismatch_rows[0].get("error", ""))
check("no judge/synthesis call was made for the mismatched question",
      glob.glob(os.path.join(llm_client.LOG_DIR, "test-mismatch__judge__*")) == []
      and glob.glob(os.path.join(llm_client.LOG_DIR, "test-mismatch__synthesis__*")) == [])
os.remove(_json_panel_wrong)
os.remove(_mismatch_out)

for f in (_json_input, _json_panel_a, _json_panel_b):
    os.remove(f)

# 20e. a reversed range in --indices ("end" before "start") must be
#      rejected, not silently contribute zero indices with no warning --
#      and a non-numeric index must give a clean error, not a raw
#      traceback.
try:
    parse_indices("4-3")
    check("parse_indices rejects a reversed range instead of silently contributing nothing", False)
except SystemExit:
    check("parse_indices rejects a reversed range instead of silently contributing nothing", True)
try:
    parse_indices("not-a-number")
    check("parse_indices gives a clean error (not a raw traceback) on a non-numeric index", False)
except SystemExit:
    check("parse_indices gives a clean error (not a raw traceback) on a non-numeric index", True)

# 20f. duplicate question_ids within a graded file (e.g. from concatenating
#      two OVERLAPPING, not disjoint, eval.sample.py batches) must be
#      deduplicated -- n must reflect distinct questions, not double-count
#      whichever ids appear twice -- and reported, not silently absorbed.
_dup_grade_path = os.path.join(_results_dir, "test_dup_grade.jsonl")
with open(_dup_grade_path, "w", encoding="utf-8") as f:
    f.write(json.dumps({"question_id": 0, "correct_letter": "A", "content": "Final Answer: A"}) + "\n")
    f.write(json.dumps({"question_id": 1, "correct_letter": "A", "content": "Final Answer: A"}) + "\n")
    f.write(json.dumps({"question_id": 0, "correct_letter": "A", "content": "Final Answer: A"}) + "\n")  # dup of 0
stderr_dup = io.StringIO()
with contextlib.redirect_stderr(stderr_dup):
    _dup_result = grade_file(_dup_grade_path)
check("grade_file dedupes by question_id -- n reflects DISTINCT questions, not the raw line count",
      _dup_result["n"] == 2 and _dup_result["duplicate_question_ids"] == 1)
check("...and reports the duplicate on stderr instead of silently absorbing it",
      "eval.grade_duplicate_question_ids" in stderr_dup.getvalue() and "[0]" in stderr_dup.getvalue())
os.remove(_dup_grade_path)

# 20g. a corrupt/unreadable file must not lose the score for every OTHER
#      file passed on the same eval.grade command line.
_good_grade_path = os.path.join(_results_dir, "test_good_grade.jsonl")
with open(_good_grade_path, "w", encoding="utf-8") as f:
    f.write(json.dumps({"question_id": 0, "correct_letter": "A", "content": "Final Answer: A"}) + "\n")
_corrupt_grade_path = os.path.join(_results_dir, "test_corrupt_grade.jsonl")
with open(_corrupt_grade_path, "w", encoding="utf-8") as f:
    f.write('{"question_id": 0, "correct_letter": "A", "content": "truncated...')  # invalid JSON, unterminated
_r = _cli("eval.grade", _good_grade_path, _corrupt_grade_path)
check("the eval.grade CLI runs end to end (exit 0) even when one file is corrupt", _r.returncode == 0)
_cli_result = json.loads(_r.stdout)
check("the good file's score survives; the corrupt file reports its own error instead of crashing everything",
      _cli_result[_good_grade_path]["accuracy"] == 1.0 and "error" in _cli_result[_corrupt_grade_path])
os.remove(_good_grade_path)
os.remove(_corrupt_grade_path)

# 20h. a `question_id` field read from a CSV --input arrives as a string
#      (csv.DictReader stringifies every field) -- must be coerced to int,
#      or it becomes a different dict key / random.Random() seed than the
#      same id read from a JSONL file, silently breaking question_id
#      matching and shuffle determinism across the two formats.
_csv_qid_path = os.path.join(_results_dir, "test_csv_qid.csv")
with open(_csv_qid_path, "w", encoding="utf-8", newline="") as f:
    f.write("Question,Correct Answer,Incorrect Answer 1,Incorrect Answer 2,Incorrect Answer 3,question_id\n")
    f.write("Q?,right,wrong1,wrong2,wrong3,7\n")
_csv_loaded = load_questions(_csv_qid_path)
check("a question_id read from CSV is coerced to int, not left as the string csv.DictReader produces",
      _csv_loaded[0][0] == 7 and isinstance(_csv_loaded[0][0], int))
os.remove(_csv_qid_path)

server.shutdown()

# call_logs/ is gitignored and reproducible-from-a-rerun -- sweep up anything
# any "test-*" experiment name above left behind.
for f in glob.glob(os.path.join(llm_client.LOG_DIR, "test-*")):
    os.remove(f)
if os.path.isdir(llm_client.LOG_DIR) and not os.listdir(llm_client.LOG_DIR):
    shutil.rmtree(llm_client.LOG_DIR)

failed = [name for name, ok in PASS if not ok]
print(f"\n{len(PASS) - len(failed)}/{len(PASS)} passed")
if failed:
    raise SystemExit("FAILED: " + ", ".join(failed))
