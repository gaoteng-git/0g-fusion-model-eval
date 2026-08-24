#!/usr/bin/env bash
# Probe 0g-router's real /v1/chat/completions responses for each of the 7
# candidate panel models, to see where thinking/reasoning content actually
# shows up (a separate `reasoning_content` field, inline in `content` with
# <think> tags, or nowhere at all) -- code-reading (0g-router/0g-serving-broker
# reasoning.go + docs/design/reasoning-translation.md) tells us what SHOULD
# happen; this script is the only way to confirm what actually does.
#
# Usage:
#   export ZG_ROUTER_BASE="https://<0g-router-endpoint>"
#   export ZG_ROUTER_KEY="<your API key>"
#   ./test_thinking_params.sh
#
# Every response is saved under results_thinking_probe/ as raw text (whatever
# the server actually returned, valid JSON or not), in addition to being
# printed. A single model's bad/unexpected response (auth error, wrong model
# id, SSE stream, empty body, ...) is reported and skipped -- it never aborts
# the rest of the script (deliberately NOT using `set -e`/`pipefail` around
# the network calls, since a non-2xx or non-JSON response is an expected,
# recoverable outcome here, not a script bug).

set -uo pipefail

BASE="${ZG_ROUTER_BASE:?set ZG_ROUTER_BASE to the 0g-router base URL}"
KEY="${ZG_ROUTER_KEY:?set ZG_ROUTER_KEY to your API key}"
# Strip any trailing slash(es) -- "https://host/" + "/v1/chat/completions"
# would otherwise build "https://host//v1/chat/completions" (double slash),
# which most routers (gin included, without an explicit redirect rule) will
# NOT match to the registered "/v1/chat/completions" route -- it falls
# through to the default 404 handler instead. This was the actual cause of
# an earlier "404 page not found" run: the route itself is confirmed correct
# by reading 0g-router's own router.go (v1 := r.Group("/v1"); inferenceGroup
# .POST("/chat/completions", ...) -- no hidden extra prefix), so a 404 here
# means BASE itself is off, not the path.
BASE="${BASE%/}"
QUESTION="9.11和9.9哪个更大？请说明理由。"
OUT_DIR="results_thinking_probe"
mkdir -p "$OUT_DIR"

# Sanity check FIRST: hit the simple, public, unauthenticated GET /v1/models
# endpoint before spending 19 chat-completions calls on a BASE that might be
# wrong entirely. A non-200 here means fix ZG_ROUTER_BASE before going further
# -- everything below will 404/fail the same way for the same reason.
echo "=== sanity check: GET \$BASE/v1/models ==="
sanity_status=$(curl -s -o /tmp/0g_sanity_check.json -w '%{http_code}' "$BASE/v1/models") || true
echo "HTTP status: $sanity_status"
if [[ "$sanity_status" != "200" ]]; then
    echo "(not 200 -- raw response below; fix ZG_ROUTER_BASE before continuing)"
    cat /tmp/0g_sanity_check.json 2>/dev/null
    echo ""
    echo "ZG_ROUTER_BASE is currently: $BASE"
    echo "Double-check this against the base URL shown on pc.0g.ai's API key page."
else
    echo "OK -- BASE looks reachable, proceeding."
fi
rm -f /tmp/0g_sanity_check.json
echo ""

# call LABEL JSON_BODY
# Sends the request, saves the raw response body to $OUT_DIR/$LABEL.json,
# prints the HTTP status code, and pretty-prints the body if (and only if)
# it parses as JSON -- otherwise prints the raw text as-is so you can see
# exactly what came back (error page, SSE stream, empty body, etc.).
call() {
    local label="$1" body="$2"
    local resp_file="$OUT_DIR/${label}.json"
    echo "=== $label ==="
    local status
    status=$(curl -s -o "$resp_file" -w '%{http_code}' \
        "$BASE/v1/chat/completions" \
        -H "Authorization: Bearer $KEY" \
        -H "Content-Type: application/json" \
        -d "$body") || {
        echo "curl itself failed (network/DNS/connection error) -- see above"
        echo ""
        return 0
    }
    echo "HTTP status: $status"
    if python3 -m json.tool "$resp_file" 2>/dev/null; then
        :
    else
        echo "(response is not valid JSON -- raw content below)"
        cat "$resp_file"
        echo ""
    fi
    echo ""
}

# --- Part 1: portable reasoning_effort, uniform across all 7 candidate
#     panel models. Per reasoning.go / reasoning-translation.md, the broker
#     translates this into whatever native control each model advertises
#     (thinking / chat_template_kwargs.enable_thinking / enable_thinking),
#     or is a no-op for models with no advertised native control
#     (kimi-k3, hy3 today). --------------------------------------------------

MODELS=(minimax-m3 kimi-k3 glm-5.2 deepseek-v4-pro qwen3.8-max hy3 0gm-1.0-35b-a3b)

for MODEL in "${MODELS[@]}"; do
    body=$(python3 -c "
import json, sys
print(json.dumps({
    'model': sys.argv[1],
    'messages': [{'role': 'user', 'content': sys.argv[2]}],
    'reasoning_effort': 'high',
    'stream': False,
}))
" "$MODEL" "$QUESTION")
    call "reasoning_effort-${MODEL}" "$body"
done

# --- Part 2: direct native parameters, bypassing the reasoning_effort
#     translation, to confirm both paths land on the same behavior. Skipped
#     for kimi-k3/hy3 -- they advertise no native control to test directly. --

call "native-thinking-minimax-m3" \
    "$(python3 -c "import json; print(json.dumps({'model':'minimax-m3','messages':[{'role':'user','content':'$QUESTION'}],'thinking':{'type':'enabled'},'stream':False}))")"

call "native-chat_template_kwargs-glm-5.2" \
    "$(python3 -c "import json; print(json.dumps({'model':'glm-5.2','messages':[{'role':'user','content':'$QUESTION'}],'chat_template_kwargs':{'enable_thinking':True},'stream':False}))")"

call "native-chat_template_kwargs-0gm-1.0-35b-a3b" \
    "$(python3 -c "import json; print(json.dumps({'model':'0gm-1.0-35b-a3b','messages':[{'role':'user','content':'$QUESTION'}],'chat_template_kwargs':{'enable_thinking':True},'stream':False}))")"

call "native-enable_thinking-deepseek-v4-pro" \
    "$(python3 -c "import json; print(json.dumps({'model':'deepseek-v4-pro','messages':[{'role':'user','content':'$QUESTION'}],'enable_thinking':True,'stream':False}))")"

call "native-enable_thinking-qwen3.8-max" \
    "$(python3 -c "import json; print(json.dumps({'model':'qwen3.8-max','messages':[{'role':'user','content':'$QUESTION'}],'enable_thinking':True,'stream':False}))")"

# --- Part 3: baseline call with NO reasoning parameter at all, per model --
#     so you can compare "asked for thinking" vs "asked for nothing" and see
#     each model's true default (some, like kimi-k3, claim thinking is
#     always on regardless). ------------------------------------------------

for MODEL in "${MODELS[@]}"; do
    body=$(python3 -c "
import json, sys
print(json.dumps({'model': sys.argv[1], 'messages': [{'role': 'user', 'content': sys.argv[2]}], 'stream': False}))
" "$MODEL" "$QUESTION")
    call "no-reasoning-param-${MODEL}" "$body"
done

echo "Done. All raw responses saved under $OUT_DIR/ -- check each for a"
echo "separate reasoning_content field, or inline <think>...</think> markup"
echo "in content. Any HTTP status != 200 or non-JSON body printed above"
echo "points at an auth/model-id/endpoint problem to fix before trusting"
echo "the thinking-related findings."
