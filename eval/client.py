"""Generic OpenAI-compatible client. Identical code path for the fusion model and any
baseline -- it has no special knowledge of what is on the other end of base_url."""
import json
import urllib.request


def call_api(base_url, model, messages, **extra):
    body = {"model": model, "messages": messages, **extra}
    req = urllib.request.Request(
        base_url.rstrip("/") + "/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        return json.loads(resp.read())
