"""Generic OpenAI-compatible client. Identical code path for the fusion model and any
baseline -- it has no special knowledge of what is on the other end of base_url."""
import json
import urllib.error
import urllib.request


def call_api(base_url, model, messages, **extra):
    body = {"model": model, "messages": messages, **extra}
    req = urllib.request.Request(
        base_url.rstrip("/") + "/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as e:
        # Bare urllib.error.HTTPError swallows the response body -- which for
        # mock_fusion_api.server is exactly where the actually-useful error
        # lives ({"error": str(exc)}, e.g. an unsupported model ID, an
        # upstream 4xx/5xx, a missing panel model). Surface it instead of
        # forcing whoever's running this to go re-derive it by hand.
        detail = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"{base_url} returned HTTP {e.code} for model={model!r}: {detail}") from e
