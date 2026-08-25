#!/usr/bin/env python3
"""One-off diagnostic: send a single minimal request to a running
mock_fusion_api.server and print exactly what comes back -- including the
real error body on a non-200 response (eval.client.call_api surfaces this as
a RuntimeError since the client.py fix; this script exists so you don't have
to hand-write the same urllib try/except every time something 500s).

Uses a generic placeholder question, not a real GPQA question -- this is for
checking whether the pipeline call itself works (right model IDs, upstream
reachable, correct auth), not for testing GPQA behavior specifically.

Run (server must already be running, e.g. `python3 -m mock_fusion_api.server 8000`):
  python3 debug_fusion_call.py
  python3 debug_fusion_call.py --base-url http://localhost:8000 --model 0g/fusion-preview
  python3 debug_fusion_call.py --model gpt-5.6-sol --question "What is 2+2?"
"""
import argparse
import json

from eval.client import call_api


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base-url", default="http://localhost:8000")
    p.add_argument("--model", default="0g/fusion-preview")
    p.add_argument("--question", default="What is 9.11 plus 9.9? Answer with just the number.")
    p.add_argument("--experiment", default="debug-fusion-call")
    args = p.parse_args()

    messages = [{"role": "user", "content": args.question}]
    print(f"POST {args.base_url}/v1/chat/completions  model={args.model!r}")
    try:
        resp = call_api(args.base_url, args.model, messages, allow_tool_call_output=False, experiment=args.experiment)
    except RuntimeError as e:
        print("\nFAILED -- real server-side error:\n")
        print(e)
        return
    except Exception as e:
        print(f"\nFAILED -- could not even reach the server ({type(e).__name__}: {e})")
        print("Check the server is actually running and --base-url has no typo/trailing slash.")
        return

    print("\nOK -- full response:\n")
    print(json.dumps(resp, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
