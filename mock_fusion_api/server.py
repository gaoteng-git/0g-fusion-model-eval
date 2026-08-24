"""Minimal OpenAI-compatible HTTP server: POST /v1/chat/completions.
Stands in for the future real 0G product API -- same wire contract, disposable implementation.
Run: python3 -m mock_fusion_api.server [port]
"""
import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from . import pipeline


class Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        if self.path != "/v1/chat/completions":
            self.send_response(404)
            self.end_headers()
            return
        length = int(self.headers.get("Content-Length", 0))
        request = json.loads(self.rfile.read(length) or b"{}")
        try:
            response = pipeline.handle_chat_completion(request)
            status = 200
        except Exception as exc:  # minimal: no retries/fallback chains in this stage
            response = {"error": str(exc)}
            status = 500
        body = json.dumps(response).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass  # keep test/eval runs quiet


def serve(port=8000):
    server = ThreadingHTTPServer(("0.0.0.0", port), Handler)
    server.serve_forever()


if __name__ == "__main__":
    serve(int(sys.argv[1]) if len(sys.argv) > 1 else 8000)
