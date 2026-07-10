"""CARM OpenAI-Compatible API Server for BFCL Evaluation.

Wraps CARM + Ollama as an OpenAI Chat Completions API endpoint.
BFCL sends messages (with function docs in system prompt), CARM forwards
to Ollama qwen3-coder, and returns the response in OpenAI format.

Usage:
    python scripts/carm_bfcl_server.py --port 11400

Then set in BFCL .env:
    OPENAI_BASE_URL=http://localhost:11400/v1
    OPENAI_API_KEY=dummy
"""

from __future__ import annotations

import argparse
import json
import logging
import time
import httpx
from http.server import HTTPServer, BaseHTTPRequestHandler

logger = logging.getLogger("carm_bfcl_server")

# ---------------------------------------------------------------------------
# Ollama client
# ---------------------------------------------------------------------------

OLLAMA_BASE_URL = "http://192.168.31.8:11434"
OLLAMA_MODEL = "qwen3-coder:latest"


def call_ollama(
    messages: list[dict],
    temperature: float = 0.001,
    ollama_url: str = None,
    ollama_model: str = None,
) -> dict:
    """Call Ollama Chat API and return response in OpenAI format."""
    base = ollama_url or OLLAMA_BASE_URL
    model = ollama_model or OLLAMA_MODEL
    try:
        with httpx.Client(timeout=300.0) as client:
            resp = client.post(
                f"{base}/api/chat",
                json={
                    "model": model,
                    "messages": messages,
                    "stream": False,
                    "options": {
                        "temperature": temperature,
                        "num_predict": 1024,
                    },
                },
            )
            resp.raise_for_status()
            data = resp.json()

        content = data.get("message", {}).get("content", "")
        prompt_eval_count = data.get("prompt_eval_count", 0)
        eval_count = data.get("eval_count", 0)

        return {
            "content": content,
            "prompt_tokens": prompt_eval_count,
            "completion_tokens": eval_count,
        }
    except Exception as e:
        logger.error(f"Ollama call failed: {e}")
        return {
            "content": f"Error: {e}",
            "prompt_tokens": 0,
            "completion_tokens": 0,
        }


# ---------------------------------------------------------------------------
# HTTP handler
# ---------------------------------------------------------------------------


class CARMServerHandler(BaseHTTPRequestHandler):
    """Minimal OpenAI-compatible API handler."""

    def _send_json(self, code: int, data: dict):
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/v1/models" or self.path == "/models":
            self._send_json(200, {"data": [{"id": "carm-router", "object": "model"}]})
        elif self.path == "/health":
            self._send_json(200, {"status": "ok"})
        else:
            self._send_json(404, {"error": "not found"})

    def do_POST(self):
        if self.path == "/v1/chat/completions" or self.path == "/chat/completions":
            self._handle_chat_completions()
        elif self.path == "/v1/completions" or self.path == "/completions":
            self._handle_completions()
        else:
            self._send_json(404, {"error": "not found"})

    def _handle_chat_completions(self):
        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length).decode("utf-8")
        try:
            req = json.loads(body)
        except json.JSONDecodeError:
            self._send_json(400, {"error": "invalid JSON"})
            return

        messages = req.get("messages", [])
        temperature = req.get("temperature", 0.001)
        # max_tokens = req.get("max_tokens", 1024)

        start = time.time()
        result = call_ollama(messages, temperature)
        latency = time.time() - start

        # Build OpenAI Chat Completions response
        response = {
            "id": f"chatcmpl-{int(time.time())}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": "carm-router",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": result["content"],
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": result["prompt_tokens"],
                "completion_tokens": result["completion_tokens"],
                "total_tokens": result["prompt_tokens"] + result["completion_tokens"],
            },
            "latency": latency,
        }
        self._send_json(200, response)

    def _handle_completions(self):
        """Legacy completions endpoint (for OSS handler compatibility)."""
        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length).decode("utf-8")
        try:
            req = json.loads(body)
        except json.JSONDecodeError:
            self._send_json(400, {"error": "invalid JSON"})
            return

        prompt = req.get("prompt", "")
        temperature = req.get("temperature", 0.001)

        # Convert prompt string to messages
        messages = [{"role": "user", "content": prompt}]
        result = call_ollama(messages, temperature)

        response = {
            "id": f"cmpl-{int(time.time())}",
            "object": "text_completion",
            "created": int(time.time()),
            "model": "carm-router",
            "choices": [
                {
                    "index": 0,
                    "text": result["content"],
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": result["prompt_tokens"],
                "completion_tokens": result["completion_tokens"],
                "total_tokens": result["prompt_tokens"] + result["completion_tokens"],
            },
        }
        self._send_json(200, response)

    def log_message(self, format, *args):
        logger.info(f"{self.client_address[0]} - {format % args}")


def main():
    global OLLAMA_BASE_URL, OLLAMA_MODEL

    parser = argparse.ArgumentParser(description="CARM BFCL API Server")
    parser.add_argument("--port", type=int, default=11400, help="Server port")
    parser.add_argument("--host", default="0.0.0.0", help="Server host")
    parser.add_argument("--ollama-url", default=OLLAMA_BASE_URL, help="Ollama base URL")
    parser.add_argument(
        "--ollama-model", default=OLLAMA_MODEL, help="Ollama model name"
    )
    args = parser.parse_args()

    OLLAMA_BASE_URL = args.ollama_url
    OLLAMA_MODEL = args.ollama_model

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    server = HTTPServer((args.host, args.port), CARMServerHandler)
    logger.info(f"CARM BFCL Server starting on {args.host}:{args.port}")
    logger.info(f"  Ollama: {OLLAMA_BASE_URL} / {OLLAMA_MODEL}")
    logger.info(f"  Endpoints: GET /v1/models, POST /v1/chat/completions")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("Server stopped")
        server.server_close()


if __name__ == "__main__":
    main()
