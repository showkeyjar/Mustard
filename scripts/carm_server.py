"""CARM REST API Server — OpenAI-compatible + native CARM endpoints.

Unlike carm_bfcl_server.py (which transparently proxies Ollama),
this server uses CARMRouter's actual routing logic to:
  1. Detect intent and select the best tool
  2. Execute the tool (calculator, search, code, LLM proxy)
  3. Return structured results

Endpoints:
  POST /v1/chat/completions   — OpenAI-compatible (content = CARM result)
  POST /v1/route              — Native CARM route (returns tool_name, result, confidence)
  POST /v1/route/parallel     — Parallel routing (returns multiple sub-results)
  GET  /v1/tools              — List registered tools
  GET  /v1/tools/{name}/info  — Tool metadata
  POST /v1/tools/register     — Register a custom tool (via JSON spec)
  GET  /v1/sessions/{id}      — Get session history
  GET  /health                — Health check
  GET  /v1/models             — OpenAI-compatible model list

Usage:
    python -m scripts.carm_server --port 8000
    python -m scripts.carm_server --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from urllib.parse import urlparse

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.environ.setdefault("CARM_NO_EMBEDDING", "1")

from carm.router import CARMRouter, RouteResult

logger = logging.getLogger("carm_server")

# ---------------------------------------------------------------------------
# Global router instance (initialized in main)
# ---------------------------------------------------------------------------

_router: CARMRouter | None = None


def get_router() -> CARMRouter:
    global _router
    if _router is None:
        _router = CARMRouter()
    return _router


# ---------------------------------------------------------------------------
# HTTP handler
# ---------------------------------------------------------------------------


class CARMHandler(BaseHTTPRequestHandler):
    """HTTP handler for CARM REST API."""

    def _send_json(self, code: int, data: dict | list):
        body = json.dumps(data, ensure_ascii=False, indent=2).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_body(self) -> dict:
        content_length = int(self.headers.get("Content-Length", 0))
        if content_length == 0:
            return {}
        body = self.rfile.read(content_length).decode("utf-8")
        try:
            return json.loads(body)
        except json.JSONDecodeError:
            return {}

    # ── GET routes ──────────────────────────────────────────────────────

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path.rstrip("/") or "/"

        if path in ("/health", "/"):
            self._send_json(200, {"status": "ok", "service": "carm-server"})
        elif path in ("/v1/models", "/models"):
            self._send_json(200, {"data": [{"id": "carm", "object": "model"}]})
        elif path in ("/v1/tools", "/tools"):
            self._handle_list_tools()
        elif path.startswith("/v1/tools/") and path.endswith("/info"):
            tool_name = path.split("/")[3]
            self._handle_tool_info(tool_name)
        elif path.startswith("/v1/sessions/"):
            session_id = path.split("/")[3]
            self._handle_get_session(session_id)
        else:
            self._send_json(404, {"error": f"not found: {path}"})

    # ── POST routes ─────────────────────────────────────────────────────

    def do_POST(self):
        parsed = urlparse(self.path)
        path = parsed.path.rstrip("/")

        if path in ("/v1/chat/completions", "/chat/completions"):
            self._handle_chat_completions()
        elif path in ("/v1/route", "/route"):
            self._handle_route()
        elif path in ("/v1/route/parallel", "/route/parallel"):
            self._handle_route_parallel()
        elif path in ("/v1/tools/register", "/tools/register"):
            self._handle_register_tool()
        else:
            self._send_json(404, {"error": f"not found: {path}"})

    # ── Handlers ────────────────────────────────────────────────────────

    def _handle_chat_completions(self):
        """OpenAI-compatible chat completions endpoint.

        Accepts standard OpenAI messages format, extracts the last user
        message, routes it through CARM, and returns the result as the
        assistant message content.
        """
        req = self._read_body()
        messages = req.get("messages", [])
        session_id = req.get("session_id") or req.get("user")

        # Extract last user message
        user_query = ""
        for msg in reversed(messages):
            if msg.get("role") == "user":
                user_query = msg.get("content", "")
                break

        if not user_query:
            self._send_json(400, {"error": "no user message found"})
            return

        router = get_router()
        start = time.time()
        use_parallel = req.get("parallel", True)

        if use_parallel:
            result = router.route_parallel(user_query, session_id=session_id)
        else:
            result = router.route(user_query, session_id=session_id)
        latency = time.time() - start

        response = {
            "id": f"chatcmpl-{int(time.time())}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": "carm",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": result.result,
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": len(user_query),  # approximate
                "completion_tokens": len(result.result),
                "total_tokens": len(user_query) + len(result.result),
            },
            "carm_meta": {
                "tool_name": result.tool_name,
                "confidence": result.confidence,
                "source": result.source,
                "ok": result.ok,
                "latency_ms": round(latency * 1000),
                "sub_results": [
                    {
                        "tool_name": sr.tool_name,
                        "result": sr.result,
                        "confidence": sr.confidence,
                    }
                    for sr in result.sub_results
                ]
                if result.sub_results
                else None,
            },
        }
        self._send_json(200, response)

    def _handle_route(self):
        """Native CARM route endpoint — returns full structured result."""
        req = self._read_body()
        query = req.get("query", "").strip()
        if not query:
            self._send_json(400, {"error": "missing 'query' field"})
            return

        session_id = req.get("session_id")
        dry_run = req.get("dry_run", False)
        timeout = req.get("timeout")

        router = get_router()
        start = time.time()
        result = router.route(
            query, session_id=session_id, dry_run=dry_run, timeout=timeout
        )
        latency = time.time() - start

        self._send_json(
            200,
            {
                "query": result.query,
                "tool_name": result.tool_name,
                "result": result.result,
                "confidence": result.confidence,
                "source": result.source,
                "ok": result.ok,
                "intent_category": result.intent_category.value
                if result.intent_category
                else None,
                "session_id": result.session_id,
                "latency_ms": round(latency * 1000),
            },
        )

    def _handle_route_parallel(self):
        """Parallel routing endpoint — splits and executes multiple sub-queries."""
        req = self._read_body()
        query = req.get("query", "").strip()
        if not query:
            self._send_json(400, {"error": "missing 'query' field"})
            return

        session_id = req.get("session_id")
        dry_run = req.get("dry_run", False)
        timeout = req.get("timeout")
        max_workers = req.get("max_workers", 4)

        router = get_router()
        start = time.time()
        result = router.route_parallel(
            query,
            session_id=session_id,
            dry_run=dry_run,
            timeout=timeout,
            max_workers=max_workers,
        )
        latency = time.time() - start

        self._send_json(
            200,
            {
                "query": result.query,
                "tool_name": result.tool_name,
                "result": result.result,
                "confidence": result.confidence,
                "source": result.source,
                "ok": result.ok,
                "latency_ms": round(latency * 1000),
                "sub_results": [
                    {
                        "query": sr.query,
                        "tool_name": sr.tool_name,
                        "result": sr.result,
                        "confidence": sr.confidence,
                        "ok": sr.ok,
                    }
                    for sr in result.sub_results
                ]
                if result.sub_results
                else [],
            },
        )

    def _handle_list_tools(self):
        router = get_router()
        tools = []
        for name in router.tool_names:
            tools.append({"name": name, "registered": True})
        self._send_json(200, {"tools": tools, "count": len(tools)})

    def _handle_tool_info(self, tool_name: str):
        router = get_router()
        if not router._tool_manager.has_tool(tool_name):
            self._send_json(404, {"error": f"tool '{tool_name}' not found"})
            return
        tool = router._tool_manager._tools[tool_name]
        tags = getattr(tool, "capability_tags", [])
        self._send_json(
            200,
            {
                "name": tool.name,
                "capability_tags": [t.value for t in tags],
            },
        )

    def _handle_get_session(self, session_id: str):
        from carm.session_memory import SessionMemoryManager

        mgr = SessionMemoryManager.get_instance()
        ctx = mgr.get_or_create(session_id)
        turns = []
        for turn in ctx.turns:
            turns.append(
                {
                    "user_input": turn.user_input,
                    "tool_name": turn.tool_name,
                    "tool_result": turn.tool_result,
                    "confidence": turn.confidence,
                    "timestamp": turn.timestamp,
                }
            )
        self._send_json(
            200,
            {
                "session_id": session_id,
                "turn_count": len(turns),
                "turns": turns,
            },
        )

    def _handle_register_tool(self):
        """Register a custom tool via JSON spec.

        Expected body:
        {
            "name": "my_tool",
            "module": "my_module",     // Python module path
            "class": "MyTool",         // Class name
            "capability_tags": ["SEARCH"]  // IntentCategory values
        }
        """
        req = self._read_body()
        name = req.get("name", "")
        module_path = req.get("module", "")
        class_name = req.get("class", "")

        if not name or not module_path or not class_name:
            self._send_json(
                400, {"error": "missing required fields: name, module, class"}
            )
            return

        try:
            import importlib

            mod = importlib.import_module(module_path)
            cls = getattr(mod, class_name)
            tool = cls()
            router = get_router()
            router.register_tool(tool)

            tags = req.get("capability_tags", [])
            if tags:
                from carm.intent import IntentCategory

                for tag_str in tags:
                    tag = IntentCategory(tag_str)
                    router.set_primary(name, tag)

            self._send_json(
                200,
                {
                    "status": "registered",
                    "name": name,
                    "capability_tags": tags,
                },
            )
        except Exception as e:
            self._send_json(500, {"error": f"registration failed: {e}"})

    def log_message(self, fmt, *args):
        logger.info(f"{self.client_address[0]} - {fmt % args}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="CARM REST API Server")
    parser.add_argument(
        "--host", default="0.0.0.0", help="Server host (default: 0.0.0.0)"
    )
    parser.add_argument(
        "--port", type=int, default=8000, help="Server port (default: 8000)"
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    # Pre-initialize router
    global _router
    _router = CARMRouter()
    logger.info(f"CARM Server starting on {args.host}:{args.port}")
    logger.info(f"  Tools: {_router.tool_names}")
    logger.info(f"  Endpoints:")
    logger.info(f"    POST /v1/chat/completions   (OpenAI-compatible)")
    logger.info(f"    POST /v1/route              (native CARM route)")
    logger.info(f"    POST /v1/route/parallel     (parallel routing)")
    logger.info(f"    GET  /v1/tools              (list tools)")
    logger.info(f"    GET  /v1/sessions/{{id}}     (session history)")
    logger.info(f"    GET  /health                (health check)")

    server = HTTPServer((args.host, args.port), CARMHandler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("Server stopped")
        server.server_close()


if __name__ == "__main__":
    main()
