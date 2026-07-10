"""CARM OpenAI-Compatible API Server for BFCL Evaluation.

Architecture (v2 — CARM-routed):
  1. Parse BFCL system prompt → extract function definitions (JSON)
  2. CARM signal-based routing → select best matching function(s)
  3. LLM (Ollama) → parameter extraction only (focused prompt)
  4. Deterministic formatting → [func_name(param=value, ...)]

This replaces the v1 pure-Ollama passthrough that scored 0% on parallel
and multi_turn because qwen3-coder couldn't reliably produce [func()] format.

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
import re
import time
from http.server import HTTPServer, BaseHTTPRequestHandler

import httpx

logger = logging.getLogger("carm_bfcl_server")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

OLLAMA_BASE_URL = "http://192.168.31.8:11434"
OLLAMA_MODEL = "qwen3-coder:latest"

# Relevance threshold: if best score < this, treat as irrelevance → return []
RELEVANCE_THRESHOLD = 0.2


# ---------------------------------------------------------------------------
# Function selection (CARM-style signal matching)
# ---------------------------------------------------------------------------


def extract_functions_from_system_prompt(messages: list[dict]) -> list[dict]:
    """Extract function definitions from BFCL system prompt.

    BFCL injects functions as a JSON array in the system message,
    after the instruction text. We find the JSON array and parse it.
    """
    for msg in messages:
        if msg.get("role") != "system":
            continue
        content = msg.get("content", "")
        # BFCL puts functions as a JSON array, usually at the end of system prompt
        # Find the last JSON array in the content using regex for [ followed by optional whitespace then {
        matches = list(re.finditer(r"\[\s*\{", content))
        if not matches:
            continue
        # Try from the last match
        for match in reversed(matches):
            idx = match.start()
            # Find matching closing bracket
            bracket_depth = 0
            end_idx = -1
            for i in range(idx, len(content)):
                if content[i] == "[":
                    bracket_depth += 1
                elif content[i] == "]":
                    bracket_depth -= 1
                    if bracket_depth == 0:
                        end_idx = i
                        break
            if end_idx == -1:
                continue
            json_str = content[idx : end_idx + 1]
            try:
                funcs = json.loads(json_str)
                if isinstance(funcs, list) and all(
                    isinstance(f, dict) and "name" in f for f in funcs
                ):
                    return funcs
            except json.JSONDecodeError:
                pass
    return []


def extract_user_query(messages: list[dict]) -> str:
    """Extract the last user message content."""
    for msg in reversed(messages):
        if msg.get("role") == "user":
            return msg.get("content", "")
    return ""


def tokenize(text: str) -> set[str]:
    """Tokenize text into lowercase word tokens for matching.

    Filters out common English stop words to avoid false matches.
    """
    STOP_WORDS = {
        "a",
        "an",
        "the",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "of",
        "in",
        "on",
        "at",
        "to",
        "for",
        "with",
        "by",
        "from",
        "as",
        "and",
        "or",
        "but",
        "not",
        "no",
        "if",
        "then",
        "else",
        "when",
        "this",
        "that",
        "these",
        "those",
        "it",
        "its",
        "i",
        "you",
        "he",
        "she",
        "we",
        "they",
        "my",
        "your",
        "his",
        "her",
        "our",
        "their",
        "do",
        "does",
        "did",
        "will",
        "would",
        "should",
        "could",
        "can",
        "may",
        "might",
        "must",
        "shall",
        "have",
        "has",
        "had",
        "get",
        "got",
        "make",
        "made",
        "go",
        "went",
        "about",
        "into",
        "out",
        "up",
        "down",
        "over",
        "under",
        "again",
        "also",
        "than",
        "too",
        "very",
        "just",
        "only",
        "more",
        "most",
        "some",
        "any",
        "all",
        "each",
        "every",
        "other",
        "such",
        "own",
        "same",
        "so",
        "what",
        "which",
        "who",
        "whom",
        "whose",
        "where",
        "why",
        "how",
        "like",
        "there",
        "here",
        "now",
        "then",
        "today",
        "tomorrow",
    }
    # Split on non-alphanumeric (works for both English and Chinese)
    tokens = set(re.findall(r"[a-zA-Z_][a-zA-Z0-9_]*", text.lower()))
    # Also extract Chinese words (2+ chars)
    cn_tokens = set(re.findall(r"[\u4e00-\u9fff]{2,}", text))
    tokens.update(cn_tokens)
    # Remove stop words
    tokens -= STOP_WORDS
    return tokens


def score_function_relevance(func: dict, query: str) -> float:
    """Score how relevant a function is to the user query (0.0 - 1.0).

    Combines:
    - Function name token overlap with query
    - Description keyword overlap
    - Parameter name overlap
    - Direct substring match of function name in query
    """
    query_lower = query.lower()
    query_tokens = tokenize(query)
    if not query_tokens:
        return 0.0

    func_name = func.get("name", "")
    func_desc = func.get("description", "")
    params = func.get("parameters", {})
    param_props = params.get("properties", {})
    param_names = list(param_props.keys())

    score = 0.0

    # 1. Direct function name substring in query (strong signal)
    func_name_lower = func_name.lower()
    # Handle dotted names like "math.triangle_area" → check both full and last part
    name_parts = func_name_lower.split(".")
    for part in name_parts:
        if len(part) > 2 and part in query_lower:
            score += 0.4
    if func_name_lower in query_lower:
        score += 0.2

    # 2. Function name token overlap
    name_tokens = tokenize(func_name)
    if name_tokens:
        # Split snake_case
        expanded = set()
        for t in name_tokens:
            expanded.add(t)
            expanded.update(t.split("_"))
        overlap = len(expanded & query_tokens)
        score += min(overlap * 0.15, 0.3)

    # 3. Description keyword overlap
    desc_tokens = tokenize(func_desc)
    if desc_tokens:
        overlap = len(desc_tokens & query_tokens)
        # Normalize by description length to avoid bias toward long descriptions
        score += min(overlap * 0.1, 0.2)

    # 4. Parameter name overlap
    param_tokens = set()
    for pn in param_names:
        param_tokens.update(tokenize(pn))
    if param_tokens:
        overlap = len(param_tokens & query_tokens)
        score += min(overlap * 0.12, 0.24)

    # 5. Semantic hints: common action verbs in description that appear in query
    # Only use specific triggers, not generic phrases like "what is" or "find"
    action_hints = {
        "calculate": ["calculate", "compute", "求", "计算"],
        "convert": ["convert", "transform", "转换"],
        "search": ["search", "lookup", "query", "查找", "搜索"],
        "check": ["check", "verify", "validate", "检查", "验证"],
        "create": ["create", "generate", "build", "创建", "生成"],
        "delete": ["delete", "remove", "drop", "删除"],
        "update": ["update", "modify", "更新", "修改"],
        "schedule": ["schedule", "book", "arrange", "预约", "安排"],
        "send": ["send", "email", "notify", "发送", "邮件"],
        "translate": ["translate", "translation", "翻译"],
    }
    desc_lower = func_desc.lower()
    for action, triggers in action_hints.items():
        if action in desc_lower:
            for trigger in triggers:
                if trigger in query_lower:
                    score += 0.1
                    break

    return min(score, 1.0)


def select_functions(functions: list[dict], query: str) -> list[tuple[dict, float]]:
    """Select the best matching function(s) for the query.

    Returns list of (function, score) tuples, sorted by score descending.
    May return multiple functions for parallel calls.
    """
    scored = [(f, score_function_relevance(f, query)) for f in functions]
    scored.sort(key=lambda x: x[1], reverse=True)

    # Filter by threshold
    relevant = [(f, s) for f, s in scored if s >= RELEVANCE_THRESHOLD]

    if not relevant:
        return []

    # Check for parallel call: if query contains separators and multiple
    # functions have similar high scores
    parallel_separators = [",", " and ", " then ", "；", "，", " and also ", " also "]
    has_separator = any(sep in query for sep in parallel_separators)

    if has_separator and len(relevant) >= 2:
        # If top-2 scores are close (within 0.2), treat as parallel
        if relevant[0][1] - relevant[1][1] < 0.2 and relevant[1][1] > 0.2:
            # Return top functions that are close in score
            result = [relevant[0]]
            for i in range(1, len(relevant)):
                if relevant[0][1] - relevant[i][1] < 0.3:
                    result.append(relevant[i])
                else:
                    break
            return result

    # Single function (top-1)
    return [relevant[0]]


# ---------------------------------------------------------------------------
# Parameter extraction (LLM)
# ---------------------------------------------------------------------------


def extract_params_via_llm(
    func: dict,
    query: str,
    ollama_url: str,
    ollama_model: str,
) -> dict:
    """Use LLM to extract parameter values from the query for the given function.

    Returns a dict of param_name → value.
    """
    func_name = func.get("name", "")
    params = func.get("parameters", {})
    param_props = params.get("properties", {})
    required = params.get("required", [])

    # Build parameter description for the prompt
    param_lines = []
    for pname, pinfo in param_props.items():
        ptype = pinfo.get("type", "any")
        pdesc = pinfo.get("description", "")
        req = "required" if pname in required else "optional"
        default = pinfo.get("default", None)
        default_str = f", default={default!r}" if default is not None else ""
        param_lines.append(f"  - {pname} ({ptype}, {req}): {pdesc}{default_str}")
    param_desc = "\n".join(param_lines) if param_lines else "  (no parameters)"

    prompt = f"""Extract parameter values from the user query for the function "{func_name}".

Function description: {func.get("description", "")}

Parameters:
{param_desc}

User query: {query}

Rules:
1. Return ONLY a JSON object with parameter names as keys and extracted values as values.
2. Use the correct Python type (int, float, string, list, etc.).
3. If a parameter is not mentioned in the query, omit it (don't set it to null).
4. For optional parameters with defaults, only include if explicitly mentioned.
5. For array/list type parameters, use JSON array syntax.
6. Do NOT include any explanation, just the JSON object.

Example output: {{"base": 10, "height": 5}}

JSON:"""

    messages = [{"role": "user", "content": prompt}]

    try:
        with httpx.Client(timeout=120.0) as client:
            resp = client.post(
                f"{ollama_url}/api/chat",
                json={
                    "model": ollama_model,
                    "messages": messages,
                    "stream": False,
                    "options": {
                        "temperature": 0.001,
                        "num_predict": 512,
                    },
                    "format": "json",
                },
            )
            resp.raise_for_status()
            data = resp.json()
            content = data.get("message", {}).get("content", "")

        # Parse JSON from response
        # Try direct parse first (Ollama format=json should give clean JSON)
        try:
            params_dict = json.loads(content)
            if isinstance(params_dict, dict):
                return params_dict
        except json.JSONDecodeError:
            pass

        # Fallback: extract JSON from text
        json_match = re.search(r"\{[^}]+\}", content, re.DOTALL)
        if json_match:
            try:
                params_dict = json.loads(json_match.group())
                if isinstance(params_dict, dict):
                    return params_dict
            except json.JSONDecodeError:
                pass

        logger.warning(f"Failed to parse LLM param extraction: {content[:200]}")
        return {}

    except Exception as e:
        logger.error(f"LLM param extraction failed: {e}")
        return {}


# ---------------------------------------------------------------------------
# Output formatting (deterministic)
# ---------------------------------------------------------------------------


def format_function_call(func_name: str, params: dict) -> str:
    """Format as BFCL expected output: [func_name(param1=value1, param2=value2)].

    Handles Python literal formatting for strings, numbers, lists, etc.
    """
    if not params:
        return f"[{func_name}()]"

    parts = []
    for k, v in params.items():
        # Format value as Python literal
        if isinstance(v, str):
            # Escape quotes in string values
            escaped = v.replace("\\", "\\\\").replace('"', '\\"')
            parts.append(f'{k}="{escaped}"')
        elif isinstance(v, bool):
            parts.append(f"{k}={v}")
        elif isinstance(v, (int, float)):
            parts.append(f"{k}={v}")
        elif isinstance(v, list):
            # Format list as Python list
            parts.append(f"{k}={v!r}")
        elif isinstance(v, dict):
            parts.append(f"{k}={v!r}")
        elif v is None:
            parts.append(f"{k}=None")
        else:
            parts.append(f"{k}={v!r}")

    return f"[{func_name}({', '.join(parts)})]"


def format_parallel_output(calls: list[tuple[str, dict]]) -> str:
    """Format multiple function calls: [func1(...), func2(...)]."""
    if len(calls) == 1:
        return format_function_call(calls[0][0], calls[0][1])
    parts = [format_function_call(name, params) for name, params in calls]
    # Remove outer brackets from each and join
    inner_parts = [p[1:-1] for p in parts]  # strip [ and ]
    return f"[{', '.join(inner_parts)}]"


# ---------------------------------------------------------------------------
# Main routing pipeline
# ---------------------------------------------------------------------------


def carm_route_bfcl(
    messages: list[dict],
    ollama_url: str,
    ollama_model: str,
) -> str:
    """Main CARM routing pipeline for BFCL.

    1. Extract functions from system prompt
    2. Extract user query
    3. Select best function(s) via signal matching
    4. Extract params via LLM
    5. Format as [func(param=value)]
    """
    # Step 1: Extract functions
    functions = extract_functions_from_system_prompt(messages)
    if not functions:
        logger.warning("No functions found in system prompt, falling back to LLM")
        # Fallback: just call Ollama directly
        result = call_ollama(messages, 0.001, ollama_url, ollama_model)
        return result["content"]

    # Step 2: Extract user query
    query = extract_user_query(messages)
    if not query:
        return "[]"

    logger.info(f"Query: {query[:100]}...")
    logger.info(f"Functions: {[f['name'] for f in functions]}")

    # Step 3: Select functions
    selected = select_functions(functions, query)

    if not selected:
        # Irrelevance: no function matches
        logger.info("No relevant function found → returning []")
        return "[]"

    logger.info(f"Selected: {[(f['name'], f'{s:.2f}') for f, s in selected]}")

    # Step 4: Extract params for each selected function
    calls = []
    for func, score in selected:
        params = extract_params_via_llm(func, query, ollama_url, ollama_model)
        calls.append((func["name"], params))
        logger.info(f"  {func['name']} params: {params}")

    # Step 5: Format output
    output = format_parallel_output(calls)
    logger.info(f"Output: {output}")
    return output


# ---------------------------------------------------------------------------
# Ollama client (kept for fallback)
# ---------------------------------------------------------------------------


def call_ollama(
    messages: list[dict],
    temperature: float = 0.001,
    ollama_url: str = None,
    ollama_model: str = None,
) -> dict:
    """Call Ollama Chat API and return response."""
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
    """OpenAI-compatible API handler with CARM routing."""

    # Class-level config (set by main())
    ollama_url = OLLAMA_BASE_URL
    ollama_model = OLLAMA_MODEL

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

        start = time.time()
        content = carm_route_bfcl(messages, self.ollama_url, self.ollama_model)
        latency = time.time() - start

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
                        "content": content,
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
            },
            "latency": latency,
        }
        self._send_json(200, response)

    def _handle_completions(self):
        """Legacy completions endpoint."""
        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length).decode("utf-8")
        try:
            req = json.loads(body)
        except json.JSONDecodeError:
            self._send_json(400, {"error": "invalid JSON"})
            return

        prompt = req.get("prompt", "")

        # Convert prompt string to messages
        messages = [{"role": "user", "content": prompt}]
        content = carm_route_bfcl(messages, self.ollama_url, self.ollama_model)

        response = {
            "id": f"cmpl-{int(time.time())}",
            "object": "text_completion",
            "created": int(time.time()),
            "model": "carm-router",
            "choices": [
                {
                    "index": 0,
                    "text": content,
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
            },
        }
        self._send_json(200, response)

    def log_message(self, format, *args):
        logger.info(f"{self.client_address[0]} - {format % args}")


def main():
    global OLLAMA_BASE_URL, OLLAMA_MODEL, RELEVANCE_THRESHOLD

    parser = argparse.ArgumentParser(
        description="CARM BFCL API Server (v2 — CARM-routed)"
    )
    parser.add_argument("--port", type=int, default=11400, help="Server port")
    parser.add_argument("--host", default="0.0.0.0", help="Server host")
    parser.add_argument("--ollama-url", default=OLLAMA_BASE_URL, help="Ollama base URL")
    parser.add_argument(
        "--ollama-model", default=OLLAMA_MODEL, help="Ollama model name"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=RELEVANCE_THRESHOLD,
        help="Relevance threshold for function selection",
    )
    args = parser.parse_args()

    OLLAMA_BASE_URL = args.ollama_url
    OLLAMA_MODEL = args.ollama_model

    # Update class-level config
    CARMServerHandler.ollama_url = OLLAMA_BASE_URL
    CARMServerHandler.ollama_model = OLLAMA_MODEL

    # Update threshold
    RELEVANCE_THRESHOLD = args.threshold

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    server = HTTPServer((args.host, args.port), CARMServerHandler)
    logger.info(f"CARM BFCL Server v2 starting on {args.host}:{args.port}")
    logger.info(f"  Ollama: {OLLAMA_BASE_URL} / {OLLAMA_MODEL}")
    logger.info(f"  Relevance threshold: {RELEVANCE_THRESHOLD}")
    logger.info(f"  Architecture: CARM routing + LLM param extraction")
    logger.info(f"  Endpoints: GET /v1/models, POST /v1/chat/completions")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("Server stopped")
        server.server_close()


if __name__ == "__main__":
    main()
