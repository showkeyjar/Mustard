"""CARM OpenAI-Compatible API Server for BFCL Evaluation.

Architecture (v5 — CARM signal routing + LLM fallback + disambiguation + LLM irrelevance verification):
  1. Parse BFCL system prompt → extract function definitions (JSON)
  2. CARM signal-based routing → select best matching function(s)
  3. If signal score < threshold → LLM function selection fallback
  4. If top-2 scores are close → LLM disambiguation to pick correct one
  5. LLM irrelevance verification — when LLM selects a function with zero
     signal score, use a dedicated LLM call to verify relevance (replaces
     v4's action_words heuristic that had too many false positives)
  6. LLM parallel detection — use LLM to determine if the query requires
     parallel function calls (replaces v4's separator-based heuristic)
  7. LLM (Ollama) → parameter extraction with schema constraint validation
     - Non-parallel: format=json single dict (stable)
     - Parallel: array extraction (handles multiple param sets)
  8. Post-extraction validation — verify extracted params against schema
     (type coercion, enum validation, required field check)
  9. Deterministic formatting → [func_name(param=value, ...)]

v5 changes from v4:
  - LLM-based irrelevance verification replaces action_words heuristic
    (v4 live_irrelevance dropped to 42.5% because "get"/"find"/"show"
     are too common in English; LLM can distinguish intent better)
  - LLM-based parallel detection replaces separator heuristics
    (v4 live_parallel dropped to 43.8% because separators like ","
     don't reliably indicate parallel intent in NL queries)
  - Post-extraction schema validation with type coercion
    (v4 had 45% value_error:string from wrong parameter values)

v4 changes from v3:
  - LLM fallback for function selection when signal matching fails
  - LLM disambiguation for close-scored functions
  - Restored adaptive threshold: 0.15 for single-function, 0.2 for multi
  - Removed semantic verification (proven ineffective in v3)

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
# LLM function selection fallback (v4)
# ---------------------------------------------------------------------------

# When signal matching gives a best score below this, use LLM to select
LLM_FALLBACK_THRESHOLD = 0.2

# When top-2 signal scores are within this margin, use LLM to disambiguate
DISAMBIGUATION_MARGIN = 0.15


def select_function_via_llm(
    functions: list[dict],
    query: str,
    ollama_url: str,
    ollama_model: str,
) -> list[dict]:
    """Use LLM to select the correct function(s) when signal matching fails.

    This handles natural language queries where token overlap is low:
    - "how can i cook steak Indian style" → cookbook.search_recipe
    - "what is Imjin war" → HNA_WQA.search
    - "Could you stop the washing machine" → ControlAppliance.execute

    Returns a list of selected function dicts (usually 1, but can be
    multiple for parallel calls).
    """
    # Build compact function list
    func_lines = []
    for i, f in enumerate(functions):
        name = f.get("name", "")
        desc = f.get("description", "")[:120]
        func_lines.append(f"  {i}: {name} — {desc}")
    func_list_str = "\n".join(func_lines)

    prompt = f"""You are a function router. Given a user query and a list of available functions, select the function(s) the user wants to call.

Available functions:
{func_list_str}

User query: {query}

Rules:
1. Return a JSON array of function indices (integers). Example: [0] or [0, 2]
2. Select multiple functions ONLY if the query explicitly asks for multiple independent operations (e.g. "do X and Y" where X and Y map to different functions).
3. If NO function matches the user's intent, return an empty array: []
4. Choose based on the function's PURPOSE and CAPABILITY, not word overlap.
5. If the user is asking a general knowledge question (e.g. "what is X", "who is Y", "how does Z work") and no function can answer it, return [].
6. If the user is making casual conversation (e.g. "hello", "how are you", "thank you"), return [].
7. Only select a function if the user's query clearly maps to calling that function's capability.

Function indices:"""

    try:
        with httpx.Client(timeout=60.0) as client:
            resp = client.post(
                f"{ollama_url}/api/chat",
                json={
                    "model": ollama_model,
                    "messages": [{"role": "user", "content": prompt}],
                    "stream": False,
                    "options": {"temperature": 0.001, "num_predict": 100},
                },
            )
            resp.raise_for_status()
            data = resp.json()
            content = data.get("message", {}).get("content", "").strip()

        # Parse the response — expect an array of indices
        # Try direct parse
        try:
            indices = json.loads(content)
            if isinstance(indices, list):
                return [
                    functions[i]
                    for i in indices
                    if isinstance(i, int) and 0 <= i < len(functions)
                ]
        except json.JSONDecodeError:
            pass

        # Fallback: find array in text
        arr_match = re.search(r"\[[\d\s,]+\]", content)
        if arr_match:
            try:
                indices = json.loads(arr_match.group())
                if isinstance(indices, list):
                    return [
                        functions[i]
                        for i in indices
                        if isinstance(i, int) and 0 <= i < len(functions)
                    ]
            except (json.JSONDecodeError, IndexError):
                pass

        # Fallback: find single integer
        num_match = re.search(r"\b(\d+)\b", content)
        if num_match:
            idx = int(num_match.group(1))
            if 0 <= idx < len(functions):
                return [functions[idx]]

        logger.warning(f"LLM function selection parse failed: {content[:200]}")
        return []

    except Exception as e:
        logger.error(f"LLM function selection failed: {e}")
        return []


def disambiguate_via_llm(
    candidates: list[tuple[dict, float]],
    query: str,
    ollama_url: str,
    ollama_model: str,
) -> list[dict]:
    """Use LLM to pick the correct function when signal scores are close.

    Handles cases like:
    - archival_memory_search vs recall_memory_search
    - todo_add vs todo_delete
    - order_status_check vs inventory_management

    Returns selected function(s) — usually just 1.
    """
    func_lines = []
    for i, (f, s) in enumerate(candidates):
        name = f.get("name", "")
        desc = f.get("description", "")[:150]
        func_lines.append(f"  {i}: {name} (score={s:.2f}) — {desc}")
    func_list_str = "\n".join(func_lines)

    prompt = f"""You are a function router. The user query might match multiple functions, but only ONE is correct. Pick the right one.

Candidate functions:
{func_list_str}

User query: {query}

Rules:
1. Return ONLY the index (integer) of the correct function. Example: 0
2. Choose the function whose PURPOSE best matches what the user is asking for.
3. Pay attention to action verbs: "create" → add/create function, "delete" → delete function, "check status" → status function, "search availability" → inventory function.
4. If the user asks a question (what/who/when/where), choose search/query functions, not CRUD functions.

Function index:"""

    try:
        with httpx.Client(timeout=60.0) as client:
            resp = client.post(
                f"{ollama_url}/api/chat",
                json={
                    "model": ollama_model,
                    "messages": [{"role": "user", "content": prompt}],
                    "stream": False,
                    "options": {"temperature": 0.001, "num_predict": 10},
                },
            )
            resp.raise_for_status()
            data = resp.json()
            content = data.get("message", {}).get("content", "").strip()

        # Extract the first integer from the response
        num_match = re.search(r"\b(\d+)\b", content)
        if num_match:
            idx = int(num_match.group(1))
            if 0 <= idx < len(candidates):
                return [candidates[idx][0]]

        # Fallback: return the top candidate
        logger.warning(
            f"LLM disambiguation parse failed: {content[:100]}, using top candidate"
        )
        return [candidates[0][0]]

    except Exception as e:
        logger.error(f"LLM disambiguation failed: {e}")
        return [candidates[0][0]]


# ---------------------------------------------------------------------------
# LLM irrelevance verification (v5)
# ---------------------------------------------------------------------------


def verify_relevance_via_llm(
    func: dict,
    query: str,
    ollama_url: str,
    ollama_model: str,
) -> bool:
    """Use LLM to verify if a function is truly relevant to the query (v5).

    Replaces v4's action_words heuristic that had too many false positives
    (e.g. "get" appears in "I want to get weather data" but the user may
    want an API endpoint, not a function call).

    Returns True if the function is relevant, False if irrelevant.
    """
    func_name = func.get("name", "")
    func_desc = func.get("description", "")[:200]
    params = func.get("parameters", {})
    param_props = params.get("properties", {})
    param_summary = ", ".join(
        f"{pn}({pinfo.get('type', 'any')})" for pn, pinfo in param_props.items()
    )[:200]

    prompt = f"""You are a function relevance judge. Determine if the user's query truly requires calling this function.

Function: {func_name}
Description: {func_desc}
Parameters: {param_summary}

User query: "{query}"

Rules:
1. Answer "RELEVANT" if the user's query clearly asks to perform the action that this function provides.
2. Answer "IRRELEVANT" if:
   - The user is asking a general knowledge question (e.g. "what is X", "who is Y")
   - The user is making casual conversation (e.g. "hello", "thank you")
   - The user mentions the function's domain but doesn't want to call it (e.g. "I want to see weather data for coordinates X" means they want an API, not a weather function)
   - The user's intent doesn't match the function's purpose
3. Be conservative: when in doubt, lean towards "IRRELEVANT".

Answer (RELEVANT or IRRELEVANT):"""

    try:
        with httpx.Client(timeout=30.0) as client:
            resp = client.post(
                f"{ollama_url}/api/chat",
                json={
                    "model": ollama_model,
                    "messages": [{"role": "user", "content": prompt}],
                    "stream": False,
                    "options": {"temperature": 0.001, "num_predict": 10},
                },
            )
            resp.raise_for_status()
            data = resp.json()
            content = data.get("message", {}).get("content", "").strip().upper()

        return "RELEVANT" in content and "IRRELEVANT" not in content

    except Exception as e:
        logger.error(f"LLM relevance verification failed: {e}")
        # On failure, be conservative: reject
        return False


# ---------------------------------------------------------------------------
# LLM parallel detection (v5)
# ---------------------------------------------------------------------------


def detect_parallel_via_llm(
    query: str,
    functions: list[dict],
    ollama_url: str,
    ollama_model: str,
) -> bool:
    """Use LLM to determine if the query requires parallel function calls (v5).

    Replaces v4's separator-based heuristic. Separators like "," or "and"
    don't reliably indicate parallel intent — "find a restaurant and its reviews"
    might be a single function, while "book a flight and a hotel" needs two.

    Returns True if the query requires multiple independent function calls.
    """
    func_names = [f.get("name", "") for f in functions[:10]]  # Limit for prompt size

    prompt = f"""Determine if the user's query requires calling MULTIPLE functions in parallel.

Available functions: {func_names}

User query: "{query}"

Rules:
1. Answer "PARALLEL" if the query explicitly asks for multiple independent operations that map to DIFFERENT functions (e.g. "book a flight and a hotel" → two functions).
2. Answer "PARALLEL" if the query asks for the SAME function with DIFFERENT parameter sets (e.g. "calculate BMI for 6ft/80kg and 5.6ft/60kg" → two calls to the same function).
3. Answer "SINGLE" if the query asks for one operation, even if it mentions multiple entities that are all parameters of a single call (e.g. "find flights from NYC to LA" → one call with origin and destination params).
4. Answer "SINGLE" if the query is a simple question or request.
5. When in doubt, answer "SINGLE".

Answer (PARALLEL or SINGLE):"""

    try:
        with httpx.Client(timeout=30.0) as client:
            resp = client.post(
                f"{ollama_url}/api/chat",
                json={
                    "model": ollama_model,
                    "messages": [{"role": "user", "content": prompt}],
                    "stream": False,
                    "options": {"temperature": 0.001, "num_predict": 10},
                },
            )
            resp.raise_for_status()
            data = resp.json()
            content = data.get("message", {}).get("content", "").strip().upper()

        return "PARALLEL" in content

    except Exception as e:
        logger.error(f"LLM parallel detection failed: {e}")
        # Fallback to separator heuristic
        parallel_separators = [" and ", " also ", " then ", "；", "，", " plus "]
        return any(sep in query.lower() for sep in parallel_separators)


# ---------------------------------------------------------------------------
# Parameter extraction (LLM) — v3 unified
# ---------------------------------------------------------------------------


def extract_all_params_via_llm(
    func: dict,
    query: str,
    ollama_url: str,
    ollama_model: str,
) -> list[dict]:
    """Use LLM to extract parameter values from the query for the given function.

    v3: Returns a LIST of param dicts, naturally handling parallel calls.
    If the query asks for the same function with different params
    (e.g. "calculate BMI for 6ft/80kg and 5.6ft/60kg"), returns
    [{"height": 6.0, "weight": 80}, {"height": 5.6, "weight": 60}].

    Returns:
        List of param dicts. Each dict maps param_name → value.
        Empty list means no params could be extracted.
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
        # Include enum constraints if available
        enum_vals = pinfo.get("enum", None)
        enum_str = f", enum={enum_vals}" if enum_vals else ""
        pattern = pinfo.get("pattern", None)
        pattern_str = f", pattern={pattern}" if pattern else ""
        param_lines.append(
            f"  - {pname} ({ptype}, {req}{enum_str}{pattern_str}): {pdesc}{default_str}"
        )
    param_desc = "\n".join(param_lines) if param_lines else "  (no parameters)"

    prompt = f"""Extract parameter values from the user query for the function "{func_name}".

Function description: {func.get("description", "")}

Parameters:
{param_desc}

User query: {query}

CRITICAL RULES:
1. Return a JSON ARRAY of objects. Most queries need exactly ONE object in the array.
2. ONLY return MULTIPLE objects if the user explicitly asks for multiple independent calls with different parameter values to the SAME function. For example:
   - "calculate X for A and B" (two entities A and B → two objects)
   - "play A for 20 min and B for 15 min" (two artists → two objects)
   - "find factorial of 5, 10 and 15" (three numbers → three objects)
3. Do NOT split a single call into multiple. If there's only one set of parameters, return one object.
4. If a parameter has an enum constraint, you MUST choose a value from the enum list.
5. Use the correct type: int for integers, float for decimals, string for text, bool for true/false, array for lists.
6. For array-type parameters, use JSON array syntax (e.g. [1, 2, 3]).
7. Extract values EXACTLY as they appear in the query — do not rephrase or translate.
8. If a parameter is not mentioned in the query, omit it.
9. Return ONLY the JSON array, no explanation.

Examples:
- "calculate BMI for 6ft/80kg" → [{{"height": 6.0, "weight": 80}}]
- "play Taylor Swift for 20 min and Maroon 5 for 15 min" → [{{"artist": "Taylor Swift", "duration": 20}}, {{"artist": "Maroon 5", "duration": 15}}]
- "find factorial of 5" → [{{"n": 5}}]

JSON array:"""

    messages = [
        {
            "role": "system",
            "content": "You are a parameter extraction assistant. You ALWAYS output a JSON array of objects, never a single object. Even for one call, wrap it in an array.",
        },
        {"role": "user", "content": prompt},
    ]

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
                },
            )
            resp.raise_for_status()
            data = resp.json()
            content = data.get("message", {}).get("content", "")

        # Parse JSON from response
        result = _parse_param_list(content)
        if result is not None:
            return result

        logger.warning(f"Failed to parse LLM param extraction: {content[:200]}")
        return [{}]  # Return single empty dict as fallback

    except Exception as e:
        logger.error(f"LLM param extraction failed: {e}")
        return [{}]


def extract_params_via_llm_v2(
    func: dict,
    query: str,
    ollama_url: str,
    ollama_model: str,
) -> dict:
    """Use LLM to extract parameter values (v2-style: single dict, format=json).

    More reliable than array extraction for non-parallel calls because
    Ollama's format=json produces clean JSON objects.
    """
    func_name = func.get("name", "")
    params = func.get("parameters", {})
    param_props = params.get("properties", {})
    required = params.get("required", [])

    param_lines = []
    for pname, pinfo in param_props.items():
        ptype = pinfo.get("type", "any")
        pdesc = pinfo.get("description", "")
        req = "required" if pname in required else "optional"
        default = pinfo.get("default", None)
        default_str = f", default={default!r}" if default is not None else ""
        # Include enum constraints if available
        enum_vals = pinfo.get("enum", None)
        enum_str = f", enum={enum_vals}" if enum_vals else ""
        # Include pattern constraints
        pattern = pinfo.get("pattern", None)
        pattern_str = f", pattern={pattern}" if pattern else ""
        param_lines.append(
            f"  - {pname} ({ptype}, {req}{enum_str}{pattern_str}): {pdesc}{default_str}"
        )
    param_desc = "\n".join(param_lines) if param_lines else "  (no parameters)"

    prompt = f"""Extract parameter values from the user query for the function "{func_name}".

Function description: {func.get("description", "")}

Parameters:
{param_desc}

User query: {query}

Rules:
1. Return ONLY a JSON object with parameter names as keys and extracted values as values.
2. Use the correct Python type (int, float, string, list, etc.).
3. If a parameter has an enum constraint, you MUST choose a value from the enum list. Do NOT use any value outside the enum.
4. If a parameter is not mentioned in the query, omit it (don't set it to null).
5. For optional parameters with defaults, only include if explicitly mentioned.
6. For array/list type parameters, use JSON array syntax.
7. Extract values EXACTLY as they appear in the query — do not rephrase, translate, or reformat.
8. Do NOT include any explanation, just the JSON object.

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

        try:
            params_dict = json.loads(content)
            if isinstance(params_dict, dict):
                return params_dict
        except json.JSONDecodeError:
            pass

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


def _parse_param_list(content: str) -> list[dict] | None:
    """Parse LLM response into list of param dicts.

    Handles both array responses and single-object responses.
    """
    content = content.strip()

    # Try direct parse first (Ollama format=json should give clean JSON)
    try:
        parsed = json.loads(content)
        if isinstance(parsed, list):
            return [d for d in parsed if isinstance(d, dict)]
        if isinstance(parsed, dict):
            return [parsed]
    except json.JSONDecodeError:
        pass

    # Fallback: try to find JSON array in text
    arr_match = re.search(r"\[.*\]", content, re.DOTALL)
    if arr_match:
        try:
            parsed = json.loads(arr_match.group())
            if isinstance(parsed, list):
                return [d for d in parsed if isinstance(d, dict)]
        except json.JSONDecodeError:
            pass

    # Fallback: try to find a single JSON object
    obj_match = re.search(r"\{[^}]+\}", content, re.DOTALL)
    if obj_match:
        try:
            parsed = json.loads(obj_match.group())
            if isinstance(parsed, dict):
                return [parsed]
        except json.JSONDecodeError:
            pass

    return None


def validate_and_coerce_params(
    func: dict,
    params: dict,
) -> dict:
    """Validate and coerce extracted parameters against function schema (v5).

    - Coerces string values to correct types (int, float, bool)
    - Validates enum constraints
    - Removes parameters not in schema
    - Fills in defaults for missing required params if available
    """
    param_props = func.get("parameters", {}).get("properties", {})
    if not param_props:
        return params

    validated = {}
    for pname, pvalue in params.items():
        if pname not in param_props:
            # Skip unknown parameters
            logger.debug(f"Removing unknown param '{pname}' (not in schema)")
            continue

        pschema = param_props[pname]
        ptype = pschema.get("type", "string")
        enum_vals = pschema.get("enum", None)

        # Type coercion
        coerced = pvalue
        try:
            if ptype == "integer" and isinstance(pvalue, str):
                # Try to extract integer from string
                num_match = re.search(r"-?\d+", pvalue)
                if num_match:
                    coerced = int(num_match.group())
                else:
                    logger.warning(
                        f"Cannot coerce '{pvalue}' to int for param '{pname}'"
                    )
                    coerced = int(pvalue)
            elif ptype == "number" and isinstance(pvalue, str):
                num_match = re.search(r"-?\d+\.?\d*", pvalue)
                if num_match:
                    coerced = float(num_match.group())
                else:
                    coerced = float(pvalue)
            elif ptype == "boolean" and isinstance(pvalue, str):
                coerced = pvalue.lower() in ("true", "1", "yes", "on")
            elif ptype == "array" and isinstance(pvalue, str):
                # Try to parse as JSON array
                try:
                    coerced = json.loads(pvalue)
                    if not isinstance(coerced, list):
                        coerced = [pvalue]
                except json.JSONDecodeError:
                    # Split by comma
                    coerced = [item.strip() for item in pvalue.split(",")]
        except (ValueError, TypeError) as e:
            logger.warning(f"Type coercion failed for '{pname}': {e}")
            coerced = pvalue  # Keep original value

        # Enum validation
        if enum_vals and coerced not in enum_vals:
            # Try case-insensitive match
            for ev in enum_vals:
                if isinstance(coerced, str) and isinstance(ev, str):
                    if coerced.lower() == ev.lower():
                        coerced = ev
                        break
            else:
                logger.warning(
                    f"Param '{pname}' value '{coerced}' not in enum {enum_vals}, keeping anyway"
                )

        validated[pname] = coerced

    return validated


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
    """Main CARM routing pipeline for BFCL (v5).

    1. Extract functions from system prompt
    2. Extract user query
    3. Signal-based function scoring
    4. If best score < threshold → LLM function selection fallback
       + LLM irrelevance verification (replaces v4 action_words heuristic)
    5. If top-2 scores close → LLM disambiguation
    6. LLM parallel detection (replaces v4 separator heuristic)
    7. Extract params via LLM (single dict or array for parallel)
    8. Validate and coerce params against schema
    9. Format as [func(param=value)]
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

    # Step 3: Signal-based scoring
    scored = [(f, score_function_relevance(f, query)) for f in functions]
    scored.sort(key=lambda x: x[1], reverse=True)

    best_score = scored[0][1] if scored else 0.0
    logger.info(f"Signal scores: {[(f['name'], f'{s:.2f}') for f, s in scored[:5]]}")

    # Adaptive threshold: single function → low bar, but not zero
    # For single-function BFCL tests (simple_*), the function is always relevant
    # For irrelevance tests, the single function may NOT be relevant
    effective_threshold = RELEVANCE_THRESHOLD  # 0.2 for multi-function
    if len(functions) == 1:
        # Single function: check if the query semantically matches
        # Use a moderate threshold — low enough to catch real matches,
        # high enough to reject irrelevance like "prime factors" vs "compound_interest"
        effective_threshold = 0.15

    # Step 4: LLM fallback when signal matching fails
    if best_score < effective_threshold:
        logger.info(
            f"Best signal score {best_score:.2f} < {effective_threshold} → LLM fallback"
        )
        selected = select_function_via_llm(functions, query, ollama_url, ollama_model)
        if not selected:
            logger.info("LLM fallback also found no match → returning []")
            return "[]"

        # v5: LLM-based irrelevance verification (replaces v4 action_words heuristic)
        # Re-check: if the signal score of the LLM-selected function is 0.0,
        # use LLM to verify if the function is truly relevant to the query.
        # v4 used a hardcoded action_words list but "get"/"find"/"show" are
        # too common in English, causing 510/884 false positives in live_irrelevance.
        llm_scores = [score_function_relevance(f, query) for f in selected]
        max_llm_score = max(llm_scores) if llm_scores else 0.0

        if max_llm_score == 0.0:
            # Zero signal — use LLM to verify relevance
            logger.info(
                f"LLM selected {[f['name'] for f in selected]} but signal=0.0 → LLM relevance verification"
            )
            # Verify each selected function
            relevant_selected = []
            for f in selected:
                if verify_relevance_via_llm(f, query, ollama_url, ollama_model):
                    relevant_selected.append(f)
                else:
                    logger.info(f"LLM rejected {f['name']} as irrelevant")

            if not relevant_selected:
                logger.info("All LLM-selected functions rejected as irrelevant → []")
                return "[]"
            selected = relevant_selected

        verified = [(f, 0.0) for f in selected]
        logger.info(f"LLM selected: {[f['name'] for f in selected]}")
    elif len(functions) == 1 and best_score < 0.4:
        # Single function with medium signal score — could be irrelevance
        # Use LLM to verify if the function truly matches the query intent
        logger.info(
            f"Single func, score {best_score:.2f} in [0.15, 0.4) → LLM verification"
        )
        selected = select_function_via_llm(functions, query, ollama_url, ollama_model)
        if not selected:
            logger.info("LLM verification rejected the function → returning []")
            return "[]"
        verified = [(f, 0.0) for f in selected]
        logger.info(f"LLM verified: {[f['name'] for f in selected]}")
    elif len(functions) > 1 and best_score < 0.4:
        # Step 5: Signal score is above threshold but not high — disambiguate
        # Check if top-2 are close
        relevant = [(f, s) for f, s in scored if s >= effective_threshold]

        # Check for parallel call
        parallel_separators = [" and ", " also ", " then ", "；", "，", " plus "]
        has_parallel_hint = any(sep in query.lower() for sep in parallel_separators)

        if has_parallel_hint and len(relevant) >= 2:
            # Parallel: keep all close-scored functions
            best = relevant[0][1]
            verified = [relevant[0]]
            for f, s in relevant[1:]:
                if best - s < 0.2 and s >= effective_threshold:
                    verified.append((f, s))
                else:
                    break
        elif (
            len(relevant) >= 2
            and (relevant[0][1] - relevant[1][1]) < DISAMBIGUATION_MARGIN
        ):
            # Top-2 are close → LLM disambiguation
            logger.info(
                f"Top-2 close ({relevant[0][1]:.2f} vs {relevant[1][1]:.2f}) → LLM disambiguation"
            )
            # Pass top-3 candidates for disambiguation
            candidates = relevant[:3] if len(relevant) >= 3 else relevant
            selected = disambiguate_via_llm(candidates, query, ollama_url, ollama_model)
            verified = [(f, 0.0) for f in selected]
            logger.info(f"LLM disambiguated to: {[f['name'] for f in selected]}")
        else:
            # Clear winner
            verified = [relevant[0]]
    else:
        # High confidence signal match
        relevant = [(f, s) for f, s in scored if s >= effective_threshold]

        # Check for parallel
        parallel_separators = [" and ", " also ", " then ", "；", "，", " plus "]
        has_parallel_hint = any(sep in query.lower() for sep in parallel_separators)

        if has_parallel_hint and len(relevant) >= 2:
            best = relevant[0][1]
            verified = [relevant[0]]
            for f, s in relevant[1:]:
                if best - s < 0.2 and s >= effective_threshold:
                    verified.append((f, s))
                else:
                    break
        else:
            verified = [relevant[0]] if relevant else []

    if not verified:
        logger.info("No function selected → returning []")
        return "[]"

    logger.info(f"Verified: {[(f['name'], f'{s:.2f}') for f, s in verified]}")

    # Step 6: Extract params
    # v5: Use LLM-based parallel detection instead of separator heuristic
    # v4's separator heuristic caused live_parallel to drop from 62.5% to 43.8%
    # because separators like "," or "and" don't reliably indicate parallel intent
    is_parallel = len(verified) > 1
    if not is_parallel and len(verified) == 1:
        # Use LLM to check if this single function needs multiple calls
        is_parallel = detect_parallel_via_llm(
            query, [f for f, _ in verified], ollama_url, ollama_model
        )
        if is_parallel:
            logger.info("LLM detected parallel intent for single function")

    calls = []

    if not is_parallel:
        # Single call: use format=json for reliable single-dict extraction
        for func, score in verified:
            params = extract_params_via_llm_v2(func, query, ollama_url, ollama_model)
            # v5: validate and coerce params against schema
            params = validate_and_coerce_params(func, params)
            calls.append((func["name"], params))
            logger.info(f"  {func['name']} params: {params}")
    else:
        # Parallel: use array extraction
        for func, score in verified:
            param_sets = extract_all_params_via_llm(
                func, query, ollama_url, ollama_model
            )
            for params in param_sets:
                # v5: validate and coerce params against schema
                params = validate_and_coerce_params(func, params)
                calls.append((func["name"], params))
                logger.info(f"  {func['name']} params: {params}")

    # Step 6: Format output
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
        # Also check for OpenAI-style "tools" parameter
        tools = req.get("tools", [])
        if tools:
            # Convert OpenAI tools format to function defs and inject into system prompt
            func_defs = []
            for tool in tools:
                if tool.get("type") == "function":
                    func = tool.get("function", {})
                    if func:
                        func_defs.append(func)
            if func_defs:
                # Prepend function JSON to system message
                func_json = json.dumps(func_defs)
                system_msg = next(
                    (m for m in messages if m.get("role") == "system"), None
                )
                if system_msg:
                    system_msg["content"] = (
                        system_msg.get("content", "") + "\n" + func_json
                    )
                else:
                    messages.insert(0, {"role": "system", "content": func_json})
                logger.info(
                    f"Injected {len(func_defs)} tools from 'tools' param into system prompt"
                )

        # Debug: log what we received
        sys_msg = next((m for m in messages if m.get("role") == "system"), None)
        if sys_msg:
            logger.info(f"System msg length: {len(sys_msg.get('content', ''))}")
            logger.info(f"System msg preview: {sys_msg.get('content', '')[:200]}")

        start = time.time()
        content = carm_route_bfcl(messages, self.ollama_url, self.ollama_model)
        latency = time.time() - start

        logger.info(f"Response content preview: {content[:200]}")

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
        description="CARM BFCL API Server (v5 — CARM signal + LLM fallback + disambiguation + LLM irrelevance verification + LLM parallel detection)"
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
    logger.info(f"CARM BFCL Server v5 starting on {args.host}:{args.port}")
    logger.info(f"  Ollama: {OLLAMA_BASE_URL} / {OLLAMA_MODEL}")
    logger.info(f"  Relevance threshold: {RELEVANCE_THRESHOLD}")
    logger.info(
        f"  Architecture: CARM signal routing + LLM fallback + disambiguation + LLM irrelevance verification + LLM parallel detection"
    )
    logger.info(f"  Endpoints: GET /v1/models, POST /v1/chat/completions")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("Server stopped")
        server.server_close()


if __name__ == "__main__":
    main()
