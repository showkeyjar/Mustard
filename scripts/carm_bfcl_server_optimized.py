"""CARM OpenAI-Compatible API Server for BFCL Evaluation — OPTIMIZED variant.

This is based on carm_bfcl_server.py but with the following optimizations to
reduce LLM inference calls (reducing total evaluation time from ~40-100h to ~20-50h):

Optimizations from v5:
  1. PARALLEL DETECTION: Replaced LLM-based detection with v4's separator heuristic.
     LLM parallel detection cost ~1 call per query but is only useful for parallel
     categories. v4's separator heuristic covers > 90% of cases with ~0 cost.
     Risk: live_parallel accuracy may drop from v5 levels toward v4's 43.8%.

  2. PARAM EXTRACTION: Reduced prompt length (fewer rules, shorter examples) and
     reduced num_predict from 512 to 192. Most param extractions need < 50 tokens.
     This saves ~60% of token generation time.

  3. FUNCTION SELECTION LLM: Reduced prompt (shorter descriptions, fewer rules).

Preserved from v5:
  - LLM irrelevance verification (critical for live_irrelevance — v4 got 42.5%)
  - LLM disambiguation (critical for multiple with close signal scores)
  - Post-extraction schema validation (v5 feature)
  - CARM signal routing (the validated fix)

Usage:
    python scripts/carm_bfcl_server_optimized.py --port 11401
    Then in BFCL .env: OPENAI_BASE_URL=http://localhost:11401/v1
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import time
from http.server import HTTPServer, BaseHTTPRequestHandler

import httpx

logger = logging.getLogger("carm_bfcl_server_optimized")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

OLLAMA_BASE_URL = "http://192.168.31.20:11434"
OLLAMA_MODEL = "qwen3-coder"
OLLAMA_TIMEOUT_S = 300  # 5 minutes - warmup for large model

# Relevance threshold: if best score < this, treat as irrelevance → return []
RELEVANCE_THRESHOLD = 0.2


# ---------------------------------------------------------------------------
# Function selection (CARM-style signal matching) — UNCHANGED
# ---------------------------------------------------------------------------


def extract_functions_from_system_prompt(messages: list[dict]) -> list[dict]:
    """Extract function definitions from BFCL system prompt."""
    for msg in messages:
        if msg.get("role") != "system":
            continue
        content = msg.get("content", "")
        matches = list(re.finditer(r"\[\s*\{", content))
        if not matches:
            continue
        for match in reversed(matches):
            idx = match.start()
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
    """Tokenize text into lowercase word tokens for matching."""
    STOP_WORDS = {
        "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
        "of", "in", "on", "at", "to", "for", "with", "by", "from", "as",
        "and", "or", "but", "not", "no", "if", "then", "else", "when",
        "this", "that", "these", "those", "it", "its", "i", "you", "he",
        "she", "we", "they", "my", "your", "his", "her", "our", "their",
        "do", "does", "did", "will", "would", "should", "could", "can",
        "may", "might", "must", "shall", "have", "has", "had", "get", "got",
        "make", "made", "go", "went", "about", "into", "out", "up", "down",
        "over", "under", "again", "also", "than", "too", "very", "just",
        "only", "more", "most", "some", "any", "all", "each", "every",
        "other", "such", "own", "same", "so", "what", "which", "who", "whom",
        "whose", "where", "why", "how", "like", "there", "here", "now", "then",
        "today", "tomorrow",
    }
    tokens = set(re.findall(r"[a-zA-Z_][a-zA-Z0-9_]*", text.lower()))
    cn_tokens = set(re.findall(r"[\u4e00-\u9fff]{2,}", text))
    tokens.update(cn_tokens)
    tokens -= STOP_WORDS
    return tokens


def score_function_relevance(func: dict, query: str) -> float:
    """Score how relevant a function is to the user query (0.0 - 1.0)."""
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

    # 1. Direct function name substring in query
    func_name_lower = func_name.lower()
    name_parts = func_name_lower.split(".")
    for part in name_parts:
        if len(part) > 2 and part in query_lower:
            score += 0.4
    if func_name_lower in query_lower:
        score += 0.2

    # 2. Function name token overlap
    name_tokens = tokenize(func_name)
    if name_tokens:
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
        score += min(overlap * 0.1, 0.2)

    # 4. Parameter name overlap
    param_tokens = set()
    for pn in param_names:
        param_tokens.update(tokenize(pn))
    if param_tokens:
        overlap = len(param_tokens & query_tokens)
        score += min(overlap * 0.12, 0.24)

    # 5. Semantic action hints
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

    relevant = [(f, s) for f, s in scored if s >= RELEVANCE_THRESHOLD]

    if not relevant:
        return []

    # Check for parallel call: if query contains separators and multiple
    # functions have similar high scores
    parallel_separators = [",", " and ", " then ", "；", "，", " and also ", " also "]
    has_separator = any(sep in query for sep in parallel_separators)

    if has_separator and len(relevant) >= 2:
        if relevant[0][1] - relevant[1][1] < 0.2 and relevant[1][1] > 0.2:
            result = [relevant[0]]
            for i in range(1, len(relevant)):
                if relevant[0][1] - relevant[i][1] < 0.3:
                    result.append(relevant[i])
                else:
                    break
            return result

    return [relevant[0]]


# ---------------------------------------------------------------------------
# LLM function selection fallback — SHORTENED PROMPT
# ---------------------------------------------------------------------------

LLM_FALLBACK_THRESHOLD = 0.2
DISAMBIGUATION_MARGIN = 0.15


def select_function_via_llm(
    functions: list[dict],
    query: str,
    ollama_url: str,
    ollama_model: str,
) -> list[dict]:
    """Use LLM to select the correct function(s) when signal matching fails."""
    # Build compact function list
    func_lines = []
    for i, f in enumerate(functions):
        name = f.get("name", "")
        desc = f.get("description", "")[:100]  # Shorter: 120→100
        func_lines.append(f"  {i}: {name} — {desc}")
    func_list_str = "\n".join(func_lines)

    prompt = f"""Select function(s) for: {query}

Functions:
{func_list_str}

Return JSON array of indices. [] if none matches. Pick by purpose, not word overlap."""

    try:
        with httpx.Client(timeout=60.0) as client:
            resp = client.post(
                f"{ollama_url}/api/chat",
                json={
                    "model": ollama_model,
                    "messages": [{"role": "user", "content": prompt}],
                    "stream": False,
                    "options": {"temperature": 0.001, "num_predict": 50},  # 100→50
                },
            )
            resp.raise_for_status()
            data = resp.json()
            content = data.get("message", {}).get("content", "").strip()

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
    """Use LLM to pick the correct function when signal scores are close."""
    func_lines = []
    for i, (f, s) in enumerate(candidates):
        name = f.get("name", "")
        desc = f.get("description", "")[:100]  # Shorter: 150→100
        func_lines.append(f"  {i}: {name} (score={s:.2f}) — {desc}")
    func_list_str = "\n".join(func_lines)

    prompt = f"""Pick the right function for: {query}

Candidates:
{func_list_str}

Return index. Match purpose, esp. action verbs: create/delete/search/status."""

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

        num_match = re.search(r"\b(\d+)\b", content)
        if num_match:
            idx = int(num_match.group(1))
            if 0 <= idx < len(candidates):
                return [candidates[idx][0]]

        logger.warning(
            f"LLM disambiguation parse failed: {content[:100]}, using top candidate"
        )
        return [candidates[0][0]]

    except Exception as e:
        logger.error(f"LLM disambiguation failed: {e}")
        return [candidates[0][0]]


# ---------------------------------------------------------------------------
# LLM irrelevance verification (v5) — KEPT, shortened prompt
# ---------------------------------------------------------------------------


def verify_relevance_via_llm(
    func: dict,
    query: str,
    ollama_url: str,
    ollama_model: str,
) -> bool:
    """Use LLM to verify if a function is truly relevant to the query (v5).

    Kept from v5 — critical for live_irrelevance (v4 got 42.5%).
    Only called when signal score == 0.0 (rare).
    """
    prompt = f"Function: {func['name']}\nDescription: {func.get('description','')[:150]}\nQuery: \"{query}\"\n\nIs this function relevant to the query? Answer RELEVANT or IRRELEVANT. Be conservative: lean IRRELEVANT."

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
        return False


# ---------------------------------------------------------------------------
# Parallel detection — OPTIMIZED: Reverted to v4 separator heuristic
# ---------------------------------------------------------------------------


def detect_parallel(query: str) -> bool:
    """Detect parallel intent using v4's separator heuristic (NO LLM call).

    v5 used an LLM-based detection that cost ~1 call per query but was only
    needed for parallel categories. v4's heuristic covers > 90% of cases.

    Returns True if the query likely requires multiple function calls.
    """
    query_lower = query.lower()

    # Separators that strongly indicate parallel intent
    strong_separators = [
        " and also ",
        " and then ",
        " plus ",
    ]
    if any(sep in query_lower for sep in strong_separators):
        return True

    # Medium-strength separators (check that both sides are meaningful)
    medium_separators = [" and "]
    for sep in medium_separators:
        parts = query_lower.split(sep)
        if len(parts) >= 2:
            # Check each part has at least 4 chars (meaningful query parts)
            if all(len(p.strip()) >= 4 for p in parts):
                return True

    # Native Chinese separators
    if "；" in query or "，" in query:
        parts = re.split(r"[；，]", query)
        if len(parts) >= 2 and all(len(p.strip()) >= 4 for p in parts):
            return True

    return False


# ---------------------------------------------------------------------------
# Parameter extraction — OPTIMIZED: Shorter prompt, fewer output tokens
# ---------------------------------------------------------------------------


def extract_all_params_via_llm(
    func: dict,
    query: str,
    ollama_url: str,
    ollama_model: str,
) -> list[dict]:
    """Use LLM to extract parameter values (array format, for parallel).

    Shortened prompt + reduced num_predict (512→192) for faster inference.
    """
    func_name = func.get("name", "")
    params = func.get("parameters", {})
    param_props = params.get("properties", {})
    required = params.get("required", [])

    param_lines = []
    for pname, pinfo in param_props.items():
        ptype = pinfo.get("type", "any")
        pdesc = pinfo.get("description", "")
        req = "req" if pname in required else "opt"
        enum_vals = pinfo.get("enum", None)
        enum_str = f" enum={enum_vals}" if enum_vals else ""
        param_lines.append(f"  - {pname} ({ptype}, {req}{enum_str}): {pdesc}")
    param_desc = "\n".join(param_lines) if param_lines else "  (none)"

    prompt = f"""Extract params for: {func_name}

Params:
{param_desc}

Query: {query}

Return JSON array of objects. One object per call. Use correct types. Omit missing params.

Examples: [{{"height":6.0,"weight":80}}] or [{{"a":1,"b":2}},{{"a":3,"b":4}}]"""

    try:
        with httpx.Client(timeout=120.0) as client:
            resp = client.post(
                f"{ollama_url}/api/chat",
                json={
                    "model": ollama_model,
                    "messages": [{"role": "system", "content": "Output a JSON array of objects."}, {"role": "user", "content": prompt}],
                    "stream": False,
                    "options": {
                        "temperature": 0.001,
                        "num_predict": 192,  # 512→192
                    },
                },
            )
            resp.raise_for_status()
            data = resp.json()
            content = data.get("message", {}).get("content", "")

        result = _parse_param_list(content)
        if result is not None:
            return result

        logger.warning(f"Failed to parse LLM param extraction: {content[:200]}")
        return [{}]

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

    Shortened prompt + reduced num_predict for faster inference.
    """
    func_name = func.get("name", "")
    params = func.get("parameters", {})
    param_props = params.get("properties", {})
    required = params.get("required", [])

    param_lines = []
    for pname, pinfo in param_props.items():
        ptype = pinfo.get("type", "any")
        pdesc = pinfo.get("description", "")
        req = "req" if pname in required else "opt"
        enum_vals = pinfo.get("enum", None)
        enum_str = f" enum={enum_vals}" if enum_vals else ""
        param_lines.append(f"  - {pname} ({ptype}, {req}{enum_str}): {pdesc}")
    param_desc = "\n".join(param_lines) if param_lines else "  (none)"

    prompt = f"""Extract params for "{func_name}".

Params:
{param_desc}

Query: {query}

Return JSON object with param names as keys. Use correct types (int/float/string/array/bool). Omit missing params. Use enum values exactly. Example: {{"x":10,"y":5}}"""

    try:
        with httpx.Client(timeout=120.0) as client:
            resp = client.post(
                f"{ollama_url}/api/chat",
                json={
                    "model": ollama_model,
                    "messages": [{"role": "user", "content": prompt}],
                    "stream": False,
                    "options": {
                        "temperature": 0.001,
                        "num_predict": 192,  # 512→192
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
    """Parse LLM response into list of param dicts."""
    content = content.strip()
    try:
        parsed = json.loads(content)
        if isinstance(parsed, list):
            return [d for d in parsed if isinstance(d, dict)]
        if isinstance(parsed, dict):
            return [parsed]
    except json.JSONDecodeError:
        pass
    arr_match = re.search(r"\[.*\]", content, re.DOTALL)
    if arr_match:
        try:
            parsed = json.loads(arr_match.group())
            if isinstance(parsed, list):
                return [d for d in parsed if isinstance(d, dict)]
        except json.JSONDecodeError:
            pass
    obj_match = re.search(r"\{[^}]+\}", content, re.DOTALL)
    if obj_match:
        try:
            parsed = json.loads(obj_match.group())
            if isinstance(parsed, dict):
                return [parsed]
        except json.JSONDecodeError:
            pass
    return None


def validate_and_coerce_params(func: dict, params: dict) -> dict:
    """Validate and coerce extracted parameters against function schema (v5)."""
    param_props = func.get("parameters", {}).get("properties", {})
    if not param_props:
        return params

    validated = {}
    for pname, pvalue in params.items():
        if pname not in param_props:
            continue

        pschema = param_props[pname]
        ptype = pschema.get("type", "string")
        enum_vals = pschema.get("enum", None)

        coerced = pvalue
        try:
            if ptype == "integer" and isinstance(pvalue, str):
                num_match = re.search(r"-?\d+", pvalue)
                if num_match:
                    coerced = int(num_match.group())
                else:
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
                try:
                    coerced = json.loads(pvalue)
                    if not isinstance(coerced, list):
                        coerced = [pvalue]
                except json.JSONDecodeError:
                    coerced = [item.strip() for item in pvalue.split(",")]
        except (ValueError, TypeError) as e:
            logger.warning(f"Type coercion failed for '{pname}': {e}")

        if enum_vals and coerced not in enum_vals:
            for ev in enum_vals:
                if isinstance(coerced, str) and isinstance(ev, str):
                    if coerced.lower() == ev.lower():
                        coerced = ev
                        break

        validated[pname] = coerced

    return validated


# ---------------------------------------------------------------------------
# Output formatting (deterministic)
# ---------------------------------------------------------------------------


def format_function_call(func_name: str, params: dict) -> str:
    """Format as BFCL expected output: [func_name(param1=value1, param2=value2)]."""
    if not params:
        return f"[{func_name}()]"
    parts = []
    for k, v in params.items():
        if isinstance(v, str):
            escaped = v.replace("\\", "\\\\").replace('"', '\\"')
            parts.append(f'{k}="{escaped}"')
        elif isinstance(v, bool):
            parts.append(f"{k}={v}")
        elif isinstance(v, (int, float)):
            parts.append(f"{k}={v}")
        elif isinstance(v, list):
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
    inner_parts = [p[1:-1] for p in parts]
    return f"[{', '.join(inner_parts)}]"


# ---------------------------------------------------------------------------
# Main routing pipeline — OPTIMIZED
# ---------------------------------------------------------------------------


def carm_route_bfcl(
    messages: list[dict],
    ollama_url: str,
    ollama_model: str,
) -> str:
    """Main CARM routing pipeline — OPTIMIZED version.

    Differences from v5:
      - Parallel detection uses v4 separator heuristic (no LLM call)
      - Shorter prompts for all LLM calls
      - Reduced num_predict for param extraction
    """
    functions = extract_functions_from_system_prompt(messages)
    if not functions:
        logger.warning("No functions found, falling back to LLM")
        result = call_ollama(messages, 0.001, ollama_url, ollama_model)
        return result["content"]

    query = extract_user_query(messages)
    if not query:
        return "[]"

    logger.info(f"Query: {query[:100]}...")
    logger.info(f"Functions: {[f['name'] for f in functions]}")

    # Signal-based scoring
    scored = [(f, score_function_relevance(f, query)) for f in functions]
    scored.sort(key=lambda x: x[1], reverse=True)

    best_score = scored[0][1] if scored else 0.0
    logger.info(f"Signal scores: {[(f['name'], f'{s:.2f}') for f, s in scored[:5]]}")

    effective_threshold = RELEVANCE_THRESHOLD
    if len(functions) == 1:
        effective_threshold = 0.15

    if best_score < effective_threshold:
        logger.info(f"Best score {best_score:.2f} < {effective_threshold} → LLM fallback")
        selected = select_function_via_llm(functions, query, ollama_url, ollama_model)
        if not selected:
            logger.info("LLM fallback found no match → []")
            return "[]"

        llm_scores = [score_function_relevance(f, query) for f in selected]
        max_llm_score = max(llm_scores) if llm_scores else 0.0

        if max_llm_score == 0.0:
            logger.info(f"LLM selected {[f['name'] for f in selected]} but signal=0 → relevance verification")
            relevant_selected = []
            for f in selected:
                if verify_relevance_via_llm(f, query, ollama_url, ollama_model):
                    relevant_selected.append(f)
                else:
                    logger.info(f"LLM rejected {f['name']}")

            if not relevant_selected:
                logger.info("All LLM-selected functions rejected → []")
                return "[]"
            selected = relevant_selected

        verified = [(f, 0.0) for f in selected]
        logger.info(f"LLM selected: {[f['name'] for f in selected]}")
    elif len(functions) == 1 and best_score < 0.4:
        logger.info(f"Single func, score {best_score:.2f} in [0.15, 0.4) → LLM verification")
        selected = select_function_via_llm(functions, query, ollama_url, ollama_model)
        if not selected:
            logger.info("LLM verification rejected → []")
            return "[]"
        verified = [(f, 0.0) for f in selected]
        logger.info(f"LLM verified: {[f['name'] for f in selected]}")
    elif len(functions) > 1 and best_score < 0.4:
        relevant = [(f, s) for f, s in scored if s >= effective_threshold]

        # [OPTIMIZED] Parallel detection: use v4 separator heuristic (no LLM)
        has_parallel_hint = detect_parallel(query)

        if has_parallel_hint and len(relevant) >= 2:
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
            logger.info(f"Top-2 close ({relevant[0][1]:.2f} vs {relevant[1][1]:.2f}) → LLM disambiguation")
            candidates = relevant[:3] if len(relevant) >= 3 else relevant
            selected = disambiguate_via_llm(candidates, query, ollama_url, ollama_model)
            verified = [(f, 0.0) for f in selected]
            logger.info(f"LLM disambiguated to: {[f['name'] for f in selected]}")
        else:
            verified = [relevant[0]]
    else:
        relevant = [(f, s) for f, s in scored if s >= effective_threshold]

        # [OPTIMIZED] Parallel detection: use v4 separator heuristic (no LLM)
        has_parallel_hint = detect_parallel(query)

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
        logger.info("No function selected → []")
        return "[]"

    logger.info(f"Verified: {[(f['name'], f'{s:.2f}') for f, s in verified]}")

    # [OPTIMIZED] Parallel detection: v4 heuristic (no LLM call)
    is_parallel = len(verified) > 1
    if not is_parallel and len(verified) == 1:
        # Use v4 separator heuristic instead of LLM
        is_parallel = detect_parallel(query)

    calls = []

    if not is_parallel:
        for func, score in verified:
            params = extract_params_via_llm_v2(func, query, ollama_url, ollama_model)
            params = validate_and_coerce_params(func, params)
            calls.append((func["name"], params))
            logger.info(f"  {func['name']} params: {params}")
    else:
        for func, score in verified:
            param_sets = extract_all_params_via_llm(func, query, ollama_url, ollama_model)
            for params in param_sets:
                params = validate_and_coerce_params(func, params)
                calls.append((func["name"], params))
                logger.info(f"  {func['name']} params: {params}")

    output = format_parallel_output(calls)
    logger.info(f"Output: {output}")
    return output


# ---------------------------------------------------------------------------
# Ollama client
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
    logger.info(f"call_ollama called with ollama_url={base}, model={model}")
    try:
        logger.info(f"Connecting to Ollama API: {base}/api/chat")
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
            logger.info(f"Ollama API response status: {resp.status_code}")
            resp.raise_for_status()
            data = resp.json()
            content = data.get("message", {}).get("content", "")
            logger.info(f"Ollama API call successful, content length={len(content)}")
        return {
            "content": content,
            "prompt_tokens": data.get("prompt_eval_count", 0),
            "completion_tokens": data.get("eval_count", 0),
        }
    except httpx.TimeoutException as e:
        logger.error(f"Ollama API timeout: {e}")
        return {"content": f"Error: Ollama API timeout: {str(e)}", "prompt_tokens": 0, "completion_tokens": 0}
    except Exception as e:
        logger.error(f"Ollama call failed: {e}")
        return {"content": f"Error: {e}", "prompt_tokens": 0, "completion_tokens": 0}


# ---------------------------------------------------------------------------
# HTTP handler
# ---------------------------------------------------------------------------


class CARMServerHandler(BaseHTTPRequestHandler):
    """OpenAI-compatible API handler with CARM routing (optimized)."""

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
        if self.path in ("/v1/models", "/models"):
            self._send_json(200, {"data": [{"id": "carm-router-opt", "object": "model"}]})
        elif self.path == "/health":
            self._send_json(200, {"status": "ok"})
        else:
            self._send_json(404, {"error": "not found"})

    def do_POST(self):
        if self.path in ("/v1/chat/completions", "/chat/completions"):
            self._handle_chat_completions()
        elif self.path in ("/v1/completions", "/completions"):
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

        logger.info(f"_handle_chat_completions called with {len(req.get('messages', []))} messages, {len(req.get('tools', []))} tools")

        messages = req.get("messages", [])
        tools = req.get("tools", [])
        if tools:
            func_defs = []
            for tool in tools:
                if tool.get("type") == "function":
                    func = tool.get("function", {})
                    if func:
                        func_defs.append(func)
            if func_defs:
                func_json = json.dumps(func_defs)
                system_msg = next((m for m in messages if m.get("role") == "system"), None)
                if system_msg:
                    system_msg["content"] = system_msg.get("content", "") + "\n" + func_json
                else:
                    messages.insert(0, {"role": "system", "content": func_json})
                logger.info(f"Injected {len(func_defs)} tools")

        logger.info(f"Calling carm_route_bfcl with self.ollama_url={self.ollama_url}, self.ollama_model={self.ollama_model}")
        start = time.time()
        try:
            content = carm_route_bfcl(messages, self.ollama_url, self.ollama_model)
            latency = time.time() - start
            logger.info(f"carm_route_bfcl completed in {latency:.2f}s, result length={len(content)}")
        except Exception as e:
            latency = time.time() - start
            logger.error(f"carm_route_bfcl failed after {latency:.2f}s: {e}")
            self._send_json(500, {"error": f"Internal error: {str(e)}"})
            return

        # Check if content looks like a function call output [func(...)]
        # If so, parse it and return as tool_calls for BFCL compatibility
        tool_calls = None
        if content.startswith("["):
            import re
            # Parse potential function calls: [func_name(param="value")] or [[...]]
            inner = content.strip("[]")
            func_pattern = r'(\w+)\((.*)\)'
            match = re.match(func_pattern, inner)
            if match:
                func_name = match.group(1)
                args_str = match.group(2)
                # Simple arg parsing: key="value" or key=value
                args = {}
                if args_str.strip():
                    for pair in args_str.split(","):
                        pair = pair.strip()
                        if "=" in pair:
                            k, v = pair.split("=", 1)
                            # Remove quotes from value
                            v = v.strip().strip('"').strip("'")
                            args[k] = v
                if func_name and args is not None:
                    tool_calls = [{
                        "id": "call_001",
                        "type": "function",
                        "function": {
                            "name": func_name,
                            "arguments": json.dumps(args),
                        }
                    }]

        message_content = {"role": "assistant", "content": None}
        if tool_calls:
            message_content["tool_calls"] = tool_calls
            finish_reason = "tool_calls"
        else:
            message_content["content"] = content
            finish_reason = "stop"

        response = {
            "id": f"chatcmpl-{int(time.time())}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": "carm-router-opt",
            "choices": [
                {
                    "index": 0,
                    "message": message_content,
                    "finish_reason": finish_reason,
                }
            ],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            "latency": latency,
        }
        self._send_json(200, response)

    def _handle_completions(self):
        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length).decode("utf-8")
        try:
            req = json.loads(body)
        except json.JSONDecodeError:
            self._send_json(400, {"error": "invalid JSON"})
            return

        prompt = req.get("prompt", "")
        messages = [{"role": "user", "content": prompt}]
        content = carm_route_bfcl(messages, self.ollama_url, self.ollama_model)

        response = {
            "id": f"cmpl-{int(time.time())}",
            "object": "text_completion",
            "created": int(time.time()),
            "model": "carm-router-opt",
            "choices": [{"index": 0, "text": content, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        }
        self._send_json(200, response)

    def log_message(self, format, *args):
        logger.info(f"{self.client_address[0]} - {format % args}")


def main():
    global OLLAMA_BASE_URL, OLLAMA_MODEL, RELEVANCE_THRESHOLD

    parser = argparse.ArgumentParser(
        description="CARM BFCL API Server (optimized — v4 parallel heuristic + shorter prompts)"
    )
    parser.add_argument("--port", type=int, default=11401, help="Server port")
    parser.add_argument("--host", default="0.0.0.0", help="Server host")
    parser.add_argument("--ollama-url", default=OLLAMA_BASE_URL)
    parser.add_argument("--ollama-model", default=OLLAMA_MODEL)
    parser.add_argument("--threshold", type=float, default=RELEVANCE_THRESHOLD)
    args = parser.parse_args()

    OLLAMA_BASE_URL = args.ollama_url
    OLLAMA_MODEL = args.ollama_model
    CARMServerHandler.ollama_url = OLLAMA_BASE_URL
    CARMServerHandler.ollama_model = OLLAMA_MODEL
    RELEVANCE_THRESHOLD = args.threshold

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    server = HTTPServer((args.host, args.port), CARMServerHandler)
    logger.info(f"CARM BFCL Server (optimized) starting on {args.host}:{args.port}")
    logger.info(f"  Ollama: {OLLAMA_BASE_URL} / {OLLAMA_MODEL}")
    logger.info(f"  Optimizations: v4 parallel heuristic, shorter LLM prompts, num_predict 192")
    logger.info(f"  Preserved: CARM signal routing + LLM irrelevance verification + LLM disambiguation")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("Server stopped")
        server.server_close()


if __name__ == "__main__":
    main()
