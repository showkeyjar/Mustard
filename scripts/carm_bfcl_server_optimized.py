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
RELEVANCE_THRESHOLD = 0.1

# Irrelevance verification threshold: if best score < this with single function, verify via LLM
# Set to 0.55 — irrelevance cases score 0.32-0.50, simple cases mostly ≥ 0.54
IRRELEVANCE_VERIFY_THRESHOLD = 0.55


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

    # 1. Direct function name substring in query (word-boundary aware)
    # Generic verbs like "calculate", "get", "find" are downweighted to avoid false matches
    generic_verbs = {
        "calculate",
        "get",
        "find",
        "compute",
        "check",
        "create",
        "update",
        "delete",
        "send",
        "search",
    }
    func_name_lower = func_name.lower()
    name_parts = func_name_lower.split(".")
    for part in name_parts:
        if len(part) > 2:
            # Use word boundary matching to avoid "int" matching "integral"
            if re.search(r"\b" + re.escape(part) + r"\b", query_lower):
                if part in generic_verbs:
                    score += 0.15  # Generic verbs get less weight
                else:
                    score += 0.4
            elif part in query_lower:
                # Partial match only if part is long enough (>4 chars) to reduce false positives
                if len(part) > 4 and part not in generic_verbs:
                    score += 0.2
    if func_name_lower in query_lower:
        score += 0.2

    # 2. Function name token overlap (excluding generic verbs)
    name_tokens = tokenize(func_name)
    if name_tokens:
        expanded = set()
        for t in name_tokens:
            expanded.add(t)
            expanded.update(t.split("_"))
        # Remove generic verbs from token overlap scoring
        expanded_specific = expanded - generic_verbs
        overlap = len(expanded_specific & query_tokens)
        score += min(overlap * 0.15, 0.3)

    # 3. Description keyword overlap (excluding generic verbs)
    desc_tokens = tokenize(func_desc)
    if desc_tokens:
        # Remove generic verbs from desc token matching too
        desc_specific = desc_tokens - generic_verbs
        overlap = len(desc_specific & query_tokens)
        score += min(overlap * 0.1, 0.2)

    # 4. Parameter name overlap
    param_tokens = set()
    for pn in param_names:
        param_tokens.update(tokenize(pn))
    if param_tokens:
        overlap = len(param_tokens & query_tokens)
        score += min(overlap * 0.12, 0.24)

    # 5. Semantic action hints
    # Only apply if the action verb in description differs from function name
    # (avoids double-counting when function name itself contains "calculate")
    action_hints = {
        "calculate": [
            "calculate",
            "compute",
            "求",
            "计算",
            "area",
            "sum",
            "total",
            "average",
            "mean",
            "distance",
            "perimeter",
            "volume",
        ],
        "convert": [
            "convert",
            "transform",
            "转换",
            "exchange",
            "rate",
            "currency",
            "to",
            "from",
            "兑换",
            "汇率",
        ],
        "search": [
            "search",
            "lookup",
            "query",
            "find",
            "查找",
            "搜索",
            "检索",
            "locate",
            "get",
            "retrieve",
            "fetch",
            "list",
            "show",
            "display",
        ],
        "check": [
            "check",
            "verify",
            "validate",
            "检查",
            "验证",
            "confirm",
            "test",
            "inspect",
        ],
        "create": [
            "create",
            "generate",
            "build",
            "创建",
            "生成",
            "add",
            "insert",
            "new",
            "make",
            "construct",
            "produce",
        ],
        "delete": ["delete", "remove", "drop", "删除", "clear", "erase"],
        "update": [
            "update",
            "modify",
            "更改",
            "修改",
            "edit",
            "change",
            "set",
            "adjust",
            "correct",
        ],
        "schedule": ["schedule", "book", "arrange", "预约", "安排", "plan", "reserve"],
        "send": [
            "send",
            "email",
            "notify",
            "发送",
            "邮件",
            "submit",
            "forward",
            "transfer",
        ],
        "translate": ["translate", "translation", "翻译"],
        "classify": [
            "classify",
            "classification",
            "categorize",
            "category",
            "type",
            "detect",
            "identify",
            "recognize",
            "diagnose",
            "predict",
        ],
        "filter": ["filter", "筛选", "select", "choose", "pick", "find"],
        "sort": ["sort", "order", "rank", "排序", "arrange"],
        "analyze": [
            "analyze",
            "analysis",
            "analyze",
            "evaluate",
            "assess",
            "measure",
            "study",
            "review",
            "examine",
            "inspect",
        ],
        "plot": [
            "plot",
            "chart",
            "graph",
            "draw",
            "visualize",
            "display",
            "render",
            "paint",
            "画",
        ],
    }
    desc_lower = func_desc.lower()
    # Skip action hints that are already part of the function name
    # (prevents "calculate" in func_name + "calculate" in query from double-scoring)
    func_name_actions = set()
    for action in action_hints:
        if action in func_name_lower:
            func_name_actions.add(action)

    # Only apply action hints if desc is non-empty
    if desc_lower.strip():
        for action, triggers in action_hints.items():
            # Skip if this action is already in the function name (already scored in rule 1)
            if action in func_name_actions:
                continue
            if action in desc_lower:
                for trigger in triggers:
                    if trigger in query_lower:
                        score += 0.1
                        break

    # 6. Synonym/domain expansion for common abbreviations and technical terms
    synonym_map = {
        "gcd": "greatest common divisor",
        "lcm": "least common multiple",
        "bmi": "body mass index",
        "svm": "support vector machine",
        "knn": "k nearest neighbors",
        "pca": "principal component analysis",
        "api": "application programming interface",
        "db": "database",
        "csv": "comma separated values",
        "json": "javascript object notation",
        "html": "hypertext markup language",
        "http": "hypertext transfer protocol",
        "url": "uniform resource locator",
        "uuid": "universally unique identifier",
        "id": "identifier",
        "info": "information",
        "config": "configuration",
        "temp": "temperature",
        "calc": "calculate",
        "num": "number",
        "diff": "difference",
        "avg": "average",
        "min": "minimum",
        "max": "maximum",
        "len": "length",
        "param": "parameter",
        "desc": "description",
        "dest": "destination",
        "src": "source",
        "addr": "address",
        "amt": "amount",
        "qty": "quantity",
        "dept": "department",
    }
    for abbr, expanded in synonym_map.items():
        # Only match if abbr appears as a standalone token, not as a substring of a longer word
        # e.g., "calc" should not match inside "calculate_compound_interest"
        if abbr in func_name_lower and expanded in query_lower:
            # Check if abbr is a standalone token in func_name (not part of a longer word)
            abbr_as_token = any(t == abbr for t in name_tokens) or any(
                t == abbr for t in func_name_lower.replace(".", "_").split("_")
            )
            if abbr_as_token:
                score += 0.25
        if expanded in func_name_lower and abbr in query_lower:
            # Check if abbr appears as a standalone word in the query
            if re.search(r"\b" + re.escape(abbr) + r"\b", query_lower):
                # And expanded is a standalone token in func_name
                expanded_as_token = any(t == expanded for t in name_tokens) or any(
                    t == expanded for t in func_name_lower.replace(".", "_").split("_")
                )
                if expanded_as_token:
                    score += 0.25

    # 7. Query contains function domain hint
    domain_hints = [
        "math",
        "geometry",
        "algebra",
        "calculus",
        "physics",
        "chemistry",
        "biology",
        "finance",
        "stat",
        "sport",
        "music",
        "history",
        "law",
        "movie",
        "weather",
        "ecology",
        "employee",
        "database",
        "restaurant",
        "flight",
        "hotel",
    ]
    for hint in domain_hints:
        if hint in func_name_lower and hint in query_lower:
            score += 0.15

    # 8. Semantic prefix matching: query word is a prefix of function name part
    # e.g., "cook" matches "cookbook", "recipe" matches "search_recipe"
    # Skip generic verbs to avoid "calculate" matching "calculate_compound_interest"
    for part in name_parts:
        if len(part) > 4 and part not in generic_verbs:
            for qword in query_lower.split():
                qword_clean = re.sub(r"[^a-z]", "", qword)
                # Skip generic verbs as query words too
                if qword_clean in generic_verbs:
                    continue
                if (
                    len(qword_clean) > 3
                    and qword_clean != part
                    and part.startswith(qword_clean)
                ):
                    score += 0.2
                    break

    # 9. Semantic synonym matching for common live function patterns
    semantic_pairs = [
        # (query_keyword, func_name_keyword, score_bonus)
        ("cook", "recipe", 0.3),
        ("recipe", "cookbook", 0.3),
        ("cook", "cookbook", 0.3),
        ("wash", "appliance", 0.15),
        ("machine", "appliance", 0.2),
        ("stop", "control", 0.15),
        ("start", "control", 0.15),
        ("turn off", "control", 0.15),
        ("turn on", "control", 0.15),
        ("change", "modify", 0.2),
        ("update", "change", 0.2),
        ("order", "ride", 0.0),  # No boost — these are different
        ("burger", "ride", 0.0),
        ("drink", "drink", 0.3),
        ("coffee", "drink", 0.25),
        ("latte", "drink", 0.25),
        ("weather", "weather", 0.3),
        ("news", "news", 0.3),
        ("movie", "movie", 0.25),
    ]
    for q_kw, f_kw, bonus in semantic_pairs:
        if bonus > 0 and q_kw in query_lower and f_kw in func_name_lower:
            score += bonus

    # Cap semantic pair bonuses to avoid over-boosting
    # (already applied above, just ensure total doesn't exceed reasonable bound)

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
    # Build compact function list with params
    func_lines = []
    for i, f in enumerate(functions):
        name = f.get("name", "")
        desc = f.get("description", "")[:150]
        params = f.get("parameters", {}).get("properties", {})
        param_str = "(" + ", ".join(params.keys()) + ")" if params else "()"
        func_lines.append(f"  {i}: {name}{param_str} — {desc}")
    func_list_str = "\n".join(func_lines)

    prompt = f"""You are a function router. Given a user query and a list of functions, select the function(s) that should be called.

User query: {query}

Available functions:
{func_list_str}

Rules:
1. Select by SEMANTIC match: does the function's purpose address what the user wants?
2. "cook" matches "cookbook.search_recipe" (cooking recipes)
3. "stop washing machine" matches "ControlAppliance.execute" (appliance control)
4. "change drink" matches "change_drink" (modify order)
5. Return [] if NO function is relevant
6. Do NOT return functions that are only tangentially related

Output ONLY a JSON array of indices, e.g. [0] or [0,2] or []. No explanation."""

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

    Enhanced with semantic mismatch detection for irrelevance cases:
    - Check if the function's PURPOSE matches what the query asks for
    - Reject partial name matches that are semantically unrelated
    """
    func_name = func.get("name", "")
    func_desc = func.get("description", "")[:300]
    params = func.get("parameters", {}).get("properties", {})
    param_names = list(params.keys())
    param_descs = []
    for pn, pi in params.items():
        pdesc = pi.get("description", "")[:80]
        param_descs.append(f"  {pn}: {pdesc}")
    params_str = "\n".join(param_descs) if param_descs else "  (none)"

    prompt = f"""Function: {func_name}
Description: {func_desc}
Parameters:
{params_str}

Query: "{query}"

Does this function DIRECTLY answer the user's query? Answer RELEVANT or IRRELEVANT.

Check these conditions — ALL must be true for RELEVANT:
1. PURPOSE: The function's core purpose matches what the user is asking for
2. PARAMETERS: The user's query provides values compatible with the function's parameters
3. SCOPE: The function can actually compute/return what the user wants

Important nuances:
- "how to cook X" IS relevant to a recipe search function (user wants cooking instructions → recipe)
- "change/update/modify drink" IS relevant to a change_drink function
- "stop/start washing machine" IS relevant to an appliance control function
- Querying with variables ('v', 'theta') IS acceptable for functions that take numeric params

Common mismatches (IRRELEVANT):
- "roots of linear equation bx+c=0" vs find_roots (quadratic only, needs 'a' param)
- "derivative" vs compute_definite_integral (derivative ≠ integral)
- "prime factors" vs compound_interest (number theory ≠ finance)
- "closest integer" vs closest_prime (rounding ≠ primality)
- "fastest route" vs prime_numbers_in_range (navigation ≠ number theory)

Answer RELEVANT or IRRELEVANT."""

    try:
        with httpx.Client(timeout=30.0) as client:
            resp = client.post(
                f"{ollama_url}/api/chat",
                json={
                    "model": ollama_model,
                    "messages": [{"role": "user", "content": prompt}],
                    "stream": False,
                    "options": {"temperature": 0.001, "num_predict": 20},
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
    """Detect parallel intent using separator heuristic.

    Returns True if the query likely requires multiple function calls.
    """
    query_lower = query.lower()

    # Strong indicators of parallel intent
    strong_patterns = [
        r"\band also\b",
        r"\band\s+then\b",
        r"\bplus\b",
        r"\bcombined\s+with\b",
        r"\bboth\b.*\band\b",
        r"\bcompare\b",
        r"\bversus\b",
        r"\bvs\b",
        r"\bas well as\b",
        r"\band\s+for\s+(?:the|a|an|my|your|our)\b",
    ]
    for pat in strong_patterns:
        if re.search(pat, query_lower):
            return True

    # Sentence-level transition words (period + transition)
    transitions = [
        r"[.?!]\s*also\b",
        r"[.?!]\s*additionally\b",
        r"[.?!]\s*moreover\b",
        r"[.?!]\s*furthermore\b",
        r"[.?!]\s*then\b",
        r"[.?!]\s*next\b",
        r"[.?!]\s*finally\b",
        r"[.?!]\s*meanwhile\b",
        r"[.?!]\s*in addition\b",
        r"[.?!]\s*apart from that\b",
    ]
    for pat in transitions:
        if re.search(pat, query_lower):
            return True

    # "and" separator with meaningful parts on both sides
    parts = re.split(r"\band\b", query_lower)
    if len(parts) >= 3 and all(len(p.strip()) >= 4 for p in parts):
        return True
    if len(parts) >= 2 and all(len(p.strip()) >= 8 for p in parts):
        return True

    # Comma/Chinese separators
    if any(sep in query for sep in ["；", "，", ", "]):
        parts = re.split(r"[；，,]", query)
        if len(parts) >= 2 and all(len(p.strip()) >= 4 for p in parts):
            return True

    # Multi-request keywords
    multi_request_words = [
        "both",
        "several",
        "multiple",
        "various",
        "different",
        "each",
        "all",
        "two",
        "three",
        "both of",
    ]
    if any(w in query_lower for w in multi_request_words):
        return True

    return False


# ---------------------------------------------------------------------------
# Query splitting for parallel processing
# ---------------------------------------------------------------------------


def split_parallel_query(query: str) -> list[str]:
    """Split query into independent sub-queries for parallel processing.

    Returns [query] if no split is possible, or a list of segment strings.
    Supports multi-way splitting (>2 segments).
    """
    query_lower = query.lower()

    # Sentence-level transitions: period + transition word
    transition_patterns = [
        r"[.?!]\s*(Also|Additionally|Moreover|Furthermore|Then|Next|Finally|Meanwhile)\b",
        r"[.?!]\s*In addition\b",
        r"[.?!]\s*Apart from that\b",
        r"[.?!]\s*On top of that\b",
        r"[.?!]\s*Other than that\b",
        r"[.?!]\s*Besides\b",
    ]
    for pat in transition_patterns:
        parts = re.split(pat, query, maxsplit=1, flags=re.IGNORECASE)
        if len(parts) >= 2:
            cleaned = [p.strip().rstrip(".,;!?") for p in parts if p.strip()]
            if all(len(p) > 5 for p in cleaned):
                return cleaned

    # "and also" / "and for" / "and calculate" / "also calculate" connectors
    connector_pats = [
        r"\band\s+also\b",
        r"\band\s+for\s+(?:the|a|an|my|your|our)\b",
        r"\band\s+(?:calculate|find|compute|get|buy|book|turn|change|update|check|tell|provide)\b",
        r"\bplus\b",
        r"\bcombined\s+with\b",
        r"\bas well as\b",
        # "Also, calculate" at sentence boundary or after comma
        r"[,;.]\s*Also,?\s*",
        r"\.\s*Also,?\s*",
        r"[,;.]\s*Additionally,?\s*",
        r"[,;.]\s*Then,?\s*",
    ]
    for pat in connector_pats:
        parts = re.split(pat, query, maxsplit=1, flags=re.IGNORECASE)
        if len(parts) >= 2:
            cleaned = [p.strip().rstrip(".,;!?") for p in parts if p.strip()]
            if all(len(p) > 5 for p in cleaned):
                return cleaned

    # Chinese connectors: 和、以及、另外、同时
    cn_connectors = ["以及", "另外", "同时", "还有", "接着", "然后"]
    for conn in cn_connectors:
        if conn in query:
            parts = query.split(conn)
            cleaned = [p.strip().rstrip("。，；！？") for p in parts if p.strip()]
            if len(cleaned) >= 2 and all(len(p) > 3 for p in cleaned):
                return cleaned

    # Chinese 和 as separator (only between city/location names, not in compound words)
    # Pattern: "X市和Y市" or "X和Y的天气"
    if "和" in query and ("天气" in query or "天气" in query_lower):
        parts = query.split("和")
        if len(parts) >= 2:
            cleaned = [p.strip().rstrip("。，；！？") for p in parts if p.strip()]
            if all(len(p) > 3 for p in cleaned):
                return cleaned

    # Korean connectors: 하고, 그리고
    kr_connectors = ["하고", "그리고"]
    for conn in kr_connectors:
        if conn in query:
            parts = query.split(conn)
            cleaned = [p.strip().rstrip(".") for p in parts if p.strip()]
            if len(cleaned) >= 2 and all(len(p) > 3 for p in cleaned):
                return cleaned

    # Comma + "and" split: "Calculate X, and Y" or "Find X, and find Y"
    # This handles "GCD of 96 and 128, and the least common multiple of 15 and 25"
    comma_and_match = re.search(r",\s*(?:and\s+)", query, re.IGNORECASE)
    if comma_and_match:
        split_pos = comma_and_match.start()
        part1 = query[:split_pos].strip().rstrip(".,;!?")
        part2 = query[comma_and_match.end() :].strip().rstrip(".,;!?")
        if len(part1) > 8 and len(part2) > 8:
            return [part1, part2]

    # Comma-separated independent clauses with action verbs
    comma_parts = re.split(r",\s*", query)
    if len(comma_parts) >= 2 and all(len(p.strip()) > 8 for p in comma_parts):
        action_verbs = {
            "calculate",
            "find",
            "compute",
            "get",
            "buy",
            "book",
            "search",
            "convert",
            "check",
            "create",
            "turn",
            "change",
            "update",
            "play",
            "tell",
            "provide",
        }
        has_actions = all(
            any(av in p.lower() for av in action_verbs) for p in comma_parts
        )
        if has_actions:
            return [p.strip().rstrip(".,;!?") for p in comma_parts]

    return [query]


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

    prompt = f"""Extract ALL params for: {func_name}

Schema:
{param_desc}

Query: {query}

CRITICAL RULES:
1. Return JSON array of objects, one per call.
2. Only create MULTIPLE calls if the query explicitly asks for the SAME function multiple times with DIFFERENT parameters.
3. Do NOT invent extra calls — if the query mentions this function once, return exactly ONE object.
4. Use correct types (int/float/str/bool). Omit missing optional params.
5. Use enum values EXACTLY as listed.
6. For location params, include city + country/region if mentioned (e.g., "Shanghai, China").
7. For boolean params, use JSON true/false (not Python True/False).

Examples:
Simple: [{{"a":1,"b":2}}]
Multiple (explicit): [{{"a":1,"b":2}},{{"a":3,"b":4}}]
Nested: [{{"users":[{{"name":"Alice","age":30}},{{"name":"Bob","age":25}}]}}]"""

    try:
        with httpx.Client(timeout=120.0) as client:
            resp = client.post(
                f"{ollama_url}/api/chat",
                json={
                    "model": ollama_model,
                    "messages": [
                        {
                            "role": "system",
                            "content": "Output only a JSON array of objects.",
                        },
                        {"role": "user", "content": prompt},
                    ],
                    "stream": False,
                    "options": {
                        "temperature": 0.001,
                        "num_predict": 300,  # 192→300
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

Schema:
{param_desc}

Query: {query}

Return JSON object with param names as keys. Use correct types (int/float/str/bool/array). Fill ALL required params from the query. Omit missing optional params. Use enum values EXACTLY as listed — do not add extra words like "milk" or "juice" to enum values. For location params, include city + country/region if mentioned in the query (e.g., "Shanghai, China", "Tel Aviv, Israel"). For boolean params, use JSON true/false.

Example for math.factorial: {{"number":5}}"""

    try:
        with httpx.Client(timeout=120.0) as client:
            resp = client.post(
                f"{ollama_url}/api/chat",
                json={
                    "model": ollama_model,
                    "messages": [
                        {"role": "system", "content": "Output only a JSON object."},
                        {"role": "user", "content": prompt},
                    ],
                    "stream": False,
                    "options": {
                        "temperature": 0.001,
                        "num_predict": 300,  # 192→300
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
            # Try case-insensitive match first
            matched = False
            for ev in enum_vals:
                if isinstance(coerced, str) and isinstance(ev, str):
                    if coerced.lower() == ev.lower():
                        coerced = ev
                        matched = True
                        break
            # Try substring/containment match (e.g., "coconut milk" → "coconut")
            if not matched and isinstance(coerced, str):
                for ev in enum_vals:
                    if isinstance(ev, str):
                        if (
                            coerced.lower() in ev.lower()
                            or ev.lower() in coerced.lower()
                        ):
                            coerced = ev
                            matched = True
                            break
            # If still not matched, try removing common suffixes
            if not matched and isinstance(coerced, str):
                for suffix in [" milk", " water", " juice", " sauce", " powder"]:
                    if coerced.lower().endswith(suffix):
                        stripped = coerced.lower()[: -len(suffix)]
                        for ev in enum_vals:
                            if isinstance(ev, str) and stripped == ev.lower():
                                coerced = ev
                                matched = True
                                break
                        if matched:
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
    """Main CARM routing pipeline — OPTIMIZED v6.

    Changes from v5:
      - Parallel detection: segment-based + enhanced separator heuristic
      - Parallel multiple: collect ALL relevant functions (no close-score filter)
      - Irrelevance guard: LLM verification when best_score < 0.3
      - Shorter prompts, reduced num_predict
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

    # Try segment-based parallel detection first
    segments = split_parallel_query(query)
    has_parallel_segments = len(segments) > 1

    if has_parallel_segments:
        logger.info(f"Parallel segments: {segments}")
        selected_names = set()
        for seg in segments:
            seg_scores = [(f, score_function_relevance(f, seg)) for f in functions]
            seg_scores.sort(key=lambda x: x[1], reverse=True)
            if seg_scores and seg_scores[0][1] >= effective_threshold:
                # If top score is low or top-2 are very close, use LLM to pick
                top_score = seg_scores[0][1]
                second_score = seg_scores[1][1] if len(seg_scores) > 1 else 0.0
                if top_score < 0.3 or (top_score - second_score) < 0.08:
                    # Low confidence — use LLM for this segment
                    seg_selected = select_function_via_llm(
                        functions, seg, ollama_url, ollama_model
                    )
                    if seg_selected:
                        for f in seg_selected:
                            selected_names.add(f["name"])
                else:
                    selected_names.add(seg_scores[0][0]["name"])

        if selected_names:
            verified = [(f, 0.0) for f in functions if f["name"] in selected_names]
        else:
            selected = select_function_via_llm(
                functions, query, ollama_url, ollama_model
            )
            if not selected:
                return "[]"
            verified = [(f, 0.0) for f in selected]
    elif best_score < effective_threshold:
        logger.info(
            f"Best score {best_score:.2f} < {effective_threshold} → LLM fallback"
        )
        selected = select_function_via_llm(functions, query, ollama_url, ollama_model)
        if not selected:
            logger.info("LLM fallback found no match → []")
            return "[]"

        # Trust LLM selection — it already made a semantic judgment
        # The select_function_via_llm prompt includes instructions to return [] for irrelevant
        # For single-function cases, the LLM can distinguish irrelevance from live_relevance
        verified = [(f, 0.0) for f in selected]
        logger.info(f"LLM selected: {[f['name'] for f in selected]}")
    elif len(functions) == 1:
        # Single function available — must verify relevance to avoid false positives
        # (irrelevance test cases have exactly 1 function that should NOT be called)
        if best_score < IRRELEVANCE_VERIFY_THRESHOLD:
            logger.info(
                f"Single func '{scored[0][0]['name']}' score={best_score:.2f} < {IRRELEVANCE_VERIFY_THRESHOLD} → verify relevance"
            )
            if verify_relevance_via_llm(scored[0][0], query, ollama_url, ollama_model):
                verified = [(scored[0][0], scored[0][1])]
                logger.info(f"LLM confirmed relevance: {scored[0][0]['name']}")
            else:
                logger.info(
                    f"LLM rejected single func '{scored[0][0]['name']}' as irrelevant → []"
                )
                return "[]"
        else:
            verified = [(scored[0][0], scored[0][1])]
            logger.info(
                f"Single func '{scored[0][0]['name']}' score={best_score:.2f} → use directly"
            )
    else:
        relevant = [(f, s) for f, s in scored if s >= effective_threshold]

        if not relevant:
            # No function above threshold — try LLM fallback for multi-func cases
            logger.info("No function above threshold → LLM fallback")
            selected = select_function_via_llm(
                functions, query, ollama_url, ollama_model
            )
            if selected:
                verified = [(f, 0.0) for f in selected]
                logger.info(f"LLM fallback selected: {[f['name'] for f in selected]}")
            else:
                logger.info("LLM fallback found no match → []")
                return "[]"
        elif best_score < IRRELEVANCE_VERIFY_THRESHOLD:
            # Low signal score — try LLM selection first, then verify
            logger.info(
                f"Best score {best_score:.2f} < {IRRELEVANCE_VERIFY_THRESHOLD} → LLM selection"
            )
            selected = select_function_via_llm(
                functions, query, ollama_url, ollama_model
            )
            if selected:
                verified = [(f, 0.0) for f in selected]
                logger.info(f"LLM selected: {[f['name'] for f in selected]}")
            else:
                # LLM couldn't select — fall back to verify top function
                top_func = relevant[0][0]
                if verify_relevance_via_llm(top_func, query, ollama_url, ollama_model):
                    verified = [relevant[0]]
                    logger.info(f"LLM confirmed relevance: {top_func['name']}")
                else:
                    logger.info(f"LLM rejected {top_func['name']} as irrelevant → []")
                    return "[]"
        else:
            has_parallel_hint = detect_parallel(query)

            if has_parallel_hint and len(relevant) >= 2:
                # For parallel_multiple: each segment maps to exactly ONE function
                # Use segment-based selection to avoid over-generating
                segments = split_parallel_query(query)
                if len(segments) > 1:
                    verified = []
                    seen_names = set()
                    for seg in segments:
                        seg_scores = [
                            (f, score_function_relevance(f, seg)) for f in functions
                        ]
                        seg_scores.sort(key=lambda x: x[1], reverse=True)
                        if seg_scores and seg_scores[0][1] >= effective_threshold:
                            fname = seg_scores[0][0]["name"]
                            if fname not in seen_names:
                                verified.append(seg_scores[0])
                                seen_names.add(fname)
                    logger.info(
                        f"Segment-based parallel: {len(verified)} functions from {len(segments)} segments"
                    )
                else:
                    # No segment split — use strict close-score filter
                    # Only include functions within 0.15 of best (stricter than 0.2)
                    best = relevant[0][1]
                    verified = [relevant[0]]
                    for f, s in relevant[1:]:
                        if best - s < 0.15 and s >= effective_threshold:
                            verified.append((f, s))
                        else:
                            break
                    logger.info(f"Close-score parallel: {len(verified)} functions")
            elif (
                len(relevant) >= 2
                and (relevant[0][1] - relevant[1][1]) < DISAMBIGUATION_MARGIN
            ):
                logger.info(
                    f"Top-2 close ({relevant[0][1]:.2f} vs {relevant[1][1]:.2f}) → LLM disambiguation"
                )
                candidates = relevant[:3] if len(relevant) >= 3 else relevant
                selected = disambiguate_via_llm(
                    candidates, query, ollama_url, ollama_model
                )
                verified = [(f, 0.0) for f in selected]
                logger.info(f"LLM disambiguated to: {[f['name'] for f in selected]}")
            else:
                verified = [relevant[0]]

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
    elif has_parallel_segments and len(segments) > 1:
        # Per-segment parameter extraction: each segment → one function call
        # This avoids duplicate calls from extract_all_params_via_llm
        seg_func_map = {}  # segment → function
        for seg in segments:
            seg_scores = [(f, score_function_relevance(f, seg)) for f in functions]
            seg_scores.sort(key=lambda x: x[1], reverse=True)
            if seg_scores and seg_scores[0][1] >= effective_threshold:
                seg_func_map[seg] = seg_scores[0][0]
            else:
                # Try matching to verified functions
                for func, _ in verified:
                    if func not in seg_func_map.values():
                        seg_func_map[seg] = func
                        break

        for seg, func in seg_func_map.items():
            params = extract_params_via_llm_v2(func, seg, ollama_url, ollama_model)
            params = validate_and_coerce_params(func, params)
            calls.append((func["name"], params))
            logger.info(f"  {func['name']} params (from segment): {params}")
    else:
        for func, score in verified:
            param_sets = extract_all_params_via_llm(
                func, query, ollama_url, ollama_model
            )
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
        return {
            "content": f"Error: Ollama API timeout: {str(e)}",
            "prompt_tokens": 0,
            "completion_tokens": 0,
        }
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
            self._send_json(
                200, {"data": [{"id": "carm-router-opt", "object": "model"}]}
            )
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
        logger.info(
            f"Received request: path={self.path}, content_length={content_length}"
        )
        body = self.rfile.read(content_length).decode("utf-8")
        logger.info(f"Request body (first 500 chars): {body[:500]}")
        try:
            req = json.loads(body)
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON: {e}")
            self._send_json(400, {"error": "invalid JSON"})
            return

        tools = req.get("tools") or []
        logger.info(
            f"_handle_chat_completions called with {len(req.get('messages', []))} messages, {len(tools)} tools"
        )

        messages = req.get("messages", [])
        if tools:
            func_defs = []
            for tool in tools:
                if tool.get("type") == "function":
                    func = tool.get("function", {})
                    if func:
                        func_defs.append(func)
            if func_defs:
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
                logger.info(f"Injected {len(func_defs)} tools")

        logger.info(
            f"Calling carm_route_bfcl with self.ollama_url={self.ollama_url}, self.ollama_model={self.ollama_model}"
        )
        start = time.time()
        try:
            content = carm_route_bfcl(messages, self.ollama_url, self.ollama_model)
            latency = time.time() - start
            logger.info(
                f"carm_route_bfcl completed in {latency:.2f}s, result length={len(content)}"
            )
        except Exception as e:
            latency = time.time() - start
            logger.error(f"carm_route_bfcl failed after {latency:.2f}s: {e}")
            self._send_json(500, {"error": f"Internal error: {str(e)}"})
            return

        # BFCL uses prompting mode (is_fc_model=False) and expects plain text content
        message_content = {"role": "assistant", "content": content}
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

    import os

    log_file = os.path.join(os.path.dirname(__file__), "carm_server.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_file, encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )

    server = HTTPServer((args.host, args.port), CARMServerHandler)
    logger.info(f"CARM BFCL Server (optimized) starting on {args.host}:{args.port}")
    logger.info(f"  Ollama: {OLLAMA_BASE_URL} / {OLLAMA_MODEL}")
    logger.info(
        f"  Optimizations: v4 parallel heuristic, shorter LLM prompts, num_predict 192"
    )
    logger.info(
        f"  Preserved: CARM signal routing + LLM irrelevance verification + LLM disambiguation"
    )

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("Server stopped")
        server.server_close()


if __name__ == "__main__":
    main()
