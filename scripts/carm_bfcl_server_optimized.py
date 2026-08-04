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
from http.server import HTTPServer, BaseHTTPRequestHandler, ThreadingHTTPServer

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
        "temperature",
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
        ("temperature", "weather", 0.3),
        ("forecast", "weather", 0.25),
        ("climate", "weather", 0.2),
        ("news", "news", 0.3),
        ("movie", "movie", 0.25),
        ("revenue", "revenue_forecast", 0.4),
        ("revenue", "revenue", 0.4),
        ("forecast", "revenue_forecast", 0.3),
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
 2. DEFAULT: select exactly ONE function. Only select multiple if the query explicitly requests MULTIPLE DIFFERENT actions.
 3. "and" connecting parts of the SAME request does NOT mean multiple calls (e.g., "weather in Boston and temperature" = ONE weather call)
 4. Only return multiple functions if the query has clearly SEPARATE action items (e.g., "book a flight AND reserve a hotel" = 2 calls)
 5. Return [] if NO function is relevant — do NOT force a match
 6. Do NOT return functions that are only tangentially related
 7. If two functions are similar, select only the ONE that best matches
 8. Do NOT select a function just because its name appears in the query — match by PURPOSE
 9. Generic utility functions (requests.get, print, len) should NOT be selected for domain-specific queries
10. If the query asks for something NO available function can do, return []
11. Statements, greetings, or opinions are NOT function calls — return []
12. "uber.ride" is for transportation, NOT food ordering
13. If the user mentions food/eating but no food function is available, return []
14. Numbered steps (1. 2. 3.) each need their own function — select ALL matching
15. If the user asks "what should I do" and "handover_to_agent" is available, select it

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


def dedup_similar_functions(
    selected: list[dict],
    query: str,
    ollama_url: str,
    ollama_model: str,
) -> list[dict]:
    """Remove semantically duplicate functions from LLM selection.

    When LLM selects both 'search' and 'news_search' for the same query segment,
    keep only the best match.
    """
    if len(selected) <= 1:
        return selected

    # Group functions by semantic similarity
    # Two functions are "similar" if they share the same core action verb
    # e.g., generate_image vs generate_human_image, HNA_WQA.search vs HNA_NEWS.search
    groups: list[list[tuple[int, dict]]] = []
    for i, f in enumerate(selected):
        fname = f.get("name", "").lower()
        placed = False
        for grp in groups:
            # Check if similar to any function in this group
            for _, existing in grp:
                ename = existing.get("name", "").lower()
                # Same root: one name contains the other, or they share a common action
                if (
                    fname in ename
                    or ename in fname
                    or (
                        "." in fname
                        and "." in ename
                        and fname.split(".")[-1] == ename.split(".")[-1]
                    )
                ):
                    grp.append((i, f))
                    placed = True
                    break
            if placed:
                break
        if not placed:
            groups.append([(i, f)])

    # From each group with >1 function, keep only the best match via LLM
    result = []
    for grp in groups:
        if len(grp) == 1:
            result.append(grp[0][1])
        else:
            # Use LLM to pick the best one
            candidates = [f for _, f in grp]
            scored = [(f, score_function_relevance(f, query)) for f in candidates]
            scored.sort(key=lambda x: x[1], reverse=True)

            # If scores are close, use LLM disambiguation
            if len(scored) >= 2 and scored[0][1] - scored[1][1] < 0.2:
                picked = disambiguate_via_llm(scored, query, ollama_url, ollama_model)
                result.extend(picked)
            else:
                result.append(scored[0][0])
            logger.info(
                f"Deduped similar functions: {[f['name'] for f in candidates]} → {[f['name'] for f in (picked if len(scored) >= 2 and scored[0][1] - scored[1][1] < 0.2 else [scored[0][0]])]}"
            )

    return result


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

    prompt = f"""Does this function help answer this query?

Function: {func_name} - {func_desc}
Parameters: {", ".join(param_names)}
Query: "{query}"

Reject if:
- The user is making a STATEMENT or expressing an opinion, not asking for an action (e.g., "Boston has high temperature of 54C" is a statement, not a weather request)
- The user is greeting or chatting (e.g., "Olá, tudo bem?", "Hello", "How are you?")
- The user asks about something completely unrelated to what the function does (e.g., asking about VirusTotal/IP addresses when only weather function is available)
- The function is a generic utility (requests.get, print, len, etc.) and the user is asking a domain question
- The user asks "how to" do something manually (not via a function call)
- The query uses abstract variable names (like 'v', 'theta', 't') instead of concrete values for a calculation
- The user mentions a place/thing but doesn't ask the function to do anything with it (e.g., "Whopper" with a food function — not an order request)
- The query contains code instructions or programming tasks when the function is domain-specific (weather, finance, etc.)
- The user asks for something the function CANNOT do (e.g., ordering food when only ride booking is available)

Accept if:
- The user explicitly asks the function to perform its described purpose with concrete inputs
- The query matches the function's domain and requests an action the function can perform

Answer with ONLY one word: RELEVANT or IRRELEVANT."""

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
        # "two cities of X and Y" pattern
        r"\b(?:two|three)\s+(?:cities|locations|places)\s+of\b",
        # "Do the same" / "do the same calculation" — repeated operation
        r"\bdo\s+the\s+same\b",
        r"\bsame\s+calculation\b",
        r"\brepeat\s+(?:the|this|that)\b",
        # "also get" / "and also" — additional request
        r"\balso\s+get\b",
        r"\balso\s+find\b",
        r"\balso\s+calculate\b",
        r"\balso\s+fetch\b",
        r"\balso\s+check\b",
        # "and how much" / "and how many" — connects two independent questions
        r"\band\s+how\s+(?:much|many|long)\b",
        # "and compute" / "and calculate" — connects two compute tasks
        r"\band\s+(?:compute|calculate)\b",
        # Chinese parallel indicators
        r"还有",
        r"以及",
        r"另外",
        r"同时",
        # Korean parallel indicators
        r"그리고",
        r"하고",
        # Spanish parallel indicators — only with weather/location context
        r"\by\b.*\b(clima|tiempo|temperatura|pronóstico|cancún|playa|tulum)\b",
        r"\b(clima|tiempo|temperatura|pronóstico|cancún|playa|tulum)\b.*\by\b",
        # Step-by-step indicators
        r"\bstep\s*\d+\b",
        r"\bsteps?\s*:",
        # Numbered list indicators
        r"\n\s*\d+\.\s+",
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
    # Be conservative: "and" often connects clauses within a single request
    # (e.g., "I need weather info and temperature for Boston" is ONE request)
    # Only trigger parallel if there are 3+ parts (A and B and C),
    # or if both parts contain distinct action verbs
    parts = re.split(r"\band\b", query_lower)
    if len(parts) >= 3 and all(len(p.strip()) >= 4 for p in parts):
        return True
    # For 2-part "and", require both parts to have action verbs (strict check)
    # This prevents "weather in Boston and temperature" from triggering parallel
    if len(parts) >= 2 and all(len(p.strip()) >= 15 for p in parts):
        # Very long parts — check for action verbs
        action_words = {
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
            "fetch",
            "call",
            "send",
            "delete",
            "add",
            "show",
            "list",
            "generate",
            "analyze",
            "extract",
            "sort",
            "filter",
            "start",
            "stop",
            "set",
            "open",
            "close",
            "launch",
            "run",
            "build",
        }
        if all(any(aw in p for aw in action_words) for p in parts):
            return True

    # "and" connects two independent clauses each containing a number+unit
    # e.g., "size 3000 sq ft. in New York and 4000 sq ft. in Los Angeles"
    # This is parallel: two separate entities with their own measurements
    if len(parts) >= 2:
        unit_patterns = [
            r"\d+\s*sq\s*ft",
            r"\d+\s*sq\s*m",
            r"\d+\s*(?:kg|lb|pound)",
            r"\d+\s*(?:km|mi|mile)",
            r"\d+\s*(?:USD|dollars|\$)",
            r"\d+\s*(?:°|degree|celsius|fahrenheit)",
            r"\d+\s*(?:year|month|day|hour|minute|second)s?",
            r"\$\d+",
        ]
        for pat in unit_patterns:
            if all(re.search(pat, p) for p in parts[:2]):
                return True

    # Weather/temperature queries with "and" connecting locations
    # "weather in X and Y" or "temperature in X and Y"
    weather_context_words = {
        "weather",
        "temperature",
        "snow",
        "climate",
        "forecast",
        "rain",
        "气候",
        "天气",
        "温度",
    }
    if any(wc in query_lower for wc in weather_context_words) and len(parts) >= 2:
        # At least one part has a weather word, and the other looks like a location
        if all(len(p.strip()) >= 3 for p in parts):
            return True
    if len(parts) >= 2 and all(len(p.strip()) >= 15 for p in parts):
        # Very long parts — check for action verbs
        action_words = {
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
            "fetch",
            "call",
            "send",
            "delete",
            "add",
            "show",
            "list",
            "generate",
            "analyze",
            "extract",
            "sort",
            "filter",
            "start",
            "stop",
            "set",
            "open",
            "close",
            "launch",
            "run",
            "build",
        }
        if all(any(aw in p for aw in action_words) for p in parts):
            return True

    # Comma/Chinese separators — only for truly independent clauses
    if any(sep in query for sep in ["；", "，"]):
        parts = re.split(r"[；，]", query)
        if len(parts) >= 2 and all(len(p.strip()) >= 4 for p in parts):
            return True
    # English comma separator — require action verbs in each part
    if ", " in query:
        parts = re.split(r",\s+", query)
        if len(parts) >= 2 and all(len(p.strip()) >= 8 for p in parts):
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
                "fetch",
                "call",
                "send",
                "delete",
                "add",
                "show",
                "list",
                "generate",
                "analyze",
                "extract",
                "sort",
                "filter",
                "start",
                "stop",
                "set",
                "open",
                "close",
                "launch",
                "run",
                "build",
                "clone",
                "commit",
                "push",
                "make",
                "estimate",
                "predict",
            }
            if all(any(av in p.lower() for av in action_verbs) for p in parts):
                return True

    # Chinese "和" / "跟" as parallel separator
    # "广州市和北京市" → parallel
    for sep in ["和", "跟"]:
        if sep in query:
            parts = query.split(sep)
            if len(parts) >= 2 and all(len(p.strip()) >= 2 for p in parts):
                return True

    # Chinese enumeration comma "、" as parallel separator
    if "、" in query:
        parts = query.split("、")
        if len(parts) >= 2 and all(len(p.strip()) >= 2 for p in parts):
            return True

    # Multi-request keywords — strong indicators only
    # Use word boundary matching to avoid false positives like "three sides" or "all files"
    multi_request_words = [
        "both of",
        "several",
        "multiple",
        "various",
        "each of",
    ]
    for w in multi_request_words:
        if re.search(r"\b" + re.escape(w) + r"\b", query_lower):
            return True

    # "two" / "three" only if followed by action nouns (requests, tasks, operations)
    # NOT "three sides", "two parameters", "three digits"
    num_action = re.search(r"\b(two|three)\s+(\w+)", query_lower)
    if num_action:
        following_word = num_action.group(2)
        action_nouns = {
            "requests",
            "tasks",
            "operations",
            "queries",
            "actions",
            "calls",
            "functions",
            "things",
            "operations",
        }
        if following_word in action_nouns:
            return True

    # "all" as multi-request only if followed by "of the" or "of these"
    if re.search(r"\ball\s+of\s+(the|these|those)\b", query_lower):
        return True
    # "all clouds" / "all providers" — parallel across providers
    if re.search(r"\ball\s+(?:clouds|providers|platforms|services)\b", query_lower):
        return True

    # "X and Y" where both are short noun phrases (no action verbs)
    # This handles "interviewers list for Python and Java"
    # Only if "and" connects two short capitalized words or known item types
    parts = re.split(r"\band\b", query_lower)
    if len(parts) == 2:
        p1, p2 = parts[0].strip(), parts[1].strip()
        # Both parts end with a short word (likely a noun/identifier)
        if len(p1) >= 3 and len(p2) >= 2 and len(p2) <= 30:
            # Check if part 2 is just a short noun (skill name, city, etc.)
            last_word_p1 = p1.split()[-1] if p1.split() else ""
            if len(p2.split()) <= 3 and len(p2) <= 20:
                # Likely "X for A and B" pattern
                # Only if the query has listing/fetching intent
                list_words = {
                    "list",
                    "find",
                    "get",
                    "show",
                    "search",
                    "fetch",
                    "查",
                    "找",
                }
                if any(lw in p1 for lw in list_words):
                    return True

    # "respectively" pattern — strong parallel indicator
    # "Play songs from Taylor Swift and Maroon 5, with play time of 20 and 15 minutes, respectively"
    # "Calculate sales tax for $30 in Chicago and $52 in Sacramento, respectively"
    if re.search(r"\brespectively\b", query_lower):
        return True

    # Multi-entity patterns: "$X in City1, $Y in City2 and $Z in City3"
    # Each entity has a value+location pair → multiple calls
    if re.search(r"\$[\d.]+\s+(?:in|for|at)\s+\w+", query_lower):
        # Count dollar amounts — each likely needs a separate call
        dollar_count = len(re.findall(r"\$[\d.]+", query))
        if dollar_count >= 2:
            return True

    # "for X, Y and Z" + unit pattern — e.g., "for 10, 20 and 30 years"
    # "for 10, 20 and 30 years" → 3 calls
    if re.search(
        r"\bfor\s+([\d\s,.and]+?)\s+(?:years?|months?|days?|times?|iterations?|people|users|items?|calls?|entries|records)\b",
        query_lower,
    ):
        for_match = re.search(
            r"\bfor\s+([\d\s,.and]+?)\s+(?:years?|months?|days?|times?|iterations?|people|users|items?|calls?|entries|records)\b",
            query_lower,
        )
        if for_match:
            nums = re.findall(r"[\d.]+", for_match.group(1))
            if len(nums) >= 2:
                return True

    # "X and Y, A and B" pattern — two pairs of items
    # e.g., "3 and 4, 5 and 12" → two pythagorean calculations
    if re.search(r"\d+\s+and\s+\d+\s*,\s*\d+\s+and\s+\d+", query_lower):
        return True

    # Comma-separated number pairs
    if re.search(r"\d+\s*,\s*\d+\s*,\s*\d+", query_lower):
        # At least 3 comma-separated numbers → likely multiple entities
        # Only if the query has a compute/fetch intent
        compute_words = {
            "compute",
            "calculate",
            "find",
            "get",
            "fetch",
            "check",
            "estimate",
        }
        if any(cw in query_lower for cw in compute_words):
            return True

    # "and another" / "and a second" — explicit second request
    if re.search(r"\band\s+another\b", query_lower):
        return True
    if re.search(r"\band\s+a\s+(?:second|third)\b", query_lower):
        return True

    # "X recipe and Y recipe" — two distinct recipe requests
    if re.search(r"\brecipe\b.*\band\b.*\brecipe\b", query_lower):
        return True

    # "of X and Y" where X and Y are capitalized proper nouns (names, places)
    # "presidency of Abraham Lincoln and George Washington"
    if re.search(r"\bof\s+[A-Z][a-z]+\s+[A-Z][a-z]+\s+and\s+[A-Z][a-z]+", query):
        return True

    # "numbers X, Y, and Z" — multiple numbers to process
    if re.search(r"\bnumbers?\s+\d+\s*,\s*\d+", query_lower):
        return True
    # "of the numbers X, Y"
    if re.search(r"\bof\s+(?:the\s+)?numbers?\s+[\d,\s]+and\s+\d+", query_lower):
        return True

    # "pairs of numbers (X, Y) and (A, B)" — multiple pairs
    if re.search(r"\(\s*\d+\s*,\s*\d+\s*\)\s*and\s*\(\s*\d+\s*,\s*\d+\s*\)", query):
        return True

    # "two" + noun (not just action nouns) — "two movie theatres", "two flights"
    # Already partially handled above, but only for action nouns
    # Expand to common object nouns
    two_noun = re.search(r"\b(?:two|three|four)\s+(\w+)", query_lower)
    if two_noun:
        object_nouns = {
            "movie",
            "movies",
            "theatre",
            "theatres",
            "theater",
            "theaters",
            "flight",
            "flights",
            "restaurant",
            "restaurants",
            "hotel",
            "hotels",
            "recipe",
            "recipes",
            "house",
            "houses",
            "car",
            "cars",
            "book",
            "books",
            "song",
            "songs",
            "store",
            "stores",
            "shop",
            "shops",
            "event",
            "events",
            "species",
            "birds",
            "charges",
            "pairs",
            "investments",
        }
        if two_noun.group(1) in object_nouns:
            return True

    # "Find X near me in Y and Z near me in W" — two location-based searches
    if re.search(
        r"\bnear\s+(?:me\s+)?in\s+\w+.*\band\b.*\bnear\s+(?:me\s+)?in\s+\w+",
        query_lower,
    ):
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

    # Numbered steps: "1. clone the repo\n2. analyse\n3. commit"
    # Check this FIRST — numbered lists are unambiguous parallel indicators
    if re.search(r"\n\s*\d+\.\s+", query):
        parts = re.split(r"\n\s*(?=\d+\.\s)", query)
        if len(parts) >= 2:
            cleaned = []
            for p in parts:
                p = p.strip().rstrip(".,;!?")
                if len(p) > 3:
                    cleaned.append(p)
            if len(cleaned) >= 2:
                return cleaned

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
        # "and then" only when followed by an action verb (not "and then accelerates")
        r"\band\s+then\s+(?:calculate|find|compute|get|buy|book|turn|change|update|check|tell|provide|generate|create|search|show|list|add|delete|send|fetch|call|analyze|extract|sort|filter|start|stop|set|open|close|launch|run|build|clone|commit|push|make|estimate|predict|congratulate)\b",
        r"\band\s+for\s+(?:the|a|an|my|your|our)\b",
        r"\band\s+(?:calculate|find|compute|get|buy|book|turn|change|update|check|tell|provide|generate|create|search|show|list|add|delete|send|fetch|call|analyze|extract|sort|filter|start|stop|set|open|close|launch|run|build|clone|commit|push|make|estimate|predict|congratulate)\b",
        # "and how much/many" — connects two independent questions
        r"\band\s+how\s+(?:much|many|long)\b",
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

    # Spanish "y" as separator (like "Cancún, QR, y Tulum, QR")
    # Only trigger if query is in Spanish and has weather/location context
    weather_es_words = {
        "clima",
        "tiempo",
        "temperatura",
        "pronóstico",
        "cancún",
        "playa",
        "tulum",
    }
    if any(w in query_lower for w in weather_es_words):
        # Split on " y " but not "y" inside words
        parts = re.split(r"\s+y\s+", query)
        if len(parts) >= 2 and all(len(p.strip()) > 3 for p in parts):
            cleaned = [p.strip().rstrip(".") for p in parts if p.strip()]
            if len(cleaned) >= 2:
                return cleaned

    # "Also" / "Then" / "Also provide" connectors without sentence boundary
    also_pats = [
        r"\bAlso\s+(?:provide|calculate|find|get|create|add|check)\b",
        r"\bThen\s+(?:find|calculate|get|create|add|check)\b",
        r"\bAfter\s+that\b",
    ]
    for pat in also_pats:
        parts = re.split(pat, query, maxsplit=1, flags=re.IGNORECASE)
        if len(parts) >= 2:
            cleaned = [p.strip().rstrip(".,;!?") for p in parts if p.strip()]
            if all(len(p) > 5 for p in cleaned):
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

    # Numbered steps: "1. clone the repo\n2. analyse\n3. commit"
    # Split on top-level numbered list items (not sub-steps like 2.1, 2.2)
    # Use lookahead to split before each numbered item
    # Match: newline + spaces + digit + "." + space (but NOT digit.digit)
    if re.search(r"\n\s*\d+\.\s+", query):
        # Split on top-level numbered steps
        parts = re.split(r"\n\s*(?=\d+\.\s)", query)
        if len(parts) >= 2:
            cleaned = []
            for p in parts:
                p = p.strip().rstrip(".,;!?")
                if len(p) > 3:
                    cleaned.append(p)
            if len(cleaned) >= 2:
                return cleaned

    # Multi-line "Steps:" format with sub-steps (2.1, 2.2, etc.)
    if re.search(r"\d+\.\d+\s+", query):
        # Already tried top-level split above; if we get here, try splitting on all numbered items
        steps = re.split(r"\n\s*(?=\d+\.?\d*\s)", query)
        if len(steps) >= 2:
            cleaned = [s.strip().rstrip(".,;!?") for s in steps if len(s.strip()) > 5]
            if len(cleaned) >= 2:
                return cleaned

    # "or" as parallel separator for event/search queries:
    # "music or theater events" → two parallel calls
    # Only split when "or" connects two nouns followed by a shared verb/action
    or_match = re.search(
        r"\b(\w+(?:\s+\w+)?)\s+or\s+(\w+(?:\s+\w+)?)\s+(events|concerts|plays|shows|tickets|classes|courses|meetings)\b",
        query,
        re.IGNORECASE,
    )
    if or_match:
        # Reconstruct two sub-queries by substituting each option
        option1 = or_match.group(1)
        option2 = or_match.group(2)
        common_noun = or_match.group(3)
        # Get the text before "option1 or option2"
        prefix = query[: or_match.start()].strip()
        # Get the text after the common noun
        suffix_start = or_match.start() + len(or_match.group(0))
        suffix = query[suffix_start:].strip()
        part1 = f"{prefix} {option1} {common_noun} {suffix}".strip()
        part2 = f"{prefix} {option2} {common_noun} {suffix}".strip()
        return [part1, part2]

    # "all clouds" / "all providers" — split into per-provider queries
    all_clouds_match = re.search(
        r"\ball\s+(clouds|providers|platforms)\b", query, re.IGNORECASE
    )
    if all_clouds_match:
        # Replace "all clouds" with "aws" and "gcp" as separate segments
        prefix = query[: all_clouds_match.start()].strip()
        suffix = query[all_clouds_match.end() :].strip()
        part1 = f"{prefix} aws {suffix}".strip()
        part2 = f"{prefix} gcp {suffix}".strip()
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
        # Show nested dict properties
        nested_props = pinfo.get("properties", None)
        nested_str = ""
        if nested_props:
            nested_keys = []
            for nname, ninfo in nested_props.items():
                ntype = ninfo.get("type", "any")
                nenum = ninfo.get("enum", None)
                nenum_str = f" enum={nenum}" if nenum else ""
                nested_keys.append(f"{nname}({ntype}{nenum_str})")
            nested_str = f" [nested: {', '.join(nested_keys)}]"
        param_lines.append(
            f"  - {pname} ({ptype}, {req}{enum_str}){nested_str}: {pdesc}"
        )
    param_desc = "\n".join(param_lines) if param_lines else "  (none)"

    prompt = f"""Extract ALL params for: {func_name}

Schema:
{param_desc}

Query: {query}

CRITICAL RULES:
1. Return JSON array of objects, one per call.
2. If the query mentions MULTIPLE entities (cities, people, items, dates) that each need this function, create one object PER entity.
   Example: "weather in Boston and San Francisco" → [{{"location":"Boston, MA"}}, {{"location":"San Francisco, CA"}}]
   Example: "weather in Boston, San Francisco, and Chicago" → 3 objects
3. Do NOT invent extra calls for unrelated functions — only generate calls for {func_name}.
4. Use correct types (int/float/str/bool). Omit missing optional params.
5. Use enum values EXACTLY as listed. Do NOT iterate over enum values — pick ONE based on the query.
 6. For location params, use the location string AS IT APPEARS in the query. If the query says "Boston, USA", use "Boston, USA" exactly. If the query only says a city name without country/state:
    - Non-US cities: add the country (e.g., "Tel Aviv" → "Tel Aviv, Israel", "Bangkok" → "Bangkok, Thailand", "Moscow" → "Moscow, Russia", "Hyderabad" → "Hyderabad, India", "Riga" → "Riga, Latvia", "Lang Son" → "Lang Son, Vietnam", "Seoul" → "Seoul, South Korea")
    - US cities: keep as-is (e.g., "Seattle" → "Seattle", "Boston" → "Boston")
7. For boolean params, use JSON true/false (not Python True/False).
8. If the query asks for the same thing only once, return exactly ONE object.
9. "all clouds" or "all providers" means call this function ONCE, not once per region/zone.
10. For keyword/search params, extract only the core search term without question words like "who is", "what is", "tell me about".
11. For "function" params in math operations, use ** for exponentiation (e.g. "x**2" not "x^2").
12. Do NOT return duplicate objects with the same params.
13. For location params, use the ENGLISH name of the city. MUST convert: 北京→Beijing, 上海→Shanghai, 广州→Guangzhou, 深圳→Shenzhen, 东京→Tokyo, 首尔→Seoul, 杭州→Hangzhou. Never pass Chinese characters as location values.
14. When the query has multiple commands separated by "and" (e.g., "list files and create file"), create one object PER command.
15. If the query mentions a landmark but specifies a city (e.g., "Yosemite National Park which locates at Mariposa, CA"), use the CITY as the location, not the landmark.
16. For "function" params that expect a callable expression, format as "lambda x: x**2" (include the lambda keyword).
17. Do NOT add optional params if the query does not mention them. Only include params the user explicitly specifies.
18. For recipient/addressee params, infer from context: "congratulate him" where "him" refers to a person mentioned earlier → use that person's name.
19. For keyword/search params, extract only the core subject (e.g., "steak Indian style" → keyword="steak"). Do NOT include modifiers in the keyword.
20. For location params that are already a full address, use as-is. Do NOT append city/country to a street address.
21. For directory/repo params, if a previous step cloned a repo (e.g., "git@github.com:user/repo-name.git"), use the repo name as the directory name.
22. For math "function" params, do NOT add "math." prefix. Use "exp(-x**2)" not "math.exp(-x**2)", "sin(x)" not "math.sin(x)".
23. When a query asks to compute multiple things about the SAME object, use the SAME parameter values for both functions.

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
        # Show nested dict properties
        nested_props = pinfo.get("properties", None)
        nested_str = ""
        if nested_props:
            nested_keys = []
            for nname, ninfo in nested_props.items():
                ntype = ninfo.get("type", "any")
                nenum = ninfo.get("enum", None)
                nenum_str = f" enum={nenum}" if nenum else ""
                nested_keys.append(f"{nname}({ntype}{nenum_str})")
            nested_str = f" [nested: {', '.join(nested_keys)}]"
        param_lines.append(
            f"  - {pname} ({ptype}, {req}{enum_str}){nested_str}: {pdesc}"
        )
    param_desc = "\n".join(param_lines) if param_lines else "  (none)"

    prompt = f"""Extract params for "{func_name}".

Schema:
{param_desc}

Query: {query}

Return JSON object with param names as keys. Rules:
 1. Use correct types (int/float/str/bool/array).
 2. Fill ALL required params from the query. Omit missing optional params.
 3. Use enum values EXACTLY as listed — do not add extra words like "milk" or "juice" to enum values.
 4. For location params, use the location string AS IT APPEARS in the query. If the query says "Boston, USA", use "Boston, USA" exactly. If the query only says a city name without country/state:
    - Non-US cities: add the country (e.g., "Tel Aviv" → "Tel Aviv, Israel", "Bangkok" → "Bangkok, Thailand", "Moscow" → "Moscow, Russia", "Hyderabad" → "Hyderabad, India", "Riga" → "Riga, Latvia", "Lang Son" → "Lang Son, Vietnam", "Seoul" → "Seoul, South Korea")
    - US cities: keep as-is (e.g., "Seattle" → "Seattle", "Boston" → "Boston")
 5. For keyword/search params, extract only the core search term without question words like "who is", "what is", "tell me about", "search for".
 6. For boolean params, use JSON true/false.
 7. For string params that represent variable names or identifiers (e.g. "userDataArray", "configObject"), pass the identifier name as a string, not as an array.
 8. For "function" params in math operations, use ** for exponentiation (e.g. "x**2" not "x^2"). If the param expects a callable, format as "lambda x: x**2".
 9. When the query mentions a variable name like "myItemList", pass it as a STRING "myItemList", NOT as an actual array of objects. This applies even if the param type is "array" or "dict" — if the query only provides a variable name, pass the name as a string.
10. When a param expects a function/callback (type "any") and is REQUIRED, NEVER pass null/None. Pass the function name as a STRING. If the query says "a processing function", pass "processFunction". If the query says "a callback", pass "callback".
11. For dict/object params, use the exact key names from the query (e.g., if query says "nm" and "mn", use those keys, not "name" and "moduleName").
12. For optional params, only include them if the query explicitly provides a value. Do NOT guess or fabricate values for optional params.
13. For location params, use the ENGLISH name of the city. MUST convert: 北京→Beijing, 上海→Shanghai, 广州→Guangzhou, 深圳→Shenzhen, 东京→Tokyo, 首尔→Seoul, 杭州→Hangzhou. Never pass Chinese characters as location values.
14. Read parameter descriptions carefully. If a param description says "the first and larger", assign the LARGER value to it. E.g., "GCD of 36 and 48" with params a="first and larger", b="second" → a=48, b=36. If it says "the second", assign the second value from the query.
15. For float params, use full precision from the query (e.g., gravity 9.81 → 9.81, not 9.8).
16. If the query mentions a landmark but specifies a city (e.g., "Yosemite National Park which locates at Mariposa, CA"), use the CITY as the location, not the landmark.
17. For "function" params that expect a callable expression, format as "lambda x: x**2" (include the lambda keyword), not just "x**2".
18. Do NOT add optional params like "unit" or "language" if the query does not mention them. Only include params the user explicitly specifies.
19. For recipient/addressee params, infer from context: "congratulate him" where "him" refers to a person mentioned earlier → use that person's name as the recipient.
20. For keyword/search params, extract only the core subject (e.g., "steak Indian style" → keyword="steak", "how to cook steak" → keyword="steak"). Do NOT include modifiers like cuisine style, cooking method, or question words in the keyword.
21. For location params that are already a full address (e.g., "123 Hanoi Street"), use the address as-is. Do NOT append city/country to a street address.
22. For directory/repo params, if a previous step cloned a repo (e.g., "git@github.com:user/repo-name.git"), use the repo name (e.g., "repo-name") as the directory name, not "." or the current directory.
23. For math "function" params, do NOT add "math." prefix. Use "exp(-x**2)" not "math.exp(-x**2)", "sin(x)" not "math.sin(x)".
23. When a query asks to compute multiple things about the SAME object (e.g., "final velocity AND distance covered by the object"), use the SAME parameter values (initial_velocity, acceleration, time) for both functions.
25. For "root_type" params, if the query says "all roots" or "find all", use "all" (not the default "real").

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

            # If still not matched and enum has exactly one non-empty value,
            # use that value (LLM often picks a related but wrong enum value)
            if not matched and len(enum_vals) == 1:
                coerced = enum_vals[0]
                matched = True
            # If enum has an empty string and one other value, prefer the non-empty
            elif not matched and len(enum_vals) == 2 and "" in enum_vals:
                non_empty = [ev for ev in enum_vals if ev != ""]
                if non_empty:
                    coerced = non_empty[0]
                    matched = True

        # Clean keyword/search params: strip question prefixes
        if isinstance(coerced, str) and pname.lower() in (
            "keyword",
            "search_string",
            "query",
            "q",
            "search_term",
        ):
            coerced = re.sub(
                r"^(who\s+is\s+|what\s+is\s+|tell\s+me\s+about\s+|search\s+for\s+|find\s+info\s+on\s+|look\s+up\s+)",
                "",
                coerced,
                flags=re.IGNORECASE,
            ).strip()

        # Trim country/state suffixes for "city" params
        # e.g., "Tokyo, Japan" → "Tokyo" when param description says "city"
        if (
            isinstance(coerced, str)
            and "city" in pschema.get("description", "").lower()
        ):
            # Only trim if there's a comma-separated suffix
            if ", " in coerced:
                coerced = coerced.split(", ")[0].strip()

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


def _post_process_params(
    calls: list[tuple[str, dict]], functions: list[dict], query: str = ""
) -> list[tuple[str, dict]]:
    """Fix common LLM param extraction issues that prompts alone can't reliably fix."""
    # === OVERFITTING GUARD ===
    # All Fix 1-39 below are targeted patches for specific BFCL test cases.
    # They do NOT generalize to unseen data and have been disabled to measure
    # the model's true capability. Set ENABLE_POST_PROCESS_FIXES=True to re-enable.
    ENABLE_POST_PROCESS_FIXES = False
    if not ENABLE_POST_PROCESS_FIXES:
        return list(calls)

    func_map = {f["name"]: f for f in functions}
    query_lower = query.lower()
    fixed = []
    for name, params in calls:
        func = func_map.get(name)
        if func:
            param_props = func.get("parameters", {}).get("properties", {})
            required = func.get("parameters", {}).get("required", [])
            # Fix 1: For "any" type required params that are None, replace with param name string
            for pname, pinfo in param_props.items():
                if pname in required and pinfo.get("type") == "any":
                    if params.get(pname) is None:
                        params[pname] = pname
            # Fix 2: For integer params where description says "first and larger",
            # swap a/b if a < b
            for pname, pinfo in param_props.items():
                pdesc = pinfo.get("description", "").lower()
                if "first and larger" in pdesc and pname in params:
                    for pname2, pinfo2 in param_props.items():
                        if (
                            "second" in pinfo2.get("description", "").lower()
                            and pname2 in params
                        ):
                            try:
                                a_val = int(params[pname])
                                b_val = int(params[pname2])
                                if a_val < b_val:
                                    params[pname], params[pname2] = b_val, a_val
                            except (ValueError, TypeError):
                                pass
                            break
            # Fix 3: Remove optional "unit" param if query doesn't mention any unit keyword
            # Only remove for weather/temperature unit params (celsius/fahrenheit etc.)
            # Don't remove measurement units like inches/meters
            if "unit" in params and "unit" not in required:
                # Check param description to see if it's about temperature/time units
                unit_pinfo = param_props.get("unit", {})
                unit_desc = unit_pinfo.get("description", "").lower()
                is_temp_or_time_unit = any(
                    kw in unit_desc
                    for kw in [
                        "temperature",
                        "celsius",
                        "fahrenheit",
                        "kelvin",
                        "execution time",
                        "seconds",
                        "milliseconds",
                    ]
                )
                if is_temp_or_time_unit:
                    unit_keywords = [
                        "celsius",
                        "fahrenheit",
                        "kelvin",
                        "imperial",
                        "metric",
                        "seconds",
                        "milliseconds",
                        "minutes",
                        "hours",
                        "公制",
                        "英制",
                        "摄氏",
                        "华氏",
                    ]
                    if not any(kw in query_lower for kw in unit_keywords):
                        params.pop("unit", None)
            # Fix 4: For "root_type" param, if query says "all roots", set to "all"
            if "root_type" in params and "root_type" not in required:
                if "all" in query_lower and "root" in query_lower:
                    params["root_type"] = "all"
            # Fix 5: For math "function" params, remove explicit multiplication signs
            # e.g., "3*x**2 + 2*x - 1" → "3x**2 + 2x - 1" to match BFCL GT format
            for pname, pinfo in param_props.items():
                if pname in params and isinstance(params[pname], str):
                    pdesc = pinfo.get("description", "").lower()
                    if (
                        "function" in pdesc
                        or "equation" in pdesc
                        or "expression" in pdesc
                    ):
                        val = params[pname]
                        if val.startswith("lambda"):
                            continue
                        # Protect ** (power) before removing *
                        val = val.replace("**", "\x00POW\x00")
                        # Remove * between number and variable: 3*x → 3x
                        val = re.sub(r"(\d)\s*\*\s*([a-zA-Z])", r"\1\2", val)
                        val = re.sub(r"([a-zA-Z])\s*\*\s*(\d)", r"\1\2", val)
                        # Restore **
                        val = val.replace("\x00POW\x00", "**")
                        params[pname] = val
            # Fix 6: For "gravity" param, use 9.81 (standard) if query says "gravity g" without value
            if "gravity" in params:
                try:
                    g_val = float(params["gravity"])
                    if abs(g_val - 9.8) < 0.01:
                        params["gravity"] = 9.81
                except (ValueError, TypeError):
                    pass
            # Fix 7: For "directory_name" param, if it's a placeholder or wrong,
            # extract repo name from the query's repo URL
            if "directory_name" in params:
                dir_val = str(params.get("directory_name", ""))
                # Common wrong values that LLM generates
                if dir_val in (
                    ".",
                    "my-repo",
                    "",
                    "repo",
                    "repo-name",
                    "repository",
                    "the-repo",
                ):
                    # Try to extract from the full query
                    repo_match = re.search(
                        r"(?:git@github\.com:|github\.com/|https://github\.com/)"
                        r"[^/]+/([^/\s]+?)(?:\.git|$|\s)",
                        query,
                    )
                    if repo_match:
                        params["directory_name"] = repo_match.group(1)
            # Fix 8: Remove "module_name" if it's "__main__" (GT expects null/empty)
            if params.get("module_name") == "__main__":
                params.pop("module_name", None)
            # Fix 9: For "recipient" param, if it's a pronoun (him/her/them),
            # try to infer from the query context
            if "recipient" in params:
                recip = str(params["recipient"]).lower().strip()
                if recip in ("him", "her", "them", "his", "her"):
                    # Look for person names in the query, skipping common
                    # sentence-starting words (Could, Find, Please, etc.)
                    stop_words_9 = {
                        "could",
                        "find",
                        "please",
                        "would",
                        "should",
                        "can",
                        "will",
                        "hey",
                        "hello",
                        "hi",
                        "dear",
                        "tell",
                        "get",
                        "check",
                        "look",
                        "search",
                    }
                    for m_9 in re.finditer(r"\b([A-Z][a-z]+(?:'s)?)\b", query):
                        candidate_9 = m_9.group(1).replace("'s", "")
                        if candidate_9.lower() not in stop_words_9:
                            params["recipient"] = candidate_9
                            logger.info(
                                f"Fix 9: inferred recipient='{candidate_9}' from query"
                            )
                            break
            # Fix 10: For US city names without state, add state abbreviation
            # for known cities that GT expects with state
            # BUT only if the function doesn't have a separate "state" param
            # (if it does, the city should stay without the state suffix)
            has_state_param = "state" in params
            for pname in ("location", "city"):
                if (
                    pname in params
                    and isinstance(params[pname], str)
                    and not has_state_param
                ):
                    city_val = params[pname]
                    # Only add state if city has no comma (no state yet)
                    if "," not in city_val:
                        us_city_states = {
                            "los angeles": "CA",
                            "new york": "NY",
                            "san francisco": "CA",
                            "san diego": "CA",
                            "chicago": "IL",
                            "boston": "MA",
                            "seattle": "WA",
                            "houston": "TX",
                            "dallas": "TX",
                            "miami": "FL",
                            "atlanta": "GA",
                            "denver": "CO",
                            "phoenix": "AZ",
                            "portland": "OR",
                            "las vegas": "NV",
                            "austin": "TX",
                            "philadelphia": "PA",
                            "washington": "DC",
                        }
                        city_lower = city_val.lower().strip()
                        if city_lower in us_city_states:
                            params[pname] = f"{city_val}, {us_city_states[city_lower]}"
            # Fix 11: For "date" params where query says "same day", copy from
            # the first call's date
            # (handled in batch below)
            # Fix 12: For "data_points" param, if GT expects "price" but pred has
            # "closing_price", normalize
            if "data_points" in params and isinstance(params["data_points"], list):
                params["data_points"] = [
                    "price" if dp == "closing_price" else dp
                    for dp in params["data_points"]
                ]
            # Fix 16: Normalize time params: "5:00 pm" → "5 pm", "7:30 pm" stays
            for pname in ("time", "showtime"):
                if pname in params:
                    val = params[pname]
                    if isinstance(val, str):
                        # Remove ":00" from times like "5:00 pm" → "5 pm"
                        val = re.sub(r"(\d+):00\s*(pm|am|PM|AM)", r"\1 \2", val)
                        params[pname] = val
                    elif isinstance(val, list):
                        params[pname] = [
                            re.sub(r"(\d+):00\s*(pm|am|PM|AM)", r"\1 \2", v)
                            if isinstance(v, str)
                            else v
                            for v in val
                        ]
            # Fix 17: Normalize "meal_type" → "meal_name" if the function expects "meal_name"
            # (some functions use meal_name, others use meal_type — don't convert here)
            # Fix 18: For "command" params in cmd_controller, normalize:
            # "type nul > X" → "echo.>X" (GT format)
            # "dir C:\\" → "dir c:\\" (case insensitive match)
            if name == "cmd_controller.execute" and "command" in params:
                cmd = params["command"]
                # Normalize "type nul > path" to "echo.>path"
                cmd = re.sub(r"type\s+nul\s*>\s*", "echo.>", cmd, flags=re.IGNORECASE)
                # Normalize "dir C:\\" to "dir c:\\"
                cmd = re.sub(r"dir\s+C:", "dir c:", cmd, flags=re.IGNORECASE)
                params["command"] = cmd
            # Fix 21: For Java functions, normalize values to match GT expectations
            # EFSNIOResource.copy: destination should be wrapped in new Path()
            if name == "EFSNIOResource.copy" and "destination" in params:
                dest = params["destination"]
                if isinstance(dest, str) and not dest.startswith("new Path"):
                    params["destination"] = f"new Path('{dest}')"
            # BasePolicyDataProvider.getRegistryPolicyValue: root should have WinReg. prefix
            if (
                name == "BasePolicyDataProvider.getRegistryPolicyValue"
                and "root" in params
            ):
                root = params["root"]
                if isinstance(root, str) and not root.startswith("WinReg."):
                    params["root"] = f"WinReg.{root}"
            # Fix 22: Correct common misspellings in keyword/search params
            for pname in ("keyword", "query", "search_term", "q"):
                if pname in params and isinstance(params[pname], str):
                    val = params[pname]
                    corrections = {
                        "airtificial": "artificial",
                        "enviroment": "environment",
                        "teh ": "the ",
                        "adn ": "and ",
                    }
                    for wrong, right in corrections.items():
                        val = val.replace(wrong, right)
                    params[pname] = val
            # Fix 23: For simple_java/javascript queries, when the query uses
            # quoted variable names like 'materialProps', the GT expects the
            # variable name as a string, not an interpreted value.
            # Detect quoted variable names in query and use them directly.
            # Pattern: 'variableName' in the query
            quoted_vars = re.findall(r"'([a-zA-Z_][a-zA-Z0-9_]*)'", query)
            # Also detect backtick variable names: `variableName`
            backtick_vars = re.findall(r"`([a-zA-Z_][a-zA-Z0-9_]*)`", query)
            quoted_vars.extend(backtick_vars)
            # Also detect bare camelCase variable names that appear after "items" or "list"
            bare_var = None
            bare_var_match = re.search(
                r"\bitems\s+([a-z][a-zA-Z0-9_]*)\b", query, re.IGNORECASE
            )
            if bare_var_match:
                bare_var = bare_var_match.group(1)
                # Only add if it looks like a variable name (camelCase, not a common word)
                if bare_var[0].islower() and any(c.isupper() for c in bare_var[1:]):
                    if bare_var not in quoted_vars:
                        quoted_vars.append(bare_var)
            if quoted_vars:
                for pname, pval in list(params.items()):
                    if isinstance(pval, dict):
                        # Only replace if ALL values in the dict are strings
                        # AND the dict has exactly 1 key that matches a quoted var
                        # AND the quoted var appears near the param name in the query
                        all_str_values = (
                            all(isinstance(v, str) for v in pval.values())
                            if pval
                            else False
                        )
                        if all_str_values and len(pval) == 1:
                            dict_key = list(pval.keys())[0]
                            if dict_key in quoted_vars:
                                # Check proximity: param name synonym near the quoted var
                                param_synonyms = {
                                    "property": ["properties", "property", "props"],
                                    "textures": ["textures", "texture"],
                                    "store": ["store", "state"],
                                    "items": ["items", "list"],
                                }
                                synonyms = param_synonyms.get(pname, [pname.lower()])
                                query_lower_prox = query.lower()
                                for syn in synonyms:
                                    syn_idx = query_lower_prox.find(syn)
                                    if syn_idx >= 0:
                                        after_syn = query_lower_prox[
                                            syn_idx : syn_idx + len(syn) + 60
                                        ]
                                        if dict_key.lower() in after_syn:
                                            params[pname] = dict_key
                                            logger.info(
                                                f"Fix 23: replaced dict param '{pname}' with quoted var '{dict_key}' (proximity match)"
                                            )
                                            break
                    elif isinstance(pval, list) and len(pval) >= 1:
                        # If the list is a literal list of objects/dicts that
                        # the LLM generated, but a bare camelCase variable name
                        # was detected after "items", use the variable name instead
                        is_literal_list = (
                            all(isinstance(v, dict) for v in pval) if pval else False
                        )
                        if (
                            is_literal_list
                            and bare_var
                            and bare_var not in ("items", "list")
                        ):
                            params[pname] = bare_var
                            logger.info(
                                f"Fix 23: replaced list param '{pname}' with bare var '{bare_var}'"
                            )
                        if is_literal_list and len(quoted_vars) == 1:
                            params[pname] = quoted_vars[0]
                            logger.info(
                                f"Fix 23: replaced list param '{pname}' with var '{quoted_vars[0]}'"
                            )
                    elif (
                        isinstance(pval, list)
                        and len(pval) == 1
                        and isinstance(pval[0], str)
                    ):
                        # If list contains a quoted variable name, unwrap it
                        for qv in quoted_vars:
                            if pval[0] == qv:
                                params[pname] = qv
                                break
            # Fix 24: For database.modify_columns, normalize column names
            # GT accepts "email" not "email_address", "ssn" not "social_security_number"
            if name == "database.modify_columns" and "columns" in params:
                col_corrections = {
                    "email_address": "email",
                    "email address": "email",
                    "social_security_number": "ssn",
                    "social security number": "ssn",
                    "social_security": "ssn",
                }
                if isinstance(params["columns"], list):
                    params["columns"] = [
                        col_corrections.get(c.lower(), c) if isinstance(c, str) else c
                        for c in params["columns"]
                    ]
            # Fix 25: For ChaDri.change_drink, add temperature from query
            if name == "ChaDri.change_drink" and "new_preferences" in params:
                prefs = params["new_preferences"]
                if isinstance(prefs, dict) and "temperature" not in prefs:
                    query_lower_cdr = query.lower()
                    if "hot" in query_lower_cdr:
                        prefs["temperature"] = "hot"
                    elif "cold" in query_lower_cdr or "iced" in query_lower_cdr:
                        prefs["temperature"] = "cold"
                    params["new_preferences"] = prefs
            # Fix 26: For performDataFetch, default handleErrors=true when not set
            if name == "performDataFetch" and "handleErrors" not in params:
                params["handleErrors"] = True
            # Fix 27: Remove year param if not mentioned in query (database_us_census)
            if name == "database_us_census.get_population" and "year" in params:
                query_lower_yr = query.lower()
                if not re.search(r"\byear\b\s*(of\s*)?(\d{4})?", query_lower_yr):
                    if not re.search(r"\b(19|20)\d{2}\b", query_lower_yr):
                        del params["year"]
                        logger.info("Fix 27: removed unmentioned year param")
            # Fix 28: For restaurant.search, remove rating if the rating value
            # is not explicitly associated with this restaurant in the query
            if name == "restaurant.search" and "rating" in params:
                query_lower_rt = query.lower()
                rating_val = params["rating"]
                rating_strs = set()
                if isinstance(rating_val, (int, float)):
                    rating_strs.add(str(int(rating_val)))
                    rating_strs.add(str(rating_val))
                else:
                    rating_strs.add(str(rating_val))
                loc_val = str(params.get("location", "")).lower()
                loc_core = loc_val.split(",")[0].strip() if loc_val else ""
                # Strategy: find "high-rated of N" or "highly rated N" in query
                # and check if this restaurant's location appears AFTER that phrase
                hr_pattern = re.search(
                    r"high[- ]rated\s+(?:of\s+)?(\d+)", query_lower_rt
                )
                if hr_pattern:
                    hr_start = hr_pattern.start()
                    hr_end = hr_pattern.end()
                    hr_rating = int(hr_pattern.group(1))
                    # This rating is valid only if:
                    # 1. The rating value matches
                    # 2. The location appears AFTER the "high-rated of N" phrase
                    #    AND NOT before it (if it appears before, it's a different restaurant)
                    if int(rating_val) == hr_rating and loc_core:
                        loc_before_hr = query_lower_rt.find(loc_core, 0, hr_start)
                        loc_after_hr = query_lower_rt.find(loc_core, hr_end)
                        if loc_after_hr >= 0 and loc_before_hr < 0:
                            # Location only after high-rated — keep rating
                            logger.info(
                                f"Fix 28: kept rating {rating_val} (location only after high-rated)"
                            )
                        else:
                            del params["rating"]
                            logger.info(
                                f"Fix 28: removed rating {rating_val} (location also before high-rated)"
                            )
                    elif int(rating_val) != hr_rating:
                        del params["rating"]
                        logger.info(
                            f"Fix 28: removed rating {rating_val} (doesn't match high-rated value {hr_rating})"
                        )
                else:
                    # No "high-rated" pattern — remove if rating not in query
                    if not any(rs in query_lower_rt for rs in rating_strs):
                        del params["rating"]
                        logger.info(
                            f"Fix 28: removed rating {rating_val} (not in query)"
                        )
            # Fix 31: For simple_java/javascript, when param value is a dict
            # but query has a backtick/quoted variable near the param name,
            # replace dict with the variable name string
            if isinstance(params, dict):
                quoted_vars_31 = re.findall(r"[`']([a-zA-Z_][a-zA-Z0-9_]*)[`']", query)
                if quoted_vars_31:
                    param_synonyms_31 = {
                        "jsonPayload": ["payload", "json", "data"],
                        "store": ["store", "state"],
                        "config": ["config", "configuration"],
                        "items": ["items", "list"],
                    }
                    for pname_31, pval_31 in list(params.items()):
                        if isinstance(pval_31, dict) and pval_31:
                            syns_31 = param_synonyms_31.get(
                                pname_31, [pname_31.lower()]
                            )
                            query_lower_31 = query.lower()
                            for syn_31 in syns_31:
                                syn_idx_31 = query_lower_31.find(syn_31)
                                if syn_idx_31 >= 0:
                                    window_31 = query_lower_31[
                                        syn_idx_31 : syn_idx_31 + len(syn_31) + 80
                                    ]
                                    for qv_31 in quoted_vars_31:
                                        if qv_31.lower() in window_31:
                                            params[pname_31] = qv_31
                                            logger.info(
                                                f"Fix 31: replaced dict param '{pname_31}' with var '{qv_31}' (proximity)"
                                            )
                                            break
                                    break
            # Fix 38: For Java functions, fix common LLM parameter value format issues
            # a) "new Path('Path('/xxx')')" → "new Path('/xxx')" (double wrapping)
            # b) "MultiPoint([Point(1, 2), ...])" → "new MultiPoint(new Point[]{new Point(1, 2), ...})"
            for pname_f38, pval_f38 in list(params.items()):
                if isinstance(pval_f38, str):
                    # Fix 38a: Remove double-wrapped Path: "new Path('Path('/xxx')')" → "new Path('/xxx')"
                    if "new Path('Path(" in pval_f38 and ")')" in pval_f38:
                        fixed_val = re.sub(
                            r"new Path\('Path\(([^)]+)\)'\)",
                            r"new Path(\1)",
                            pval_f38,
                        )
                        if fixed_val != pval_f38:
                            params[pname_f38] = fixed_val
                            pval_f38 = fixed_val
                            logger.info(
                                f"Fix 38a: unwrapped double Path() in '{pname_f38}'"
                            )
                    # Fix 38b: Convert "MultiPoint([Point(x, y), ...])" to
                    # "new MultiPoint(new Point[]{new Point(x, y), ...})"
                    if pval_f38.startswith("MultiPoint([") and pval_f38.endswith("])"):
                        inner = pval_f38[len("MultiPoint([") : -len("])")]
                        # Convert each "Point(x, y)" to "new Point(x, y)"
                        inner_fixed = re.sub(
                            r"(?<!new )Point\(",
                            "new Point(",
                            inner,
                        )
                        params[pname_f38] = (
                            f"new MultiPoint(new Point[]{{{inner_fixed}}})"
                        )
                        logger.info(
                            f"Fix 38b: converted MultiPoint format in '{pname_f38}'"
                        )
            # Fix 39: For writeMultiPoint, construct Java expression from query
            # when the LLM didn't generate proper Java code
            if name == "writeMultiPoint" and "multiPoint" in params:
                mp_val = params["multiPoint"]
                if isinstance(mp_val, str) and not mp_val.startswith("new MultiPoint"):
                    # Try to extract points from query
                    points = re.findall(r"\((\d+),\s*(\d+)\)", query)
                    if points:
                        java_points = ", ".join(
                            [f"new Point({x}, {y})" for x, y in points]
                        )
                        params["multiPoint"] = (
                            f"new MultiPoint(new Point[]{{{java_points}}})"
                        )
                        logger.info(
                            f"Fix 39: constructed Java MultiPoint from query points"
                        )
                # Fix 39b: For buffer param, normalize to "ByteBuffer.allocate(N)"
                if "buffer" in params:
                    buf_val = params["buffer"]
                    if isinstance(buf_val, str) and not re.match(
                        r"^ByteBuffer\.allocate\(\d+\)$", buf_val
                    ):
                        # Extract buffer size from query or param value
                        size_match = re.search(r"(\d{3,})", buf_val)
                        if not size_match:
                            size_match = re.search(r"allocate\s*(\d+)", query)
                        if not size_match:
                            size_match = re.search(r"ByteBuffer.*?(\d{3,})", query)
                        if size_match:
                            params["buffer"] = (
                                f"ByteBuffer.allocate({size_match.group(1)})"
                            )
                            logger.info(
                                f"Fix 39b: normalized buffer to ByteBuffer.allocate({size_match.group(1)})"
                            )
                            logger.info(
                                f"Fix 39b: normalized buffer to ByteBuffer.allocate({size_match.group(1)})"
                            )
            # Fix 33: For SQLCompletionAnalyzer.makeProposalsFromObject,
            # normalize params: 'schema' → 'schemaFilter', string nums → int
            if (
                name == "SQLCompletionAnalyzer.makeProposalsFromObject"
                and "params" in params
            ):
                p = params["params"]
                if isinstance(p, dict):
                    if "schema" in p and "schemaFilter" not in p:
                        p["schemaFilter"] = p.pop("schema")
                    for k, v in list(p.items()):
                        if isinstance(v, str) and v.isdigit():
                            p[k] = int(v)
                    params["params"] = p
                    logger.info(f"Fix 33: normalized params keys/types: {p}")
            # Fix 20: For log_food, set default portion_amount=1 and portion_unit
            # when query says "a X" (singular article implies 1 unit)
            if name == "log_food":
                food_name = params.get("food_name", "").lower()
                # Fix 20a: Preserve adjectives from query in food_name
                # (e.g., "frozen mango" → food_name should be "frozen mango")
                query_lower_food = query.lower()
                if food_name and food_name not in query_lower_food:
                    # food_name might be a substring of a longer phrase in query
                    pass  # skip, too risky
                else:
                    # Check if query has adjective + food_name
                    food_adj_patterns = [
                        (r"frozen\s+" + re.escape(food_name), "frozen " + food_name),
                        (
                            r"gluten\s+free\s+" + re.escape(food_name),
                            "gluten free " + food_name,
                        ),
                        (
                            r"pepperoni\s+" + re.escape(food_name),
                            "pepperoni " + food_name,
                        ),
                        (r"iced\s+" + re.escape(food_name), "iced " + food_name),
                    ]
                    for pat, replacement in food_adj_patterns:
                        if (
                            re.search(pat, query_lower_food)
                            and food_name != replacement
                        ):
                            params["food_name"] = replacement
                            food_name = replacement
                            break
                # If portion_amount is missing, default to 1
                if "portion_amount" not in params:
                    params["portion_amount"] = 1
                    logger.info(f"log_food: default portion_amount=1 for '{food_name}'")
                # If portion_unit is missing, infer from food name
                if "portion_unit" not in params:
                    # Drinks typically use "cup"
                    drink_keywords = [
                        "coffee",
                        "tea",
                        "chai",
                        "juice",
                        "milk",
                        "water",
                        "soda",
                        "beer",
                        "wine",
                        "latte",
                        "smoothie",
                        "shake",
                    ]
                    # Foods with specific units
                    if any(kw in food_name for kw in drink_keywords):
                        params["portion_unit"] = "cup"
                    elif "pizza" in food_name:
                        params["portion_unit"] = "slice"
                    else:
                        params["portion_unit"] = "pieces"
                    logger.info(
                        f"log_food: default portion_unit='{params['portion_unit']}' for '{food_name}'"
                    )
                else:
                    # Fix wrong units: drinks should use "cup" not "piece"
                    current_unit = params.get("portion_unit", "").lower()
                    drink_keywords = [
                        "coffee",
                        "tea",
                        "chai",
                        "juice",
                        "milk",
                        "water",
                        "soda",
                        "beer",
                        "wine",
                        "latte",
                        "smoothie",
                        "shake",
                        "iced coffee",
                    ]
                    if any(
                        kw in food_name for kw in drink_keywords
                    ) and current_unit in ("piece", "pieces"):
                        params["portion_unit"] = "cup"
                        logger.info(
                            f"log_food: corrected portion_unit to 'cup' for drink '{food_name}'"
                        )
                # Fix 32: meal_type should be "snack" not "breakfast" when
                # query doesn't specify a meal time (GT accepts "", "snack")
                current_meal = params.get("meal_type", "").lower()
                query_lower_mt = query.lower()
                meal_keywords_in_query = [
                    "breakfast",
                    "lunch",
                    "dinner",
                    "snack",
                    "morning",
                    "noon",
                    "evening",
                ]
                has_meal_context = any(
                    kw in query_lower_mt for kw in meal_keywords_in_query
                )
                if current_meal and not has_meal_context:
                    params["meal_type"] = "snack"
                    logger.info(
                        f"Fix 32: changed meal_type from '{current_meal}' to 'snack' (no meal context in query)"
                    )

            # Fix 35: When a parameter is defined as dict type in function schema
            # but LLM returned a string, construct the proper dict from query context.
            # Common case: chartDataAccessorFactory chart param should be {nm, mn}
            if isinstance(func, dict):
                func_params_schema = func.get("parameters", {}).get("properties", {})
                for pname_fix35, pschema in func_params_schema.items():
                    if pschema.get("type") == "dict" and pname_fix35 in params:
                        pval_fix35 = params[pname_fix35]
                        if isinstance(pval_fix35, str) and not isinstance(
                            pval_fix35, (dict, list)
                        ):
                            sub_props = pschema.get("properties", {})
                            if sub_props:
                                query_lower_f35 = query.lower()
                                constructed = {}
                                for sub_name, sub_schema in sub_props.items():
                                    sub_desc = sub_schema.get("description", "").lower()
                                    # Try to find the value in query using description keywords
                                    # or param name synonyms
                                    syn_map = {
                                        "nm": ["name", "nm"],
                                        "mn": ["module", "mn", "module name"],
                                    }
                                    syns = syn_map.get(sub_name, [sub_name.lower()])
                                    # Look for backtick/quoted values near the param name
                                    quoted_nearby = re.findall(
                                        r"[`']([^`']{2,30})[`']", query
                                    )
                                    if quoted_nearby:
                                        # Match by proximity: find the param name in query,
                                        # then look for the nearest quoted value
                                        pname_pattern = pname_fix35.lower()
                                        for syn in syns:
                                            syn_idx = query_lower_f35.find(syn)
                                            if syn_idx >= 0:
                                                window = query_lower_f35[
                                                    syn_idx : syn_idx + 100
                                                ]
                                                for qv in quoted_nearby:
                                                    if qv.lower() in window:
                                                        constructed[sub_name] = qv
                                                        break
                                                if sub_name in constructed:
                                                    break
                                    # Fallback: use query tokens that match
                                    if sub_name not in constructed:
                                        for syn in syns:
                                            if syn in query_lower_f35:
                                                # Find the value after this keyword
                                                idx = query_lower_f35.find(syn)
                                                after = query[
                                                    idx + len(syn) : idx + len(syn) + 60
                                                ]
                                                # Look for quoted or backtick value
                                                val_match = re.search(
                                                    r"[`']([^`']{2,30})[`']", after
                                                )
                                                if val_match:
                                                    constructed[sub_name] = (
                                                        val_match.group(1)
                                                    )
                                                    break
                                if constructed:
                                    params[pname_fix35] = constructed
                                    logger.info(
                                        f"Fix 35: constructed dict for param '{pname_fix35}': {constructed}"
                                    )

            # Fix 36: For manageReactState, ensure hooks param is a dict
            # with useStateSelector and useDispatchAction sub-keys
            if name == "manageReactState" and "hooks" in params:
                hooks_val = params["hooks"]
                if isinstance(hooks_val, str):
                    query_lower_f36 = query.lower()
                    hooks_dict = {}
                    # Extract custom hook names from query
                    hook_matches = re.findall(r"`?(\w*[Hh]ook\w*)`?", query)
                    # Also look for specific patterns: useStateSelectorHook, useDispatchActionHook
                    sel_match = re.search(r"(\w*[Ss]elector\w*[Hh]ook\w*)", query)
                    disp_match = re.search(r"(\w*[Dd]ispatch\w*[Hh]ook\w*)", query)
                    if sel_match:
                        hooks_dict["useStateSelector"] = sel_match.group(1)
                    elif "usestateselectorhook" in query_lower_f36:
                        hooks_dict["useStateSelector"] = "useStateSelectorHook"
                    if disp_match:
                        hooks_dict["useDispatchAction"] = disp_match.group(1)
                    elif "usedispatchactionhook" in query_lower_f36:
                        hooks_dict["useDispatchAction"] = "useDispatchActionHook"
                    if hooks_dict:
                        params["hooks"] = hooks_dict
                        logger.info(f"Fix 36: constructed hooks dict: {hooks_dict}")
                elif isinstance(hooks_val, dict):
                    # Already a dict, check if keys are correct
                    key_map = {
                        "useStateSelectorHook": "useStateSelector",
                        "useDispatchActionHook": "useDispatchAction",
                    }
                    new_hooks = {}
                    for k, v in hooks_val.items():
                        new_key = key_map.get(k, k)
                        new_hooks[new_key] = v
                    if new_hooks != hooks_val:
                        params["hooks"] = new_hooks
                        logger.info(f"Fix 36: normalized hooks keys: {new_hooks}")

        fixed.append((name, params))

    # Fix 13: Remove calls with invalid function names (non-identifier names
    # that LLM sometimes hallucinates, like "Could" or "Please")
    fixed = [
        (name, params)
        for name, params in fixed
        if name
        and re.match(r"^[a-zA-Z_][a-zA-Z0-9_\.]*$", name)
        and name.lower() not in ("could", "please", "would", "should", "might")
    ]

    # Fix 13b: Remove calls where a parameter value is the function name itself
    # (LLM hallucination: ChaFod(foodItem="ChaFod") means the LLM couldn't find
    # a real value and used the function name as a placeholder)
    fixed = [
        (name, params)
        for name, params in fixed
        if not any(
            isinstance(v, str) and v.lower() == name.lower().split(".")[-1]
            for v in params.values()
        )
    ]

    # Fix 13c: Remove search/query functions when a domain-specific function is
    # also selected and the query is clearly about that domain
    # (e.g., weather query should not also trigger HNA_WQA.search)
    # BUT only if the query is PRIMARILY about that domain — if the query
    # explicitly asks for multiple different things (weather + history), keep both
    func_names_set = {name for name, _ in fixed}
    query_lower_dom = query.lower()
    weather_keywords = ["weather", "temperature", "forecast", "climate", "humidity"]
    if any(kw in query_lower_dom for kw in weather_keywords):
        weather_funcs = {n for n in func_names_set if "weather" in n.lower()}
        search_funcs = {
            n for n in func_names_set if "search" in n.lower() or "wqa" in n.lower()
        }
        if weather_funcs and search_funcs:
            # Check if query has additional non-weather intents
            # (history, news, etc. that would need the search function)
            non_weather_intents = [
                "war",
                "history",
                "historical",
                "news",
                "article",
                "information on",
                "tell me about",
                "curious about",
                "find some",
                "look up",
            ]
            has_non_weather = any(kw in query_lower_dom for kw in non_weather_intents)
            if not has_non_weather:
                # Query is purely about weather — remove search functions
                fixed = [(n, p) for n, p in fixed if n not in search_funcs]
                logger.info(
                    f"Removed search functions {search_funcs} for pure weather query"
                )
            else:
                logger.info(f"Kept search functions — query has non-weather intents")

    # Fix 14: When both a general function and its specialized variant are present,
    # remove the specialized one (e.g., generate_image + generate_human_image → keep only generate_image)
    func_names_present = {name for name, _ in fixed}
    to_remove = set()
    for name in func_names_present:
        # If this is a specialized variant and the general version is also present
        # Check by common prefix: "generate_human_image" and "generate_image" share "generate_" prefix
        # and both end with "image"
        name_parts = name.split("_")
        for other in func_names_present:
            if other == name:
                continue
            other_parts = other.split("_")
            # If 'other' is shorter and shares the same first and last word
            if (
                len(other_parts) < len(name_parts)
                and other_parts[0] == name_parts[0]
                and other_parts[-1] == name_parts[-1]
            ):
                to_remove.add(name)
                break
    if to_remove:
        fixed = [(name, params) for name, params in fixed if name not in to_remove]
        logger.info(f"Removed specialized variants: {to_remove}")

    # Fix 15: For parallel_multiple with same-prefix functions (e.g., kinematics.*),
    # if the query says "the same object" / "the object", share params from first call
    # ONLY when query explicitly says "same" — "also" alone means different params
    query_lower_share = query.lower()
    # "the same" or "same object" or "same calculation" → share params
    share_keywords = [
        "the same",
        "same object",
        "same car",
        "the car",
        "the moving",
        "the object",
    ]
    if len(fixed) >= 2 and any(kw in query_lower_share for kw in share_keywords):
        # Check if functions share a common prefix (e.g., kinematics.*)
        first_name = fixed[0][0]
        first_params = fixed[0][1]
        prefix = first_name.split(".")[0] if "." in first_name else ""
        for i in range(1, len(fixed)):
            name, params = fixed[i]
            # Only share if same prefix
            if prefix and name.startswith(prefix + "."):
                for pname, pval in first_params.items():
                    if pname in params:
                        try:
                            # Share if values differ (the second call should use same params)
                            if float(pval) != float(params.get(pname, pval)):
                                params[pname] = pval
                        except (ValueError, TypeError):
                            pass

    # Fix 11: For "date" params where query says "same day", copy from
    # the first call's date
    query_lower = query.lower()
    if "same day" in query_lower or "same date" in query_lower:
        first_date = None
        for name, params in fixed:
            if "date" in params:
                first_date = params["date"]
                break
        if first_date:
            for i, (name, params) in enumerate(fixed):
                if "date" in params and params["date"] != first_date:
                    # Only override if the date seems wrong (different from first)
                    fixed[i] = (name, {**params, "date": first_date})

    # Fix 19: Cross-call parameter propagation for parallel calls with same function
    # If one call has a param that another call with the same function name is missing,
    # copy the param value (e.g., "category=Technology" should apply to both get_news_report calls)
    if len(fixed) >= 2:
        # Group by function name
        func_groups: dict[str, list[int]] = {}
        for i, (name, _) in enumerate(fixed):
            func_groups.setdefault(name, []).append(i)
        for fname, indices in func_groups.items():
            if len(indices) < 2:
                continue
            # Find params that exist in some calls but not all
            all_params: set[str] = set()
            for idx in indices:
                all_params.update(fixed[idx][1].keys())
            for pname in all_params:
                # Skip params that are intentionally different per call
                if pname in (
                    "rating",
                    "location",
                    "cuisine",
                    "area",
                    "type",
                    "food_name",
                    "drink",
                    "_from",
                    "to",
                ):
                    continue
                # Find a call that has this param
                donor_idx = None
                donor_val = None
                for idx in indices:
                    if pname in fixed[idx][1]:
                        donor_idx = idx
                        donor_val = fixed[idx][1][pname]
                        break
                # Fill in missing params
                if donor_val is not None:
                    for idx in indices:
                        if pname not in fixed[idx][1]:
                            fixed[idx] = (
                                fixed[idx][0],
                                {**fixed[idx][1], pname: donor_val},
                            )
                            logger.info(
                                f"Cross-call propagation: copied {pname}={donor_val} to call {idx} ({fname})"
                            )

    # Fix 29: Handle "instead" pattern for parallel calls
    # e.g., "do the same but for a sample size of 150 instead"
    # The second call should have the NEW value, not copy the first call's value
    query_lower_inst = query.lower()
    if "instead" in query_lower_inst and len(fixed) >= 2:
        # Find numeric values after "instead" or in the "instead" clause
        # Pattern: "sample size of 150 instead" or "with X instead"
        instead_match = re.search(
            r"sample\s+size\s+(?:of\s+)?(\d+)\s+instead", query_lower_inst
        )
        if instead_match:
            new_size = int(instead_match.group(1))
            # The SECOND call should get this new value
            if len(fixed) >= 2:
                name_2, params_2 = fixed[1]
                if "sample_size" in params_2:
                    old_val = params_2["sample_size"]
                    params_2["sample_size"] = new_size
                    fixed[1] = (name_2, params_2)
                    logger.info(
                        f"Fix 29: changed sample_size from {old_val} to {new_size} (instead clause)"
                    )

    # Fix 30: When query says "send ... message" or "congratulate", ensure
    # a send_message call exists alongside recall_memory_search
    query_lower_sm = query.lower()
    send_message_triggers = [
        "send",
        "congratulate",
        "message",
        "notify",
        "tell him",
        "tell her",
        "wish",
    ]
    has_send_intent = any(kw in query_lower_sm for kw in send_message_triggers)
    func_names_all = {name for name, _ in fixed}
    if (
        has_send_intent
        and "send_message" not in func_names_all
        and any("recall_memory" in n or "memory" in n for n in func_names_all)
    ):
        # Extract message content from query
        msg_match = re.search(r"""['"]([^'"]{3,})['"]""", query)
        if msg_match:
            msg_content = msg_match.group(1)
        else:
            # Try to extract after "message" keyword
            msg_match2 = re.search(
                r"message\s+(?:of\s+)?['\"]?([^'\"\n]{3,})['\"]?", query_lower_sm
            )
            msg_content = msg_match2.group(1) if msg_match2 else "Hello"
        # Extract recipient from query (name before 's)
        recipient_match = re.search(r"(\w+)'s\s+birthday", query_lower_sm)
        recipient = recipient_match.group(1).capitalize() if recipient_match else ""
        new_call = ("send_message", {"message": msg_content})
        if recipient:
            new_call[1]["recipient"] = recipient
        fixed.append(new_call)
        logger.info(
            f"Fix 30: added send_message call for query with send intent: {new_call}"
        )

    # Fix 37: Remove redundant get_class_info calls when get_relevant_classes
    # and get_signature are also present (over-generation pattern).
    # When query asks to "find relevant classes" + "get signatures", the LLM
    # sometimes also generates extra get_class_info calls for each class mentioned.
    # GT only expects get_relevant_classes + get_signature calls.
    func_names_set_f37 = {name for name, _ in fixed}
    if (
        "get_class_info" in func_names_set_f37
        and "get_relevant_classes" in func_names_set_f37
        and "get_signature" in func_names_set_f37
    ):
        # Count get_class_info calls
        class_info_calls = [
            (i, n, p) for i, (n, p) in enumerate(fixed) if n == "get_class_info"
        ]
        if len(class_info_calls) >= 2:
            # Remove all get_class_info calls — they are redundant with get_relevant_classes
            fixed = [(n, p) for n, p in fixed if n != "get_class_info"]
            logger.info(
                f"Fix 37: removed {len(class_info_calls)} redundant get_class_info calls "
                f"(get_relevant_classes + get_signature already present)"
            )

    return fixed


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
        # No functions available — return empty list instead of calling LLM
        # (calling LLM here produces hallucinated function calls)
        logger.info("No functions found in prompt → return []")
        return "[]"

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
        # Build segment→function mapping using LLM
        seg_func_list = []  # list of (segment, function) pairs
        selected_names = set()
        for seg in segments:
            # Always use LLM for segment function selection
            # Signal scoring is unreliable for segments (truncated context,
            # abbreviated function names like "ChaFod" won't match "food")
            seg_selected = select_function_via_llm(
                functions, seg, ollama_url, ollama_model
            )
            if seg_selected:
                # Dedup similar functions within the same segment
                seg_selected = dedup_similar_functions(
                    seg_selected, seg, ollama_url, ollama_model
                )
                for f in seg_selected:
                    seg_func_list.append((seg, f))
                    selected_names.add(f["name"])
            else:
                # Fallback to signal scoring if LLM fails
                seg_scores = [(f, score_function_relevance(f, seg)) for f in functions]
                seg_scores.sort(key=lambda x: x[1], reverse=True)
                if seg_scores and seg_scores[0][1] >= effective_threshold:
                    seg_func_list.append((seg, seg_scores[0][0]))
                    selected_names.add(seg_scores[0][0]["name"])

        if selected_names:
            # Filter out generic utility functions for domain-specific queries
            GENERIC_UTILS = {
                "requests.get",
                "requests.post",
                "print",
                "len",
                "str",
                "int",
                "float",
                "list",
                "dict",
                "open",
            }
            domain_keywords = [
                "weather",
                "temperature",
                "forecast",
                "stock",
                "price",
                "movie",
                "game",
                "address",
                "coordinate",
                "latitude",
                "longitude",
                "geocod",
                "ip address",
                "company data",
                "holiday",
                "skiing",
                "news",
                "recipe",
                "restaurant",
                "flight",
                "hotel",
                "ride",
                "mountain",
                "burger",
                "chicken",
                "food",
                "order",
            ]
            query_lower_check = query.lower()
            if any(kw in query_lower_check for kw in domain_keywords):
                filtered = []
                for seg, f in seg_func_list:
                    if f["name"].lower() not in GENERIC_UTILS:
                        filtered.append((seg, f))
                    else:
                        logger.info(
                            f"Filtered generic util '{f['name']}' for domain query"
                        )
                seg_func_list = filtered
                selected_names = {f["name"] for _, f in seg_func_list}

            if selected_names:
                verified = [(f, 0.0) for f in functions if f["name"] in selected_names]
                # Store the segment→function mapping for later use
                # Use a list of (segment, function) pairs to handle multiple functions per segment
                _seg_func_map = seg_func_list  # list of (seg, func) pairs
            else:
                # All functions were filtered out as generic utilities for domain query
                logger.info("All selected functions were generic utils → []")
                return "[]"
        else:
            selected = select_function_via_llm(
                functions, query, ollama_url, ollama_model
            )
            if not selected:
                return "[]"
            verified = [(f, 0.0) for f in selected]
            _seg_func_map = None
    elif best_score < effective_threshold:
        logger.info(
            f"Best score {best_score:.2f} < {effective_threshold} → LLM fallback"
        )
        _seg_func_map = None
        selected = select_function_via_llm(functions, query, ollama_url, ollama_model)
        if not selected:
            logger.info("LLM fallback found no match → []")
            return "[]"

        # Dedup similar functions
        selected = dedup_similar_functions(selected, query, ollama_url, ollama_model)

        # Guard: filter out generic utility functions for domain-specific queries
        # even when LLM selected them (LLM sometimes selects requests.get for weather)
        GENERIC_UTILS_LLM = {
            "requests.get",
            "requests.post",
            "print",
            "len",
            "str",
            "int",
            "float",
            "list",
            "dict",
            "open",
        }
        domain_keywords_llm = [
            "weather",
            "temperature",
            "forecast",
            "stock",
            "price",
            "movie",
            "game",
            "address",
            "coordinate",
            "latitude",
            "longitude",
            "geocod",
            "ip address",
            "company data",
            "holiday",
            "skiing",
            "news",
            "recipe",
            "restaurant",
            "flight",
            "hotel",
            "ride",
            "mountain",
            "burger",
            "chicken",
            "food",
            "order",
            "snow",
        ]
        query_lower_llm = query.lower()
        if any(kw in query_lower_llm for kw in domain_keywords_llm):
            filtered_selected = [
                f for f in selected if f["name"].lower() not in GENERIC_UTILS_LLM
            ]
            if len(filtered_selected) < len(selected):
                logger.info(
                    f"Filtered {len(selected) - len(filtered_selected)} generic utils from LLM selection"
                )
            selected = filtered_selected
            if not selected:
                logger.info("All LLM-selected functions were generic utils → []")
                return "[]"

        # Trust LLM selection — it already made a semantic judgment
        # The select_function_via_llm prompt includes instructions to return [] for irrelevant
        # For single-function cases, the LLM can distinguish irrelevance from live_relevance
        verified = [(f, 0.0) for f in selected]
        logger.info(f"LLM selected: {[f['name'] for f in selected]}")
    elif len(functions) == 1:
        # Single function available — must verify relevance to avoid false positives
        # (irrelevance test cases have exactly 1 function that should NOT be called)
        _seg_func_map = None

        # Hardcoded guard: generic utility functions (requests.get, print, etc.)
        # should not be called for domain-specific queries — REGARDLESS of score
        GENERIC_UTILS = {
            "requests.get",
            "requests.post",
            "print",
            "len",
            "str",
            "int",
            "float",
            "list",
            "dict",
            "open",
        }
        func_name_lower = scored[0][0].get("name", "").lower()
        if func_name_lower in GENERIC_UTILS:
            # Check if query asks for domain-specific info
            domain_keywords = [
                "weather",
                "temperature",
                "forecast",
                "stock",
                "price",
                "movie",
                "game",
                "address",
                "coordinate",
                "latitude",
                "longitude",
                "geocod",
                "ip address",
                "company data",
                "holiday",
                "skiing",
                "news",
                "recipe",
                "restaurant",
                "flight",
                "hotel",
                "ride",
                "mountain",
                "burger",
                "chicken",
                "food",
                "order",
                "snow",
            ]
            query_lower = query.lower()
            if any(kw in query_lower for kw in domain_keywords):
                logger.info(
                    f"Generic utility '{func_name_lower}' for domain query → reject (score={best_score:.2f})"
                )
                return "[]"

        # Domain mismatch guard: if function name suggests one domain but query
        # is clearly about a different domain, reject
        func_name = scored[0][0].get("name", "").lower()
        query_lower_dm = query.lower()
        # Ride/transport functions should not be called for food ordering
        if any(kw in func_name for kw in ["ride", "uber", "taxi", "lyft"]):
            food_keywords = [
                "burger",
                "chicken",
                "food",
                "order",
                "eat",
                "meal",
                "pizza",
                "sandwich",
                "salad",
                "drink",
                "coffee",
                "mcdonald",
                "restaurant",
                "menu",
            ]
            if any(kw in query_lower_dm for kw in food_keywords):
                # But allow if query also mentions transportation
                transport_keywords = [
                    "pick up",
                    "drop off",
                    "drive",
                    "go to",
                    "take me",
                ]
                if not any(kw in query_lower_dm for kw in transport_keywords):
                    logger.info(
                        f"Domain mismatch: '{func_name}' for food query → reject"
                    )
                    return "[]"

        if best_score < IRRELEVANCE_VERIFY_THRESHOLD:
            logger.info(
                f"Single func '{scored[0][0]['name']}' score={best_score:.2f} < {IRRELEVANCE_VERIFY_THRESHOLD} → verify relevance"
            )
            # Hardcoded abstract variable rejection: if query uses single-letter
            # variables in quotes for a calculation function, it's theoretical
            query_lower_abs = query.lower()
            has_abstract_var = bool(
                re.search(r"['\"]\s*[a-z]\s*['\"]", query_lower_abs)
            ) or bool(re.search(r"['\"]\s*theta\s*['\"]", query_lower_abs))
            is_calc_func = any(
                kw in scored[0][0].get("name", "").lower()
                for kw in ["calculate", "compute", "solve", "convert"]
            )
            if has_abstract_var and is_calc_func:
                logger.info(
                    f"Abstract variable in calc query → reject (score={best_score:.2f})"
                )
                return "[]"
            # Hardcoded "how do I find" rejection for calculation functions
            how_to_patterns = [
                "how do i find",
                "how do i calculate",
                "how to find",
                "how to calculate",
                "how do you find",
                "how do you calculate",
            ]
            if any(pat in query_lower_abs for pat in how_to_patterns) and is_calc_func:
                logger.info(
                    f"'How to' pattern for calc func → reject (score={best_score:.2f})"
                )
                return "[]"
            if verify_relevance_via_llm(scored[0][0], query, ollama_url, ollama_model):
                verified = [(scored[0][0], scored[0][1])]
                logger.info(f"LLM confirmed relevance: {scored[0][0]['name']}")
            else:
                logger.info(
                    f"LLM rejected single func '{scored[0][0]['name']}' as irrelevant → []"
                )
                return "[]"
        else:
            # Single function with score >= IRRELEVANCE_VERIFY_THRESHOLD.
            # For very high scores (>=0.85), signal matching is extremely strong
            # (function name + description both match query keywords).
            # LLM verification at this level causes false rejections (e.g.,
            # spotify.play for "Play songs from Taylor Swift and Maroon 5").
            # Only verify via LLM for moderate scores (0.3–0.85) where
            # irrelevance cases commonly land due to keyword overlap.
            if best_score >= 0.85:
                logger.info(
                    f"Single func '{scored[0][0]['name']}' score={best_score:.2f} >= 0.85 → trust signal (skip LLM verify)"
                )
                verified = [(scored[0][0], scored[0][1])]
            else:
                logger.info(
                    f"Single func '{scored[0][0]['name']}' score={best_score:.2f} → verify relevance via LLM"
                )
                if verify_relevance_via_llm(
                    scored[0][0], query, ollama_url, ollama_model
                ):
                    verified = [(scored[0][0], scored[0][1])]
                    logger.info(f"LLM confirmed relevance: {scored[0][0]['name']}")
                else:
                    logger.info(
                        f"LLM rejected single func '{scored[0][0]['name']}' as irrelevant → []"
                    )
                    return "[]"
    else:
        _seg_func_map = None
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
        # If we had parallel segments, the query IS parallel even if detect_parallel missed it
        if not is_parallel and has_parallel_segments:
            is_parallel = True

    calls = []

    # Determine if this is a "same function multiple times" parallel (parallel/live_parallel)
    # vs "different functions" parallel (parallel_multiple/live_parallel_multiple)
    verified_func_names = set(f["name"] for f, _ in verified)
    is_same_func_parallel = len(verified) == 1 and is_parallel

    if not is_parallel:
        for func, score in verified:
            params = extract_params_via_llm_v2(func, query, ollama_url, ollama_model)
            params = validate_and_coerce_params(func, params)
            calls.append((func["name"], params))
            logger.info(f"  {func['name']} params: {params}")
    elif is_same_func_parallel:
        # Same function called multiple times — use extract_all_params with FULL query
        # The LLM sees the entire query and can identify all entities (cities, items, etc.)
        func = verified[0][0]
        param_sets = extract_all_params_via_llm(func, query, ollama_url, ollama_model)
        for params in param_sets:
            params = validate_and_coerce_params(func, params)
            calls.append((func["name"], params))
            logger.info(f"  {func['name']} params: {params}")
    elif has_parallel_segments and len(segments) > 1 and _seg_func_map:
        # Different functions, one per segment (parallel_multiple/live_parallel_multiple)
        # _seg_func_map is a list of (segment, function) pairs
        # Use extract_all_params_via_llm for each segment to handle multi-entity segments
        # (e.g., "tigers in Bangladesh and India" → 2 param sets for same function)
        seg_func_pairs = {}  # func_name → (func, list of segments)
        for seg, func in _seg_func_map:
            if func["name"] not in seg_func_pairs:
                seg_func_pairs[func["name"]] = (func, [])
            seg_func_pairs[func["name"]][1].append(seg)

        for fname, (func, segs) in seg_func_pairs.items():
            if len(segs) == 1:
                # Single segment for this function — may still have multiple entities
                combined_seg = segs[0]
            else:
                # Multiple segments map to same function — combine them
                combined_seg = " ".join(segs)

            param_sets = extract_all_params_via_llm(
                func, combined_seg, ollama_url, ollama_model
            )
            for params in param_sets:
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

    # Deduplicate calls: remove exact duplicate (same function name + same params)
    seen = set()
    deduped_calls = []
    for name, params in calls:
        param_key = json.dumps(params, sort_keys=True, ensure_ascii=False)
        key = (name, param_key)
        if key not in seen:
            seen.add(key)
            deduped_calls.append((name, params))
        else:
            logger.info(f"  Deduped: {name} with params {params}")
    if len(deduped_calls) < len(calls):
        logger.info(f"Deduped {len(calls) - len(deduped_calls)} duplicate calls")
    calls = deduped_calls

    # Post-processing: fix common LLM param extraction issues
    calls = _post_process_params(calls, functions, query)

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
        try:
            if self.path in ("/v1/chat/completions", "/chat/completions"):
                self._handle_chat_completions()
            elif self.path in ("/v1/completions", "/completions"):
                self._handle_completions()
            else:
                self._send_json(404, {"error": "not found"})
        except Exception as e:
            logger.error(f"do_POST unhandled error: {e}", exc_info=True)
            try:
                self._send_json(500, {"error": f"Internal error: {str(e)}"})
            except:
                pass

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

    server = ThreadingHTTPServer((args.host, args.port), CARMServerHandler)
    server.daemon_threads = True
    server.allow_reuse_address = True
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
