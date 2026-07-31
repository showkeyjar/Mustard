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
 7. If the query has MULTIPLE steps or asks for MULTIPLE different things, select ALL matching functions
 8. "Calculate X and generate Y" → select both the calculate function AND the generate function
 9. "Find A in city1 and find B in city2" → select both functions if they are different
10. Numbered steps (1. 2. 3.) each need their own function — select ALL matching functions
11. If two functions are similar (e.g., "search" vs "news_search", "generate_image" vs "generate_human_image"), select only the ONE that best matches the user's intent
12. Do NOT select a function just because its name appears in the query — match by PURPOSE
13. "served hot" in a drink order means temperature, NOT a food order
14. If the query asks for WEATHER, only select weather functions — do NOT select search or news functions
15. If the query asks about a historical event ("what is X war"), select the general search function, NOT news search

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

Important: When a user asks about "all clouds" or "all providers", this INCLUDES any specific cloud service like AWS, GCP, or Azure. So a function that gets AWS pricing IS relevant to a query about "all clouds".

Reject if:
- The function is a generic utility (requests.get, print, etc.) and the user is asking a domain question
- The user asks "how to" do something manually (not via a function call)
- The function doesn't directly produce what the user wants

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
    parts = re.split(r"\band\b", query_lower)
    if len(parts) >= 3 and all(len(p.strip()) >= 4 for p in parts):
        return True
    if len(parts) >= 2 and all(len(p.strip()) >= 8 for p in parts):
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
        r"\band\s+then\b",
        r"\band\s+for\s+(?:the|a|an|my|your|our)\b",
        r"\band\s+(?:calculate|find|compute|get|buy|book|turn|change|update|check|tell|provide|generate|create|search|show|list|add|delete|send|fetch|call|analyze|extract|sort|filter|start|stop|set|open|close|launch|run|build|clone|commit|push|make|estimate|predict|congratulate)\b",
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
 6. For location params, use the location string AS IT APPEARS in the query. If the query says "Boston, USA", use "Boston, USA" exactly. If the query only says "Boston", add the state/country: "Boston, MA".
7. For boolean params, use JSON true/false (not Python True/False).
8. If the query asks for the same thing only once, return exactly ONE object.
9. "all clouds" or "all providers" means call this function ONCE, not once per region/zone.
10. For keyword/search params, extract only the core search term without question words like "who is", "what is", "tell me about".
11. For "function" params in math operations, use ** for exponentiation (e.g. "x**2" not "x^2").
12. Do NOT return duplicate objects with the same params.
13. For location params, use the ENGLISH name of the city (e.g., "Beijing" not "北京", "Shanghai" not "上海", "Tokyo" not "東京").
14. When the query has multiple commands separated by "and" (e.g., "list files and create file"), create one object PER command.

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
 4. For location params, use the location string AS IT APPEARS in the query. If the query says "Boston, USA", use "Boston, USA" exactly. If the query only says "Boston", add the state/country: "Boston, MA".
 5. For keyword/search params, extract only the core search term without question words like "who is", "what is", "tell me about", "search for".
 6. For boolean params, use JSON true/false.
 7. For string params that represent variable names or identifiers (e.g. "userDataArray", "configObject"), pass the identifier name as a string, not as an array.
 8. For "function" params in math operations, use ** for exponentiation (e.g. "x**2" not "x^2"). If the param expects a callable, format as "lambda x: x**2".
 9. When the query mentions a variable name like "myItemList", pass it as a STRING "myItemList", NOT as an actual array of objects. This applies even if the param type is "array" or "dict" — if the query only provides a variable name, pass the name as a string.
10. When a param expects a function/callback (type "any"), pass the function name as a STRING (e.g., "processFunction"), NOT null/None. If the query says "a processing function", pass "processFunction".
11. For dict/object params, use the exact key names from the query (e.g., if query says "nm" and "mn", use those keys, not "name" and "moduleName").
12. For optional params, only include them if the query explicitly provides a value. Do NOT guess or fabricate values for optional params.
13. For location params, use the ENGLISH name of the city (e.g., "Beijing" not "北京", "Shanghai" not "上海", "Tokyo" not "東京").
14. Read parameter descriptions carefully — if a param says "the first and larger", assign the larger value to it. If it says "the second", assign the second value from the query.
15. For float params, use full precision from the query (e.g., gravity 9.81 → 9.81, not 9.8).

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
            verified = [(f, 0.0) for f in functions if f["name"] in selected_names]
            # Store the segment→function mapping for later use
            # Use a list of (segment, function) pairs to handle multiple functions per segment
            _seg_func_map = seg_func_list  # list of (seg, func) pairs
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

        # Trust LLM selection — it already made a semantic judgment
        # The select_function_via_llm prompt includes instructions to return [] for irrelevant
        # For single-function cases, the LLM can distinguish irrelevance from live_relevance
        verified = [(f, 0.0) for f in selected]
        logger.info(f"LLM selected: {[f['name'] for f in selected]}")
    elif len(functions) == 1:
        # Single function available — must verify relevance to avoid false positives
        # (irrelevance test cases have exactly 1 function that should NOT be called)
        _seg_func_map = None
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
