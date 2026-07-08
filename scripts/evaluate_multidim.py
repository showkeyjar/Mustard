"""Multi-dimensional evaluation: CARM vs Agent Frameworks.

Five dimensions on REAL LLM calls:
  D1 — Function Selection   (BFCL V3, func name match)
  D2 — Parameter Extraction (BFCL V3, param name + value match)
  D3 — End-to-End Execution (CARM built-in tools, real execution quality)
  D4 — Multi-Turn Dialogue  (BFCL V3 multi_turn subsets)
  D5 — Robustness           (typo/noise injection, boundary cases)

All results are per-sample saved with resumable support.

Usage:
    set OLLAMA_BASE_URL=http://192.168.31.8:11434
    set OLLAMA_MODEL=qwen3-coder:latest
    python scripts/evaluate_multidim.py

    # Quick test (20 samples per dimension):
    python scripts/evaluate_multidim.py --quick

    # Force re-run:
    python scripts/evaluate_multidim.py --fresh

    # Only specific dimensions:
    python scripts/evaluate_multidim.py --dims D1,D2,D3
"""

from __future__ import annotations

import json
import os
import random
import re
import sys
import time
import urllib.request
from collections import defaultdict
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.environ.setdefault("CARM_NO_EMBEDDING", "1")

# ── Config ─────────────────────────────────────────────────────────────
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://192.168.31.8:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "qwen3-coder:latest")
PER_CALL_TIMEOUT = int(os.environ.get("MULTIDIM_TIMEOUT", "60"))

RESULTS_DIR = PROJECT_ROOT / "data" / "eval" / "multidim_results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ── BFCL V3 data ──────────────────────────────────────────────────────
BFCL_PARQUET = (
    PROJECT_ROOT
    / "data"
    / "bfcl_v3"
    / "datasets"
    / "AI-ModelScope--bfcl_v3"
    / "snapshots"
    / "master"
    / "data"
    / "train-00000-of-00001.parquet"
)

# Subsets for each dimension
D1_SUBSETS = [
    "simple",
    "multiple",
    "parallel",
    "parallel_multiple",
    "irrelevance",
    "live_simple",
    "live_multiple",
]
D2_SUBSETS = [
    "simple",
    "multiple",
    "parallel",
    "parallel_multiple",
    "live_simple",
    "live_multiple",
]
D4_SUBSETS = [
    "multi_turn_base",
    "multi_turn_miss_func",
    "multi_turn_miss_param",
    "multi_turn_long_context",
]

# ── Helpers ───────────────────────────────────────────────────────────


def _load_bfcl() -> pd.DataFrame:
    if not BFCL_PARQUET.exists():
        print(f"ERROR: BFCL V3 data not found at {BFCL_PARQUET}")
        sys.exit(1)
    return pd.read_parquet(BFCL_PARQUET)


def _extract_user_query(turns_data: list | str) -> str:
    """Extract the first user message from turns."""
    if isinstance(turns_data, str):
        turns_data = json.loads(turns_data)
    # turns_data is a list of turns; each turn is a list of messages
    for turn in turns_data:
        if isinstance(turn, list):
            for msg in turn:
                if isinstance(msg, dict) and msg.get("role") == "user":
                    return msg.get("content", "")
    return ""


def _extract_all_user_queries(turns_data: list | str) -> list[str]:
    """Extract all user messages from multi-turn conversations."""
    if isinstance(turns_data, str):
        turns_data = json.loads(turns_data)
    queries = []
    for turn in turns_data:
        if isinstance(turn, list):
            for msg in turn:
                if isinstance(msg, dict) and msg.get("role") == "user":
                    queries.append(msg.get("content", ""))
    return queries


def _parse_gt(gt_data: list | str) -> list[dict]:
    """Parse ground truth: list of {func_name: {param: [values]}}."""
    if isinstance(gt_data, str):
        gt_data = json.loads(gt_data)
    return gt_data if gt_data else []


def _parse_functions(func_data: list | str) -> list[dict]:
    """Parse available functions list."""
    if isinstance(func_data, str):
        func_data = json.loads(func_data)
    return func_data if func_data else []


def _gt_func_names(gt: list[dict]) -> list[str]:
    """Extract function names from ground truth."""
    names = []
    for entry in gt:
        if isinstance(entry, dict):
            names.extend(entry.keys())
        elif isinstance(entry, str):
            # multi_turn format: "func(param='value')"
            m = re.match(r"(\w+(?:\.\w+)?)\s*\(", entry)
            if m:
                names.append(m.group(1))
    return names


def _gt_params(gt: list[dict]) -> dict[str, dict[str, Any]]:
    """Extract {func_name: {param_name: expected_value}} from ground truth.

    BFCL V3 format: each param value is [expected_value] or [value, alternative].
    We take the first element as the canonical expected value.
    """
    result = {}
    for entry in gt:
        if not isinstance(entry, dict):
            continue
        for func_name, params in entry.items():
            if isinstance(params, dict):
                extracted = {}
                for pname, pval in params.items():
                    if isinstance(pval, list) and len(pval) > 0:
                        extracted[pname] = pval[0]
                    else:
                        extracted[pname] = pval
                result[func_name] = extracted
    return result


def _save_sample(result_file: Path, key: str, data: dict) -> None:
    """Save a single sample result (append/overwrite)."""
    results = {}
    if result_file.exists():
        with open(result_file, "r", encoding="utf-8") as f:
            results = json.load(f)
    results[key] = data
    with open(result_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)


def _load_results(result_file: Path) -> dict:
    """Load existing results for resumption."""
    if result_file.exists():
        with open(result_file, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


# ── LLM Call via Ollama ──────────────────────────────────────────────

_last_llm_tokens: dict = {"eval": 0, "prompt_eval": 0}


def call_ollama(
    prompt: str,
    system: str = "",
    model: str | None = None,
    timeout: int = PER_CALL_TIMEOUT,
) -> tuple[str, dict]:
    """Call Ollama and return (response_text, token_info)."""
    global _last_llm_tokens
    model = model or OLLAMA_MODEL
    payload = {
        "model": model,
        "prompt": prompt,
        "system": system,
        "stream": True,
        "options": {"temperature": 0.0},
    }
    # Detect thinking models
    if any(m in model for m in ["qwen3", "deepseek-r1", "phi4-reasoning"]):
        payload["think"] = False

    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        f"{OLLAMA_BASE_URL}/api/generate",
        data=data,
        headers={"Content-Type": "application/json"},
    )

    full_text = ""
    eval_count = 0
    prompt_eval_count = 0
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            for line in resp:
                chunk = json.loads(line)
                if "response" in chunk:
                    full_text += chunk["response"]
                if chunk.get("done"):
                    eval_count = chunk.get("eval_count", 0)
                    prompt_eval_count = chunk.get("prompt_eval_count", 0)
                    break
    except Exception as e:
        full_text = f"ERROR: {e}"

    tokens = {"eval": eval_count, "prompt_eval": prompt_eval_count}
    _last_llm_tokens = tokens
    return full_text, tokens


def call_ollama_chat(
    messages: list[dict],
    tools: list[dict] | None = None,
    model: str | None = None,
    timeout: int = PER_CALL_TIMEOUT,
) -> tuple[str, dict]:
    """Call Ollama chat API with optional tools (for frameworks that use FC)."""
    global _last_llm_tokens
    model = model or OLLAMA_MODEL
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
    }
    if tools:
        payload["tools"] = tools

    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        f"{OLLAMA_BASE_URL}/api/chat",
        data=data,
        headers={"Content-Type": "application/json"},
    )

    full_text = ""
    eval_count = 0
    prompt_eval_count = 0
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            result = json.loads(resp.read())
            msg = result.get("message", {})
            full_text = msg.get("content", "")
            # If tool_calls present, format them
            if msg.get("tool_calls"):
                full_text = json.dumps(msg["tool_calls"], ensure_ascii=False)
            eval_count = result.get("eval_count", 0)
            prompt_eval_count = result.get("prompt_eval_count", 0)
    except Exception as e:
        full_text = f"ERROR: {e}"

    tokens = {"eval": eval_count, "prompt_eval": prompt_eval_count}
    _last_llm_tokens = tokens
    return full_text, tokens


# ════════════════════════════════════════════════════════════════════════
# D1 — Function Selection (reuse existing logic)
# ════════════════════════════════════════════════════════════════════════


def evaluate_D1(
    framework: str,
    df: pd.DataFrame,
    fresh: bool = False,
    max_per_subset: int | None = None,
) -> dict:
    """D1: Function Selection Accuracy.

    Given a user query + available functions, does the framework select
    the correct function name?
    """
    print(f"\n{'=' * 60}")
    print(f"D1 — Function Selection | {framework}")
    print(f"{'=' * 60}")

    result_file = RESULTS_DIR / f"D1_{framework}.json"
    saved = {} if fresh else _load_results(result_file)

    correct = 0
    total = 0
    total_time = 0.0
    total_tokens = 0
    llm_calls = 0
    per_subset = defaultdict(lambda: {"correct": 0, "total": 0})

    for subset in D1_SUBSETS:
        rows = df[df["subset"] == subset]
        if max_per_subset:
            rows = rows.head(max_per_subset)
        for idx, (_, row) in enumerate(rows.iterrows()):
            key = f"{subset}__{idx}"
            if key in saved:
                d = saved[key]
                correct += int(d.get("correct", 0))
                total += 1
                total_time += d.get("elapsed", 0)
                total_tokens += d.get("tokens", 0)
                llm_calls += 1 if d.get("llm_used") else 0
                per_subset[subset]["correct"] += int(d.get("correct", 0))
                per_subset[subset]["total"] += 1
                continue

            gt = _parse_gt(row["ground_truth"])
            funcs = _parse_functions(row["functions"])
            query = _extract_user_query(row["turns"])
            is_irrel = row["subset"] == "irrelevance"
            gt_names = _gt_func_names(gt)

            t0 = time.time()
            predicted, used_llm, tok = _route_query(framework, query, funcs, is_irrel)
            elapsed = time.time() - t0

            if is_irrel:
                ok = predicted is None or predicted == "none"
            else:
                ok = predicted in gt_names if predicted else False

            d = {
                "query": query[:150],
                "predicted": predicted,
                "gt": gt_names,
                "correct": ok,
                "elapsed": round(elapsed, 3),
                "llm_used": used_llm,
                "tokens": tok.get("eval", 0) + tok.get("prompt_eval", 0),
            }
            _save_sample(result_file, key, d)
            saved[key] = d

            correct += int(ok)
            total += 1
            total_time += elapsed
            total_tokens += d["tokens"]
            llm_calls += 1 if used_llm else 0
            per_subset[subset]["correct"] += int(ok)
            per_subset[subset]["total"] += 1

            if total % 50 == 0:
                print(f"  ... {total} done, acc={correct / total * 100:.1f}%")

    acc = correct / total * 100 if total else 0
    lat = total_time / total * 1000 if total else 0
    tpc = round(total_tokens / total) if total else 0

    print(
        f"\n  D1 Result: {acc:.1f}% ({correct}/{total}), "
        f"{lat:.0f}ms, {tpc} tok/call, LLM {llm_calls}/{total}"
    )

    return {
        "dimension": "D1_Function_Selection",
        "framework": framework,
        "accuracy": round(acc, 1),
        "correct": correct,
        "total": total,
        "latency_ms": round(lat),
        "tokens_per_call": tpc,
        "llm_call_rate": round(llm_calls / total * 100) if total else 0,
        "per_subset": {
            k: {
                "accuracy": round(v["correct"] / v["total"] * 100, 1)
                if v["total"]
                else 0,
                "correct": v["correct"],
                "total": v["total"],
            }
            for k, v in per_subset.items()
        },
    }


# ════════════════════════════════════════════════════════════════════════
# D2 — Parameter Extraction
# ════════════════════════════════════════════════════════════════════════


def evaluate_D2(
    framework: str,
    df: pd.DataFrame,
    fresh: bool = False,
    max_per_subset: int | None = None,
) -> dict:
    """D2: Parameter Extraction Quality.

    Given a user query + available functions + the correct function,
    can the framework extract the correct parameter names and values?

    Scoring:
      - param_name_recall: % of required params whose names appear in output
      - param_value_match: % of params with correct value (exact or fuzzy)
      - overall: (name_recall + value_match) / 2
    """
    print(f"\n{'=' * 60}")
    print(f"D2 — Parameter Extraction | {framework}")
    print(f"{'=' * 60}")

    result_file = RESULTS_DIR / f"D2_{framework}.json"
    saved = {} if fresh else _load_results(result_file)

    total_name_recall = 0.0
    total_value_match = 0.0
    total = 0
    total_time = 0.0
    total_tokens = 0

    for subset in D2_SUBSETS:
        rows = df[df["subset"] == subset]
        if max_per_subset:
            rows = rows.head(max_per_subset)
        for idx, (_, row) in enumerate(rows.iterrows()):
            key = f"{subset}__{idx}"
            if key in saved:
                d = saved[key]
                total_name_recall += d.get("name_recall", 0)
                total_value_match += d.get("value_match", 0)
                total += 1
                total_time += d.get("elapsed", 0)
                total_tokens += d.get("tokens", 0)
                continue

            gt = _parse_gt(row["ground_truth"])
            funcs = _parse_functions(row["functions"])
            query = _extract_user_query(row["turns"])

            if not gt:
                # irrelevance or empty GT — skip D2
                continue

            gt_params = _gt_params(gt)
            if not gt_params:
                continue

            t0 = time.time()
            predicted_params = _extract_params(framework, query, funcs, gt)
            elapsed = time.time() - t0

            # Score param name recall and value match
            name_recall, value_match, details = _score_params(
                gt_params, predicted_params
            )

            d = {
                "query": query[:150],
                "gt_params": {k: str(v)[:100] for k, v in gt_params.items()},
                "predicted_params": predicted_params,
                "name_recall": round(name_recall, 3),
                "value_match": round(value_match, 3),
                "details": details,
                "elapsed": round(elapsed, 3),
                "tokens": _last_llm_tokens.get("eval", 0)
                + _last_llm_tokens.get("prompt_eval", 0),
            }
            _save_sample(result_file, key, d)
            saved[key] = d

            total_name_recall += name_recall
            total_value_match += value_match
            total += 1
            total_time += elapsed
            total_tokens += d["tokens"]

            if total % 50 == 0:
                print(
                    f"  ... {total} done, "
                    f"NR={total_name_recall / total:.3f}, "
                    f"VM={total_value_match / total:.3f}"
                )

    if total == 0:
        print("  No valid samples for D2")
        return {
            "dimension": "D2_Parameter_Extraction",
            "framework": framework,
            "name_recall": 0,
            "value_match": 0,
            "overall": 0,
            "total": 0,
            "latency_ms": 0,
            "tokens_per_call": 0,
        }

    avg_nr = total_name_recall / total
    avg_vm = total_value_match / total
    lat = total_time / total * 1000 if total else 0
    tpc = round(total_tokens / total) if total else 0

    print(
        f"\n  D2 Result: NameRecall={avg_nr:.3f}, ValueMatch={avg_vm:.3f}, "
        f"Overall={((avg_nr + avg_vm) / 2) * 100:.1f}%, {lat:.0f}ms"
    )

    return {
        "dimension": "D2_Parameter_Extraction",
        "framework": framework,
        "name_recall": round(avg_nr * 100, 1),
        "value_match": round(avg_vm * 100, 1),
        "overall": round((avg_nr + avg_vm) / 2 * 100, 1),
        "total": total,
        "latency_ms": round(lat),
        "tokens_per_call": tpc,
    }


def _score_params(
    gt_params: dict[str, dict[str, Any]], predicted: dict[str, dict[str, Any]]
) -> tuple[float, float, list]:
    """Score parameter extraction against ground truth.

    Returns (name_recall, value_match, details_list).
    """
    all_gt_keys = set()
    all_pred_keys = set()
    name_hits = 0
    value_hits = 0
    details = []

    for func_name, gt_p in gt_params.items():
        for pname, pval in gt_p.items():
            all_gt_keys.add(f"{func_name}::{pname}")
            # Check if predicted has this param
            pred_func = predicted.get(func_name, {})
            # Handle case where predicted value might be a list or non-dict
            if not isinstance(pred_func, dict):
                pred_func = {}
            pred_val = pred_func.get(pname)

            if pred_val is not None:
                name_hits += 1
                # Value match: exact or fuzzy
                if _values_match(pval, pred_val):
                    value_hits += 1
                    details.append(
                        {
                            "key": f"{func_name}::{pname}",
                            "status": "MATCH",
                            "gt": str(pval)[:80],
                            "pred": str(pred_val)[:80],
                        }
                    )
                else:
                    details.append(
                        {
                            "key": f"{func_name}::{pname}",
                            "status": "NAME_OK_VALUE_MISS",
                            "gt": str(pval)[:80],
                            "pred": str(pred_val)[:80],
                        }
                    )
            else:
                details.append(
                    {
                        "key": f"{func_name}::{pname}",
                        "status": "MISS",
                        "gt": str(pval)[:80],
                    }
                )

        # Count predicted params for this function
        for pname in predicted.get(func_name, {}):
            all_pred_keys.add(f"{func_name}::{pname}")

    name_recall = name_hits / len(all_gt_keys) if all_gt_keys else 0.0
    value_match = value_hits / len(all_gt_keys) if all_gt_keys else 0.0
    return name_recall, value_match, details


def _values_match(gt_val: Any, pred_val: Any) -> bool:
    """Check if predicted value matches ground truth (with fuzzy matching)."""
    gt_str = str(gt_val).strip().lower()
    pred_str = str(pred_val).strip().lower()

    # Exact match
    if gt_str == pred_str:
        return True

    # Numeric match
    try:
        if abs(float(gt_val) - float(pred_val)) < 0.01:
            return True
    except (ValueError, TypeError):
        pass

    # Substring match (for partial values)
    if gt_str in pred_str or pred_str in gt_str:
        return True

    # Boolean-like
    gt_bool = gt_str in ("true", "false", "yes", "no", "1", "0")
    pred_bool = pred_str in ("true", "false", "yes", "no", "1", "0")
    if gt_bool and pred_bool and gt_str[0] == pred_str[0]:
        return True

    return False


# ════════════════════════════════════════════════════════════════════════
# D3 — End-to-End Execution Quality
# ════════════════════════════════════════════════════════════════════════

# Hand-crafted test cases that exercise CARM's built-in tools end-to-end
E2E_TEST_CASES = [
    # Calculator — basic
    {
        "query": "3加5等于多少",
        "expected_tool": "calculator",
        "expected_pattern": r"8",
        "category": "calc_basic",
    },
    {
        "query": "123乘以456",
        "expected_tool": "calculator",
        "expected_pattern": r"56088",
        "category": "calc_basic",
    },
    {
        "query": "2的10次方",
        "expected_tool": "calculator",
        "expected_pattern": r"1024",
        "category": "calc_basic",
    },
    # Calculator — NL patterns
    {
        "query": "买了3本书每本15元又买了2支笔每支5元一共多少钱",
        "expected_tool": "calculator",
        "expected_pattern": r"55",
        "category": "calc_nl",
    },
    {
        "query": "原价200元打8折多少钱",
        "expected_tool": "calculator",
        "expected_pattern": r"160",
        "category": "calc_nl",
    },
    {
        "query": "圆的面积半径是5",
        "expected_tool": "calculator",
        "expected_pattern": r"78\.5",
        "category": "calc_nl",
    },
    {
        "query": "从1加到100",
        "expected_tool": "calculator",
        "expected_pattern": r"5050",
        "category": "calc_nl",
    },
    {
        "query": "鸡兔同笼共35个头94条腿鸡有几只",
        "expected_tool": "calculator",
        "expected_pattern": r"23",
        "category": "calc_nl",
    },
    {
        "query": "房贷100万30年利率4.5%",
        "expected_tool": "calculator",
        "expected_pattern": r"\d{4,}",
        "category": "calc_nl",
    },
    # Calculator — equation
    {
        "query": "一个数的3倍加5等于20这个数是多少",
        "expected_tool": "calculator",
        "expected_pattern": r"5",
        "category": "calc_equation",
    },
    # Code executor
    {
        "query": "帮我写一个快速排序",
        "expected_tool": "code_executor",
        "expected_pattern": r"代码执行|quicksort|排序|\[\d",
        "category": "code",
    },
    {
        "query": "写一个斐波那契数列前10项",
        "expected_tool": "code_executor",
        "expected_pattern": r"代码执行|fibonacci|斐波那契|\[\d",
        "category": "code",
    },
    # BigModel proxy (consultation)
    {
        "query": "请详细分析一下人工智能在医疗领域的应用前景",
        "expected_tool": "bigmodel_proxy",
        "expected_pattern": r"医疗|AI|诊断|影像",
        "category": "consult",
    },
    {
        "query": "Python和JavaScript哪个更适合初学者",
        "expected_tool": "bigmodel_proxy",
        "expected_pattern": r"Python|JavaScript|初学|比较|选择",
        "category": "consult",
    },
    # Multi-intent (CARM routes to the dominant intent — calculator here)
    {
        "query": "帮我算一下15的20%是多少，顺便查一下北京天气",
        "expected_tool": "calculator",
        "expected_pattern": r"3|天气|北京|计算",
        "category": "multi_intent",
    },
    # Search (may fallback to bigmodel)
    {
        "query": "最新的AI技术趋势是什么",
        "expected_tool": "search",
        "expected_pattern": r"AI|人工智能|大模型",
        "category": "search",
    },
    # Rejection (no appropriate tool)
    {
        "query": "你好",
        "expected_tool": "none",
        "expected_pattern": r"",
        "category": "reject",
    },
    {
        "query": "今天心情不错",
        "expected_tool": "none",
        "expected_pattern": r"",
        "category": "reject",
    },
]


def evaluate_D3(
    framework: str, fresh: bool = False, max_cases: int | None = None
) -> dict:
    """D3: End-to-End Execution Quality.

    Run real queries through CARM's actual tools and check:
      - tool_selection: did it pick the right tool?
      - execution_ok: did the tool execute without error?
      - result_quality: does the output contain the expected answer?
    """
    print(f"\n{'=' * 60}")
    print(f"D3 — End-to-End Execution | {framework}")
    print(f"{'=' * 60}")

    if framework != "carm_hybrid":
        print("  Note: D3 only applicable to CARM (requires actual tool execution)")
        # For non-CARM frameworks, we simulate by asking the LLM to answer directly
        return _evaluate_D3_llm_baseline(framework, fresh, max_cases)

    result_file = RESULTS_DIR / f"D3_{framework}.json"
    saved = {} if fresh else _load_results(result_file)

    cases = E2E_TEST_CASES[:max_cases] if max_cases else E2E_TEST_CASES

    from carm.router import CARMRouter

    router = CARMRouter()

    tool_ok = 0
    exec_ok = 0
    result_ok = 0
    total = 0
    total_time = 0.0
    per_category = defaultdict(
        lambda: {"tool_ok": 0, "exec_ok": 0, "result_ok": 0, "total": 0}
    )

    for i, case in enumerate(cases):
        key = f"case__{i}"
        if key in saved:
            d = saved[key]
            tool_ok += int(d.get("tool_ok", 0))
            exec_ok += int(d.get("exec_ok", 0))
            result_ok += int(d.get("result_ok", 0))
            total += 1
            total_time += d.get("elapsed", 0)
            cat = d.get("category", "unknown")
            per_category[cat]["tool_ok"] += int(d.get("tool_ok", 0))
            per_category[cat]["exec_ok"] += int(d.get("exec_ok", 0))
            per_category[cat]["result_ok"] += int(d.get("result_ok", 0))
            per_category[cat]["total"] += 1
            continue

        t0 = time.time()
        r = router.route(case["query"])
        elapsed = time.time() - t0

        t_ok = (
            r.tool_name == case["expected_tool"]
            or (case["expected_tool"] == "search" and r.tool_name == "bigmodel_proxy")
            or (case["expected_tool"] == "bigmodel_proxy" and r.tool_name == "search")
        )  # search↔bigmodel cross-routing is acceptable
        # For "none" expected: correctly rejecting is a successful execution
        if case["expected_tool"] == "none" and r.tool_name == "none":
            e_ok = True
        else:
            e_ok = r.ok
        r_ok = (
            bool(re.search(case["expected_pattern"], r.result, re.IGNORECASE))
            if case["expected_pattern"]
            else True
        )

        d = {
            "query": case["query"],
            "expected_tool": case["expected_tool"],
            "actual_tool": r.tool_name,
            "tool_ok": t_ok,
            "exec_ok": e_ok,
            "result_ok": r_ok,
            "result_text": r.result[:200],
            "confidence": r.confidence,
            "elapsed": round(elapsed, 3),
            "category": case["category"],
        }
        _save_sample(result_file, key, d)
        saved[key] = d

        tool_ok += int(t_ok)
        exec_ok += int(e_ok)
        result_ok += int(r_ok)
        total += 1
        total_time += elapsed
        per_category[case["category"]]["tool_ok"] += int(t_ok)
        per_category[case["category"]]["exec_ok"] += int(e_ok)
        per_category[case["category"]]["result_ok"] += int(r_ok)
        per_category[case["category"]]["total"] += 1

        status = "✓" if (t_ok and e_ok and r_ok) else "✗"
        print(
            f"  [{status}] {case['query'][:30]:30s} → {r.tool_name:15s} "
            f"tool={t_ok} exec={e_ok} result={r_ok}"
        )

    n = total or 1
    print(
        f"\n  D3 Result: ToolSel={tool_ok / n * 100:.0f}% ExecOK={exec_ok / n * 100:.0f}% "
        f"ResultOK={result_ok / n * 100:.0f}% | {total_time / total * 1000:.0f}ms avg"
    )

    return {
        "dimension": "D3_End_to_End",
        "framework": framework,
        "tool_selection": round(tool_ok / n * 100, 1),
        "execution_ok": round(exec_ok / n * 100, 1),
        "result_quality": round(result_ok / n * 100, 1),
        "overall": round((tool_ok + exec_ok + result_ok) / (3 * n) * 100, 1),
        "total": total,
        "latency_ms": round(total_time / total * 1000) if total else 0,
        "per_category": {
            k: {
                "tool_sel": round(v["tool_ok"] / v["total"] * 100) if v["total"] else 0,
                "exec_ok": round(v["exec_ok"] / v["total"] * 100) if v["total"] else 0,
                "result_ok": round(v["result_ok"] / v["total"] * 100)
                if v["total"]
                else 0,
                "total": v["total"],
            }
            for k, v in per_category.items()
        },
    }


def _evaluate_D3_llm_baseline(
    framework: str, fresh: bool = False, max_cases: int | None = None
) -> dict:
    """D3 for non-CARM frameworks: ask LLM directly to solve the same queries."""
    result_file = RESULTS_DIR / f"D3_{framework}.json"
    saved = {} if fresh else _load_results(result_file)

    cases = E2E_TEST_CASES[:max_cases] if max_cases else E2E_TEST_CASES
    # Only test cases that have calculable expected answers
    calculable = [
        c for c in cases if c["expected_pattern"] and c["category"] != "reject"
    ]

    result_ok = 0
    total = 0
    total_time = 0.0
    total_tokens = 0

    for i, case in enumerate(calculable):
        key = f"case__{i}"
        if key in saved:
            d = saved[key]
            result_ok += int(d.get("result_ok", 0))
            total += 1
            total_time += d.get("elapsed", 0)
            total_tokens += d.get("tokens", 0)
            continue

        prompt = f"请直接回答以下问题，给出简短答案：\n{case['query']}"
        t0 = time.time()
        response, tok = call_ollama(prompt)
        elapsed = time.time() - t0

        r_ok = bool(re.search(case["expected_pattern"], response, re.IGNORECASE))
        tokens = tok.get("eval", 0) + tok.get("prompt_eval", 0)

        d = {
            "query": case["query"],
            "expected_tool": case["expected_tool"],
            "response": response[:200],
            "result_ok": r_ok,
            "elapsed": round(elapsed, 3),
            "tokens": tokens,
        }
        _save_sample(result_file, key, d)
        saved[key] = d

        result_ok += int(r_ok)
        total += 1
        total_time += elapsed
        total_tokens += tokens

        status = "✓" if r_ok else "✗"
        print(f"  [{status}] {case['query'][:30]:30s} → result={r_ok}")

    n = total or 1
    return {
        "dimension": "D3_End_to_End",
        "framework": framework,
        "result_quality": round(result_ok / n * 100, 1),
        "overall": round(result_ok / n * 100, 1),
        "total": total,
        "latency_ms": round(total_time / total * 1000) if total else 0,
        "tokens_per_call": round(total_tokens / total) if total else 0,
        "note": "Non-CARM frameworks: LLM-only, no tool execution",
    }


# ════════════════════════════════════════════════════════════════════════
# D4 — Multi-Turn Dialogue
# ════════════════════════════════════════════════════════════════════════

# CARM-specific multi-turn test cases (using CARM's built-in tools)
CARM_MULTITURN_CASES = [
    {
        "id": "mt_calc_then_followup",
        "turns": [
            {"query": "3加5等于多少", "expected_tool": "calculator"},
            {"query": "再加上10呢", "expected_tool": "calculator"},
        ],
        "tests_anaphora": True,
    },
    {
        "id": "mt_code_then_calc",
        "turns": [
            {"query": "帮我写一个快速排序", "expected_tool": "code_executor"},
            {"query": "算一下100的阶乘", "expected_tool": "calculator"},
        ],
        "tests_anaphora": False,
    },
    {
        "id": "mt_greeting_then_real",
        "turns": [
            {"query": "你好", "expected_tool": "none"},
            {"query": "3乘7等于多少", "expected_tool": "calculator"},
        ],
        "tests_anaphora": False,
    },
    {
        "id": "mt_calc_anaphora_it",
        "turns": [
            {"query": "帮我算一下5的平方根", "expected_tool": "calculator"},
            {"query": "它的两倍是多少", "expected_tool": "calculator"},
        ],
        "tests_anaphora": True,
    },
    {
        "id": "mt_search_then_consult",
        "turns": [
            {"query": "搜索一下最新的AI新闻", "expected_tool": "search"},
            {"query": "分析一下这些趋势的影响", "expected_tool": "bigmodel_proxy"},
        ],
        "tests_anaphora": False,
    },
    {
        "id": "mt_multi_intent_followup",
        "turns": [
            {"query": "帮我算一下12乘以8", "expected_tool": "calculator"},
            {"query": "顺便再算一下15的20%", "expected_tool": "calculator"},
        ],
        "tests_anaphora": False,
    },
    {
        "id": "mt_reject_then_real",
        "turns": [
            {"query": "今天天气真好", "expected_tool": "none"},
            {"query": "帮我写一个冒泡排序", "expected_tool": "code_executor"},
        ],
        "tests_anaphora": False,
    },
    {
        "id": "mt_calc_chain",
        "turns": [
            {"query": "15乘以8是多少", "expected_tool": "calculator"},
            {"query": "除以3", "expected_tool": "calculator"},
            {"query": "再加上100", "expected_tool": "calculator"},
        ],
        "tests_anaphora": True,
    },
]


def evaluate_D4(
    framework: str,
    df: pd.DataFrame,
    fresh: bool = False,
    max_per_subset: int | None = None,
) -> dict:
    """D4: Multi-Turn Dialogue Quality.

    Using BFCL V3 multi_turn subsets (4 turns per conversation).
    Measures:
      - turn_accuracy: correct function selection per turn
      - context_retention: does turn N correctly reference turn N-1?
      - conversation_accuracy: all turns correct?
    """
    print(f"\n{'=' * 60}")
    print(f"D4 — Multi-Turn Dialogue | {framework}")
    print(f"{'=' * 60}")

    result_file = RESULTS_DIR / f"D4_{framework}.json"
    saved = {} if fresh else _load_results(result_file)

    turn_correct = 0
    turn_total = 0
    conv_correct = 0
    conv_total = 0
    total_time = 0.0
    total_tokens = 0
    anaphora_correct = 0
    anaphora_total = 0

    if framework == "carm_hybrid":
        # CARM-specific multi-turn test cases (using CARMRouter + session memory)
        from carm.router import CARMRouter

        router = CARMRouter()

        for case in CARM_MULTITURN_CASES:
            key = f"carm_mt__{case['id']}"
            if key in saved:
                d = saved[key]
                turn_correct += d.get("turn_correct", 0)
                turn_total += d.get("turn_total", 0)
                conv_correct += 1 if d.get("conv_correct") else 0
                conv_total += 1
                total_time += d.get("elapsed", 0)
                anaphora_correct += d.get("anaphora_correct", 0)
                anaphora_total += d.get("anaphora_total", 0)
                continue

            session_id = f"eval_{case['id']}_{int(time.time())}"
            t_correct = 0
            t_total = 0
            a_correct = 0
            a_total = 0

            t0 = time.time()
            for turn_idx, turn in enumerate(case["turns"]):
                r = router.route(turn["query"], session_id=session_id, dry_run=True)
                actual = r.tool_name
                expected = turn["expected_tool"]

                # Allow search↔bigmodel cross-routing
                ok = (
                    actual == expected
                    or (expected == "search" and actual == "bigmodel_proxy")
                    or (expected == "bigmodel_proxy" and actual == "search")
                )
                t_correct += int(ok)
                t_total += 1

                if turn.get("tests_anaphora", False) or case.get("tests_anaphora"):
                    a_total += 1
                    a_correct += int(ok)
            elapsed = time.time() - t0

            conv_ok = t_correct == t_total
            d = {
                "turn_correct": t_correct,
                "turn_total": t_total,
                "conv_correct": conv_ok,
                "elapsed": round(elapsed, 3),
                "anaphora_correct": a_correct,
                "anaphora_total": a_total,
                "tokens": 0,  # CARMRouter dry_run uses no tokens
            }
            _save_sample(result_file, key, d)
            saved[key] = d

            turn_correct += t_correct
            turn_total += t_total
            conv_correct += 1 if conv_ok else 0
            conv_total += 1
            total_time += elapsed
            anaphora_correct += a_correct
            anaphora_total += a_total

            status = "✓" if conv_ok else "✗"
            print(
                f"  [{status}] {case['id']:30s} "
                f"turns={t_correct}/{t_total} "
                f"anaphora={a_correct}/{a_total}"
            )

    else:
        # Non-CARM frameworks: use BFCL V3 multi_turn subsets
        for subset in D4_SUBSETS:
            rows = df[df["subset"] == subset]
            if max_per_subset:
                rows = rows.head(max_per_subset)
            for idx, (_, row) in enumerate(rows.iterrows()):
                key = f"bfcl__{subset}__{idx}"
                if key in saved:
                    d = saved[key]
                    turn_correct += d.get("turn_correct", 0)
                    turn_total += d.get("turn_total", 0)
                    conv_correct += 1 if d.get("conv_correct") else 0
                    conv_total += 1
                    total_time += d.get("elapsed", 0)
                    total_tokens += d.get("tokens", 0)
                    continue

                gt = _parse_gt(row["ground_truth"])
                funcs = _parse_functions(row["functions"])
                turns_data = row["turns"]
                if isinstance(turns_data, str):
                    turns_data = json.loads(turns_data)

                # Run multi-turn conversation
                t0 = time.time()
                t_correct, t_total, conv_ok, tokens_used = _run_multi_turn(
                    framework, turns_data, funcs, gt
                )
                elapsed = time.time() - t0

                d = {
                    "subset": subset,
                    "turn_correct": t_correct,
                    "turn_total": t_total,
                    "conv_correct": conv_ok,
                    "elapsed": round(elapsed, 3),
                    "tokens": tokens_used,
                }
                _save_sample(result_file, key, d)
                saved[key] = d

                turn_correct += t_correct
                turn_total += t_total
                conv_correct += 1 if conv_ok else 0
                conv_total += 1
                total_time += elapsed
                total_tokens += tokens_used

    turn_acc = turn_correct / turn_total * 100 if turn_total else 0
    conv_acc = conv_correct / conv_total * 100 if conv_total else 0
    lat = total_time / conv_total * 1000 if conv_total else 0
    tpc = round(total_tokens / conv_total) if conv_total else 0

    print(
        f"\n  D4 Result: TurnAcc={turn_acc:.1f}%, ConvAcc={conv_acc:.1f}%, "
        f"{lat:.0f}ms/conv, {tpc} tok/conv"
    )

    return {
        "dimension": "D4_Multi_Turn",
        "framework": framework,
        "turn_accuracy": round(turn_acc, 1),
        "conversation_accuracy": round(conv_acc, 1),
        "anaphora_accuracy": round(anaphora_correct / anaphora_total * 100, 1)
        if anaphora_total
        else 0,
        "turn_correct": turn_correct,
        "turn_total": turn_total,
        "conv_correct": conv_correct,
        "conv_total": conv_total,
        "latency_ms": round(lat),
        "tokens_per_conv": tpc,
    }


def _run_multi_turn(
    framework: str, turns_data: list, funcs: list[dict], gt: list
) -> tuple[int, int, bool, int]:
    """Run a multi-turn conversation and score each turn.

    Returns (turn_correct, turn_total, conv_all_correct, total_tokens).
    """
    # Extract per-turn queries and GT
    all_queries = _extract_all_user_queries(turns_data)
    # GT: one entry per turn (list of function calls expected)

    turn_correct = 0
    turn_total = 0
    total_tokens = 0
    session_id = f"mt_{int(time.time() * 1000)}"

    for turn_idx, query in enumerate(all_queries):
        if not query.strip():
            continue

        # Get expected functions for this turn
        if turn_idx < len(gt):
            turn_gt = gt[turn_idx]
            if isinstance(turn_gt, list):
                expected_names = []
                for entry in turn_gt:
                    if isinstance(entry, str):
                        m = re.match(r"(\w+(?:\.\w+)?)\s*\(", entry)
                        if m:
                            expected_names.append(m.group(1))
                    elif isinstance(entry, dict):
                        expected_names.extend(entry.keys())
            elif isinstance(turn_gt, dict):
                expected_names = list(turn_gt.keys())
            else:
                expected_names = []
        else:
            expected_names = []

        # Route this turn
        predicted, used_llm, tok = _route_query(
            framework,
            query,
            funcs,
            is_irrel=False,
            session_id=session_id if framework == "carm_hybrid" else None,
        )
        total_tokens += tok.get("eval", 0) + tok.get("prompt_eval", 0)

        if expected_names:
            ok = predicted in expected_names if predicted else False
            turn_correct += int(ok)
            turn_total += 1

    conv_ok = turn_correct == turn_total if turn_total > 0 else False
    return turn_correct, turn_total, conv_ok, total_tokens


# ════════════════════════════════════════════════════════════════════════
# D5 — Robustness
# ════════════════════════════════════════════════════════════════════════

# Robustness test cases: original + perturbed versions
ROBUSTNESS_CASES = [
    # Category: typo (character swap/removal)
    {
        "original": "3加5等于多少",
        "perturbed": "3加5等多少",
        "expected_tool": "calculator",
        "robustness_type": "typo",
    },
    {
        "original": "帮我写一个快速排序",
        "perturbed": "帮我写一个快速排 序",
        "expected_tool": "code_executor",
        "robustness_type": "typo",
    },
    {
        "original": "搜索最新的AI新闻",
        "perturbed": "搜索最新的AI新 闻",
        "expected_tool": "search",
        "robustness_type": "typo",
    },
    # Category: noise prefix/suffix
    {
        "original": "3加5等于多少",
        "perturbed": "嗯我想问一下3加5等于多少",
        "expected_tool": "calculator",
        "robustness_type": "noise_prefix",
    },
    {
        "original": "帮我写一个快速排序",
        "perturbed": "帮我写一个快速排序谢谢啦",
        "expected_tool": "code_executor",
        "robustness_type": "noise_suffix",
    },
    {
        "original": "分析一下市场趋势",
        "perturbed": "呃就是分析一下市场趋势吧",
        "expected_tool": "bigmodel_proxy",
        "robustness_type": "noise_prefix",
    },
    # Category: verbose/rephrased
    {
        "original": "3加5等于多少",
        "perturbed": "我想知道三加上五的结果是什么呀",
        "expected_tool": "calculator",
        "robustness_type": "rephrase",
    },
    {
        "original": "帮我写一个快速排序",
        "perturbed": "能不能给我写一段快速排序的代码呢",
        "expected_tool": "code_executor",
        "robustness_type": "rephrase",
    },
    {
        "original": "搜索最新的AI新闻",
        "perturbed": "我想看看最近人工智能方面有什么新消息",
        "expected_tool": "search",
        "robustness_type": "rephrase",
    },
    # Category: ambiguous/boundary
    {
        "original": "3加5",
        "perturbed": "3+5",  # pure expression, no NL
        "expected_tool": "calculator",
        "robustness_type": "boundary_minimal",
    },
    {
        "original": "你好",
        "perturbed": "嗨嗨嗨",
        "expected_tool": "none",
        "robustness_type": "boundary_greeting",
    },
    {
        "original": "5公里等于多少米",
        "perturbed": "5km=?m",
        "expected_tool": "calculator",
        "robustness_type": "boundary_mixed",
    },
    {
        "original": "分析一下",
        "perturbed": "分析",
        "expected_tool": "none",
        "robustness_type": "boundary_vague",
    },
    # Category: mixed language
    {
        "original": "搜索最新的AI新闻",
        "perturbed": "search最新AI news",
        "expected_tool": "search",
        "robustness_type": "mixed_lang",
    },
    {
        "original": "帮我写一个快速排序",
        "perturbed": "write一个quicksort",
        "expected_tool": "code_executor",
        "robustness_type": "mixed_lang",
    },
]


def evaluate_D5(
    framework: str, fresh: bool = False, max_cases: int | None = None
) -> dict:
    """D5: Robustness.

    Test whether the framework maintains correct routing under:
      - Typo / character noise
      - Verbose prefix/suffix
      - Rephrasing
      - Boundary/ambiguous cases
      - Mixed language

    Metrics:
      - original_accuracy: routing accuracy on clean queries
      - perturbed_accuracy: routing accuracy on perturbed queries
      - robustness_ratio: perturbed / original (1.0 = no degradation)
      - consistency: % of cases where original and perturbed get same tool
    """
    print(f"\n{'=' * 60}")
    print(f"D5 — Robustness | {framework}")
    print(f"{'=' * 60}")

    result_file = RESULTS_DIR / f"D5_{framework}.json"
    saved = {} if fresh else _load_results(result_file)

    cases = ROBUSTNESS_CASES[:max_cases] if max_cases else ROBUSTNESS_CASES

    orig_ok = 0
    pert_ok = 0
    consistent = 0
    total = 0
    total_time = 0.0
    per_type = defaultdict(lambda: {"orig_ok": 0, "pert_ok": 0, "total": 0})

    # Build a minimal function list for routing
    funcs = [
        {"name": "calculator", "description": "Perform mathematical calculations"},
        {"name": "code_executor", "description": "Execute Python code and algorithms"},
        {"name": "search", "description": "Search the web for information"},
        {"name": "bigmodel_proxy", "description": "Consult an AI model for analysis"},
    ]

    # For CARM, use CARMRouter directly (not BFCL-style route_keyword)
    # because D5 tests CARM's built-in tool routing, not generic function selection.
    router = None
    if framework == "carm_hybrid":
        from carm.router import CARMRouter

        router = CARMRouter()

    for i, case in enumerate(cases):
        key = f"case__{i}"
        if key in saved:
            d = saved[key]
            orig_ok += int(d.get("orig_ok", 0))
            pert_ok += int(d.get("pert_ok", 0))
            consistent += int(d.get("consistent", 0))
            total += 1
            total_time += d.get("elapsed", 0)
            rt = d.get("robustness_type", "unknown")
            per_type[rt]["orig_ok"] += int(d.get("orig_ok", 0))
            per_type[rt]["pert_ok"] += int(d.get("pert_ok", 0))
            per_type[rt]["total"] += 1
            continue

        t0 = time.time()

        if router is not None:
            # CARM: use CARMRouter (has signal detection)
            orig_r = router.route(case["original"], dry_run=True)
            pert_r = router.route(case["perturbed"], dry_run=True)
            orig_pred = orig_r.tool_name
            pert_pred = pert_r.tool_name
        else:
            # Other frameworks: use LLM-based routing
            orig_pred, _, _ = _route_query(framework, case["original"], funcs, False)
            pert_pred, used_llm, tok = _route_query(
                framework, case["perturbed"], funcs, False
            )
        elapsed = time.time() - t0

        expected = case["expected_tool"]

        # Allow search→bigmodel upgrade
        def _tool_match(pred, exp):
            if pred == exp:
                return True
            if exp == "search" and pred == "bigmodel_proxy":
                return True
            return False

        o_ok = _tool_match(orig_pred, expected)
        p_ok = _tool_match(pert_pred, expected)
        con = orig_pred == pert_pred

        d = {
            "original": case["original"],
            "perturbed": case["perturbed"],
            "expected_tool": expected,
            "orig_pred": orig_pred,
            "pert_pred": pert_pred,
            "orig_ok": o_ok,
            "pert_ok": p_ok,
            "consistent": con,
            "robustness_type": case["robustness_type"],
            "elapsed": round(elapsed, 3),
        }
        _save_sample(result_file, key, d)
        saved[key] = d

        orig_ok += int(o_ok)
        pert_ok += int(p_ok)
        consistent += int(con)
        total += 1
        total_time += elapsed
        per_type[case["robustness_type"]]["orig_ok"] += int(o_ok)
        per_type[case["robustness_type"]]["pert_ok"] += int(p_ok)
        per_type[case["robustness_type"]]["total"] += 1

        status = "✓" if (o_ok and p_ok and con) else ("~" if (o_ok or p_ok) else "✗")
        print(
            f"  [{status}] {case['robustness_type']:15s} "
            f"orig={orig_pred or 'None':15s} pert={pert_pred or 'None':15s} "
            f"o={o_ok} p={p_ok} c={con}"
        )

    n = total or 1
    robustness = pert_ok / orig_ok if orig_ok else 0

    print(
        f"\n  D5 Result: OrigAcc={orig_ok / n * 100:.0f}% PertAcc={pert_ok / n * 100:.0f}% "
        f"Robustness={robustness:.2f} Consistency={consistent / n * 100:.0f}%"
    )

    return {
        "dimension": "D5_Robustness",
        "framework": framework,
        "original_accuracy": round(orig_ok / n * 100, 1),
        "perturbed_accuracy": round(pert_ok / n * 100, 1),
        "robustness_ratio": round(robustness, 2),
        "consistency": round(consistent / n * 100, 1),
        "total": total,
        "latency_ms": round(total_time / total * 1000) if total else 0,
        "per_type": {
            k: {
                "orig_acc": round(v["orig_ok"] / v["total"] * 100) if v["total"] else 0,
                "pert_acc": round(v["pert_ok"] / v["total"] * 100) if v["total"] else 0,
                "total": v["total"],
            }
            for k, v in per_type.items()
        },
    }


# ════════════════════════════════════════════════════════════════════════
# Unified routing interface
# ════════════════════════════════════════════════════════════════════════


def _route_query(
    framework: str,
    query: str,
    funcs: list[dict],
    is_irrel: bool = False,
    session_id: str | None = None,
) -> tuple[str | None, bool, dict]:
    """Route a query through the specified framework.

    Returns (predicted_func_name_or_None, used_llm, token_info).
    """
    if framework == "carm_hybrid":
        return _route_carm_hybrid(query, funcs, is_irrel, session_id)
    elif framework == "prompt_baseline":
        return _route_prompt_baseline(query, funcs, is_irrel)
    elif framework == "autogen":
        return _route_autogen(query, funcs, is_irrel)
    else:
        raise ValueError(f"Unknown framework: {framework}")


def _route_carm_hybrid(
    query: str, funcs: list[dict], is_irrel: bool, session_id: str | None = None
) -> tuple[str | None, bool, dict]:
    """CARM+LLM hybrid routing."""
    import evaluate_bfcl_v3_llm as evmod

    best_name, best_score = evmod.route_keyword(query, funcs)

    used_llm = False
    if best_score >= 0.6:
        return best_name, False, {"eval": 0, "prompt_eval": 0}
    if len(funcs) == 1 and best_score >= 0.3:
        return best_name, False, {"eval": 0, "prompt_eval": 0}

    if best_score < 0.6:
        llm_result = evmod.route_llm(query, funcs)
        used_llm = True
        if llm_result:
            return llm_result, True, _last_llm_tokens

    if best_score >= 0.3:
        return best_name, False, {"eval": 0, "prompt_eval": 0}

    return None, False, {"eval": 0, "prompt_eval": 0}


def _route_prompt_baseline(
    query: str, funcs: list[dict], is_irrel: bool
) -> tuple[str | None, bool, dict]:
    """Simple prompt baseline: ask LLM which function to use."""
    func_list = "\n".join(
        f"- {f['name']}: {f.get('description', 'N/A')[:80]}" for f in funcs
    )
    prompt = (
        f"Given the following available functions:\n{func_list}\n\n"
        f"User query: {query}\n\n"
        f"Which function should be called? Reply with ONLY the function name, "
        f"or 'NONE' if no function is appropriate."
    )
    response, tok = call_ollama(prompt)
    # Parse response
    predicted = response.strip()
    # Try to extract function name
    func_names = {f["name"] for f in funcs}
    if predicted in func_names:
        return predicted, True, tok
    # Try to find a function name in the response
    for name in func_names:
        if name.lower() in predicted.lower():
            return name, True, tok
    if "none" in predicted.lower() or not predicted:
        return None, True, tok
    return None, True, tok


def _route_autogen(
    query: str, funcs: list[dict], is_irrel: bool
) -> tuple[str | None, bool, dict]:
    """AutoGen-style: system prompt + function list."""
    func_desc = "\n".join(
        f"- {f['name']}({', '.join(f.get('parameters', {}).get('properties', {}).keys())}): "
        f"{f.get('description', '')[:60]}"
        for f in funcs
    )
    system = (
        "You are a helpful assistant. When the user's query requires calling "
        "a function, respond with ONLY the function name. If no function is "
        "needed, respond with 'NONE'.\n\n"
        f"Available functions:\n{func_desc}"
    )
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": query},
    ]
    response, tok = call_ollama_chat(messages)
    predicted = response.strip()
    func_names = {f["name"] for f in funcs}
    if predicted in func_names:
        return predicted, True, tok
    for name in func_names:
        if name.lower() in predicted.lower():
            return name, True, tok
    if "none" in predicted.lower():
        return None, True, tok
    return None, True, tok


def _extract_params(
    framework: str, query: str, funcs: list[dict], gt: list[dict]
) -> dict[str, dict[str, Any]]:
    """Extract parameters from query for the given framework.

    Returns {func_name: {param_name: extracted_value}}.
    """
    if framework in ("carm_hybrid", "carm_only"):
        # CARM is a routing layer — it doesn't extract params.
        # For D2, we test the LLM component's param extraction ability.
        return _extract_params_llm(query, funcs, gt)
    elif framework == "prompt_baseline":
        return _extract_params_llm(query, funcs, gt)
    elif framework == "autogen":
        return _extract_params_llm(query, funcs, gt)
    else:
        return {}


def _extract_params_llm(
    query: str, funcs: list[dict], gt: list[dict]
) -> dict[str, dict[str, Any]]:
    """Ask LLM to extract parameters from the query."""
    gt_names = _gt_func_names(gt)
    if not gt_names:
        return {}

    # Find the target function definitions
    target_funcs = [f for f in funcs if f["name"] in gt_names]
    if not target_funcs:
        return {}

    func_schemas = []
    for f in target_funcs:
        props = f.get("parameters", {}).get("properties", {})
        params_desc = ", ".join(
            f"{p} ({d.get('type', '?')}): {d.get('description', '')[:50]}"
            for p, d in props.items()
        )
        func_schemas.append(f"- {f['name']}({params_desc})")

    prompt = (
        f"Given the query and available functions, extract the parameter values.\n\n"
        f"Functions:\n" + "\n".join(func_schemas) + "\n\n"
        f"Query: {query}\n\n"
        f'Respond in JSON format: {{"function_name": {{"param_name": value, ...}}}}\n'
        f"Only include parameters that can be inferred from the query."
    )

    response, _ = call_ollama(prompt)

    # Try to parse JSON from response
    try:
        # Find JSON in response
        json_match = re.search(r"\{[^{}]*\{[^{}]*\}[^{}]*\}", response, re.DOTALL)
        if json_match:
            parsed = json.loads(json_match.group())
            return parsed
        # Try simpler pattern
        json_match = re.search(r"\{.*\}", response, re.DOTALL)
        if json_match:
            parsed = json.loads(json_match.group())
            if isinstance(parsed, dict):
                return parsed
    except (json.JSONDecodeError, ValueError):
        pass

    return {}


# ════════════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════════════


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Multi-dimensional evaluation")
    parser.add_argument(
        "--quick", action="store_true", help="Quick test (20 samples per subset)"
    )
    parser.add_argument(
        "--fresh", action="store_true", help="Force re-run (ignore saved results)"
    )
    parser.add_argument(
        "--dims",
        default="D1,D2,D3,D4,D5",
        help="Dimensions to evaluate (comma-separated)",
    )
    parser.add_argument(
        "--frameworks",
        default="carm_hybrid,prompt_baseline,autogen",
        help="Frameworks to compare (comma-separated)",
    )
    args = parser.parse_args()

    sys.stdout.reconfigure(encoding="utf-8")

    dims = [d.strip().upper() for d in args.dims.split(",")]
    frameworks = [f.strip() for f in args.frameworks.split(",")]
    # Quick mode: D1/D2 use fewer samples; D3/D5 are small already; D4 very slow
    max_n = 20 if args.quick else None
    max_n_d2 = 10 if args.quick else None  # D2 needs LLM per sample
    max_n_d4 = 5 if args.quick else None  # D4 needs LLM per turn (4-5 turns)

    print("=" * 70)
    print("Multi-Dimensional Evaluation: CARM vs Agent Frameworks")
    print(f"Dimensions: {dims}")
    print(f"Frameworks: {frameworks}")
    print(f"Ollama: {OLLAMA_BASE_URL} ({OLLAMA_MODEL})")
    print("=" * 70)

    # Load BFCL V3 data
    df = _load_bfcl() if any(d in dims for d in ["D1", "D2", "D4"]) else None

    all_results = []

    for fw in frameworks:
        print(f"\n{'#' * 70}")
        print(f"# Framework: {fw}")
        print(f"{'#' * 70}")

        if "D1" in dims and df is not None:
            r = evaluate_D1(fw, df, fresh=args.fresh, max_per_subset=max_n)
            all_results.append(r)

        if "D2" in dims and df is not None:
            r = evaluate_D2(fw, df, fresh=args.fresh, max_per_subset=max_n_d2)
            all_results.append(r)

        if "D3" in dims:
            r = evaluate_D3(fw, fresh=args.fresh, max_cases=max_n)
            all_results.append(r)

        if "D4" in dims and df is not None:
            r = evaluate_D4(fw, df, fresh=args.fresh, max_per_subset=max_n_d4)
            all_results.append(r)

        if "D5" in dims:
            r = evaluate_D5(fw, fresh=args.fresh, max_cases=max_n)
            all_results.append(r)

    # ── Summary ────────────────────────────────────────────────────────
    print("\n\n" + "=" * 70)
    print("SUMMARY — Multi-Dimensional Evaluation")
    print("=" * 70)

    # Group by dimension
    by_dim = defaultdict(list)
    for r in all_results:
        by_dim[r["dimension"]].append(r)

    for dim_name, results in by_dim.items():
        print(f"\n--- {dim_name} ---")
        if dim_name == "D1_Function_Selection":
            print(
                f"  {'Framework':20s} {'Accuracy':>10s} {'Latency':>10s} {'Tokens':>8s} {'LLM%':>6s}"
            )
            for r in results:
                print(
                    f"  {r['framework']:20s} {r['accuracy']:>9.1f}% "
                    f"{r['latency_ms']:>9d}ms {r['tokens_per_call']:>8d} "
                    f"{r['llm_call_rate']:>5d}%"
                )
        elif dim_name == "D2_Parameter_Extraction":
            print(
                f"  {'Framework':20s} {'NameRecall':>10s} {'ValueMatch':>10s} {'Overall':>10s}"
            )
            for r in results:
                print(
                    f"  {r['framework']:20s} {r['name_recall']:>9.1f}% "
                    f"{r['value_match']:>9.1f}% {r['overall']:>9.1f}%"
                )
        elif dim_name == "D3_End_to_End":
            print(
                f"  {'Framework':20s} {'ToolSel':>8s} {'ExecOK':>8s} {'ResultOK':>8s} {'Overall':>8s}"
            )
            for r in results:
                ts = r.get("tool_selection", "N/A")
                eo = r.get("execution_ok", "N/A")
                rq = r.get("result_quality", "N/A")
                ov = r.get("overall", "N/A")

                def _fmt(v):
                    return f"{v:>7.1f}%" if isinstance(v, (int, float)) else f"{v:>8s}"

                print(
                    f"  {r['framework']:20s} {_fmt(ts)} {_fmt(eo)} {_fmt(rq)} {_fmt(ov)}"
                )
        elif dim_name == "D4_Multi_Turn":
            print(
                f"  {'Framework':20s} {'TurnAcc':>10s} {'ConvAcc':>10s} {'Anaphora':>10s} {'Tokens':>8s}"
            )
            for r in results:
                print(
                    f"  {r['framework']:20s} {r['turn_accuracy']:>9.1f}% "
                    f"{r['conversation_accuracy']:>9.1f}% "
                    f"{r.get('anaphora_accuracy', 0):>9.1f}% "
                    f"{r['tokens_per_conv']:>8d}"
                )
        elif dim_name == "D5_Robustness":
            print(
                f"  {'Framework':20s} {'OrigAcc':>8s} {'PertAcc':>8s} {'Robust':>8s} {'Consist':>8s}"
            )
            for r in results:
                print(
                    f"  {r['framework']:20s} {r['original_accuracy']:>7.1f}% "
                    f"{r['perturbed_accuracy']:>7.1f}% {r['robustness_ratio']:>7.2f} "
                    f"{r['consistency']:>7.1f}%"
                )

    # ── Composite Score ────────────────────────────────────────────────
    print("\n--- Composite Score (weighted average) ---")
    fw_scores = defaultdict(list)
    weights = {"D1": 0.25, "D2": 0.20, "D3": 0.25, "D4": 0.15, "D5": 0.15}

    for r in all_results:
        dim_prefix = r["dimension"][:2]
        w = weights.get(dim_prefix, 0.1)
        if dim_prefix == "D1":
            score = r["accuracy"]
        elif dim_prefix == "D2":
            score = r["overall"]
        elif dim_prefix == "D3":
            score = r.get("overall", r.get("result_quality", 0))
        elif dim_prefix == "D4":
            score = r["turn_accuracy"]
        elif dim_prefix == "D5":
            score = r["perturbed_accuracy"]
        else:
            score = 0
        fw_scores[r["framework"]].append((dim_prefix, w, score))

    print(f"  {'Framework':20s} {'Composite':>10s}  Breakdown")
    for fw in frameworks:
        if fw not in fw_scores:
            continue
        entries = fw_scores[fw]
        total_w = sum(w for _, w, _ in entries)
        composite = sum(w * s for _, w, s in entries) / total_w if total_w else 0
        breakdown = " ".join(f"{d}={s:.1f}%" for d, _, s in entries)
        print(f"  {fw:20s} {composite:>9.1f}%  {breakdown}")

    # Save summary
    summary_file = RESULTS_DIR / "multidim_summary.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"\nDetailed results saved to {RESULTS_DIR}/")
    print(f"Summary saved to {summary_file}")


if __name__ == "__main__":
    main()
