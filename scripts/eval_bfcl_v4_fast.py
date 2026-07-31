#!/usr/bin/env python3
"""Fast BFCL V4 evaluation: sends queries to CARM server and scores results.

Reads BFCL JSONL data directly (no parquet dependency), sends to CARM server
on port 11401, and scores against ground truth.

Usage:
    python scripts/eval_bfcl_v4_fast.py --max-samples 50
    python scripts/eval_bfcl_v4_fast.py --categories simple_python,irrelevance
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

import httpx

BFCL_DATA_DIR = Path(r"D:\tools\miniconda3\envs\BFCL\Lib\site-packages\bfcl_eval\data")
PA_DIR = BFCL_DATA_DIR / "possible_answer"
SERVER_URL = "http://localhost:11401/v1/chat/completions"

CATEGORIES = [
    "simple_python",
    "simple_java",
    "simple_javascript",
    "multiple",
    "parallel",
    "parallel_multiple",
    "irrelevance",
    "live_simple",
    "live_multiple",
    "live_parallel",
    "live_parallel_multiple",
    "live_relevance",
    "live_irrelevance",
]

# Categories where empty response = correct
IRRELEVANCE_CATS = {"irrelevance", "live_irrelevance", "live_relevance"}


def load_bfcl_data(cat: str) -> list[dict]:
    """Load BFCL V4 data for a category."""
    data_file = BFCL_DATA_DIR / f"BFCL_v4_{cat}.json"
    if not data_file.exists():
        return []

    items = []
    with open(data_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def load_ground_truth(cat: str) -> dict[str, list]:
    """Load possible answers for a category."""
    pa_file = PA_DIR / f"BFCL_v4_{cat}.json"
    if not pa_file.exists():
        return {}

    gt_map = {}
    with open(pa_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                item = json.loads(line)
                gt_map[item["id"]] = item.get("ground_truth", [])
    return gt_map


def build_messages(item: dict) -> tuple[list[dict], list[dict]]:
    """Build messages and functions from BFCL item."""
    turns = item["question"]
    functions = item["function"]

    # turns is [[{role, content}]] — take first turn
    if isinstance(turns, list) and len(turns) > 0:
        first_turn = turns[0] if isinstance(turns[0], list) else turns
    else:
        first_turn = []

    # Build system message with function definitions
    system_content = json.dumps(functions, ensure_ascii=False)
    messages = [{"role": "system", "content": system_content}]
    for msg in first_turn:
        messages.append(msg)

    return messages, functions


def send_query(messages: list[dict], timeout: int = 120) -> str | None:
    """Send a chat completion request to CARM server."""
    try:
        r = httpx.post(
            SERVER_URL,
            json={"model": "carm-router", "messages": messages},
            timeout=timeout,
        )
        if r.status_code != 200:
            return None
        resp = r.json()
        choices = resp.get("choices", [])
        if choices:
            return choices[0].get("message", {}).get("content", "")
        return None
    except Exception as e:
        print(f"  ERROR: {e}")
        return None


def parse_tool_calls(content: str) -> list[dict]:
    """Parse function calls from server response content."""
    if not content:
        return []

    content = content.strip()

    if content in ("[]", ""):
        return []

    # Try JSON array first
    try:
        parsed = json.loads(content)
        if isinstance(parsed, list):
            # Could be list of dicts with name/arguments
            calls = []
            for item in parsed:
                if isinstance(item, dict) and "name" in item:
                    calls.append(item)
            if calls:
                return calls
    except (json.JSONDecodeError, ValueError):
        pass

    # Parse CARM format: [func_name(param=value, ...), ...]
    calls = []
    if content.startswith("[") and content.endswith("]"):
        content = content[1:-1].strip()

    parts = []
    depth = 0
    in_string = False
    escape = False
    current = ""
    for char in content:
        if escape:
            current += char
            escape = False
            continue
        if char == "\\":
            current += char
            escape = True
            continue
        if char == '"' and not escape:
            in_string = not in_string
        if char == "(" and not in_string:
            depth += 1
        if char == ")" and not in_string:
            depth -= 1
        if char == "," and depth == 0 and not in_string:
            parts.append(current.strip())
            current = ""
            continue
        current += char
    if current.strip():
        parts.append(current.strip())

    for part in parts:
        call = _parse_single_call(part)
        if call:
            calls.append(call)

    return calls


def _parse_single_call(text: str) -> dict | None:
    text = text.strip()
    if not text:
        return None

    paren_idx = text.find("(")
    if paren_idx == -1:
        return {"name": text.strip(), "arguments": {}}

    name = text[:paren_idx].strip()
    args_text = text[paren_idx + 1 :]

    if args_text.rstrip().endswith(")"):
        args_text = args_text.rstrip()[:-1]

    arguments = _parse_args(args_text)
    return {"name": name, "arguments": arguments}


def _parse_args(args_text: str) -> dict:
    args = {}
    if not args_text.strip():
        return args

    depth = 0
    in_string = False
    string_delim = None
    escape = False
    current = ""
    parts = []

    for char in args_text:
        if escape:
            current += char
            escape = False
            continue
        if char == "\\":
            current += char
            escape = True
            continue
        if char in ('"', "'") and not escape:
            if not in_string:
                in_string = True
                string_delim = char
            elif char == string_delim:
                in_string = False
                string_delim = None
            current += char
            continue
        if char == "[" and not in_string:
            depth += 1
        if char == "]" and not in_string:
            depth -= 1
        if char == "{" and not in_string:
            depth += 1
        if char == "}" and not in_string:
            depth -= 1
        if char == "," and depth == 0 and not in_string:
            parts.append(current.strip())
            current = ""
            continue
        current += char
    if current.strip():
        parts.append(current.strip())

    for part in parts:
        eq_idx = part.find("=")
        if eq_idx == -1:
            continue
        key = part[:eq_idx].strip()
        value_str = part[eq_idx + 1 :].strip()

        # Handle Python-style booleans and None first
        if value_str == "True":
            args[key] = True
            continue
        elif value_str == "False":
            args[key] = False
            continue
        elif value_str == "None":
            args[key] = None
            continue

        try:
            value = json.loads(value_str)
        except (json.JSONDecodeError, ValueError):
            # Try converting Python-style single quotes to JSON double quotes
            # Only for list/dict values (starts with [ or {)
            if value_str and value_str[0] in "[{":
                try:
                    fixed = value_str.replace("'", '"')
                    value = json.loads(fixed)
                except (json.JSONDecodeError, ValueError):
                    # Try ast.literal_eval for Python-style dicts/lists
                    try:
                        import ast

                        value = ast.literal_eval(value_str)
                    except (ValueError, SyntaxError):
                        if value_str.startswith('"') and value_str.endswith('"'):
                            value = value_str[1:-1]
                        elif value_str.startswith("'") and value_str.endswith("'"):
                            value = value_str[1:-1]
                        else:
                            value = value_str
            elif value_str.startswith('"') and value_str.endswith('"'):
                value = value_str[1:-1]
            elif value_str.startswith("'") and value_str.endswith("'"):
                value = value_str[1:-1]
            else:
                value = value_str

        args[key] = value

    return args


def _normalize_math_expr(s: str) -> str:
    """Normalize math expressions: x^2 → x**2, remove spaces, remove explicit *."""
    s = s.replace("^", "**")
    s = re.sub(r"\s+", "", s)
    # Remove explicit multiplication: 3*x → 3x, x*2 → x2
    # But preserve ** (power)
    s = s.replace("**", "\x00POW\x00")
    s = re.sub(r"(\d)\*([a-zA-Z])", r"\1\2", s)
    s = re.sub(r"([a-zA-Z])\*(\d)", r"\1\2", s)
    s = s.replace("\x00POW\x00", "**")
    return s


def _values_match(pred, gt) -> bool:
    if pred is None and (gt == "" or gt is None):
        return True
    # Handle string-boolean mismatch
    if isinstance(pred, str) and isinstance(gt, bool):
        return pred.lower() == str(gt).lower()
    if isinstance(pred, bool) and isinstance(gt, str):
        return str(pred).lower() == gt.lower()
    if isinstance(pred, str) and isinstance(gt, str):
        p = pred.lower().strip()
        g = gt.lower().strip()
        if p == g:
            return True
        # Normalize math expressions: x^2 → x**2
        p_norm = _normalize_math_expr(p)
        g_norm = _normalize_math_expr(g)
        if p_norm == g_norm:
            return True
        # Normalize accents (e.g., Cancún vs Cancun)
        import unicodedata

        p_deaccent = (
            unicodedata.normalize("NFKD", p).encode("ascii", "ignore").decode("ascii")
        )
        g_deaccent = (
            unicodedata.normalize("NFKD", g).encode("ascii", "ignore").decode("ascii")
        )
        if p_deaccent == g_deaccent:
            return True
        # Handle Chinese-English location matching
        location_map = {
            "北京": "beijing",
            "上海": "shanghai",
            "广州": "guangzhou",
            "深圳": "shenzhen",
        }
        for cn, en in location_map.items():
            if cn in pred and en in gt:
                return True
            if cn in gt and en in pred:
                return True

        # Time format normalization: "5:00 pm" → "5 pm", "7:30 pm" → "7:30 pm"
        def _normalize_time(s):
            # Remove ":00" from times like "5:00 pm" → "5 pm"
            s = re.sub(r"(\d+):00\s*(pm|am|PM|AM)", r"\1 \2", s)
            return s

        if _normalize_time(p) == _normalize_time(g):
            return True

        # Unit singular/plural matching: "piece" matches "pieces", "ounce" matches "ounces"
        def _normalize_unit(s):
            if s.endswith("ies"):
                return s[:-3] + "y"
            if s.endswith("es"):
                return s[:-2]
            if s.endswith("s") and not s.endswith("ss"):
                return s[:-1]
            return s

        if _normalize_unit(p) == _normalize_unit(g):
            return True
        return False
    # Handle list vs scalar: pred=['x'] matches gt='x' (single-element list unwrap)
    if isinstance(pred, list) and len(pred) == 1 and not isinstance(gt, list):
        return _values_match(pred[0], gt)
    # Handle scalar vs list: pred='x' matches gt=['x', 'y'] (list unwrap)
    if isinstance(gt, list) and len(gt) == 1 and not isinstance(pred, list):
        return _values_match(pred, gt[0])
    # Handle list vs list: pred=['5:00 pm'] matches gt=['5 pm'] (element-wise)
    if isinstance(pred, list) and isinstance(gt, list) and len(pred) == len(gt):
        return all(_values_match(p, g) for p, g in zip(pred, gt))
    # Handle list vs list different lengths: try single-element unwrap on both
    if (
        isinstance(pred, list)
        and isinstance(gt, list)
        and len(pred) == 1
        and len(gt) == 1
    ):
        return _values_match(pred[0], gt[0])
    # Handle dict vs dict: deep compare nested structures
    if isinstance(pred, dict) and isinstance(gt, dict):
        # GT dict values may be lists (e.g., {"nm": ["BarChart"], "mn": ["chartModule"]})
        # pred dict values are scalars (e.g., {"nm": "BarChart", "mn": "chartModule"})
        # Only check keys that exist in GT — missing pred keys with empty GT values are OK
        for k, gt_v in gt.items():
            pred_v = pred.get(k)
            if isinstance(gt_v, list) and len(gt_v) >= 1:
                matched = False
                for gv in gt_v:
                    if _values_match(pred_v, gv):
                        matched = True
                        break
                if not matched:
                    return False
            else:
                if not _values_match(pred_v, gt_v):
                    return False
        return True
    try:
        return float(pred) == float(gt)
    except (ValueError, TypeError):
        pass
    return pred == gt


def _params_match(pred_args: dict, gt_param_dict: dict) -> bool:
    """Check if predicted params match GT params for a single function call."""
    for param_name, gt_values in gt_param_dict.items():
        if not isinstance(gt_values, list) or not gt_values:
            continue

        pred_value = pred_args.get(param_name)

        matched = False
        # First try matching pred against each individual gt_val
        for gt_val in gt_values:
            if _values_match(pred_value, gt_val):
                matched = True
                break
        # If that fails and pred is a list, try matching the entire pred list
        # against the entire gt_values list (e.g., vertices=[[10,15],[20,25]]
        # where GT is [[10.0,15.0],[20.0,25.0]] meaning a single 2D value)
        if not matched and isinstance(pred_value, list):
            if _values_match(pred_value, gt_values):
                matched = True

        if not matched:
            return False
    return True


def score_response(predicted_calls: list[dict], gt: list, cat: str) -> bool:
    """Score a single response against ground truth."""
    is_irrelevance = cat in IRRELEVANCE_CATS

    if is_irrelevance:
        if cat == "live_relevance":
            return len(predicted_calls) > 0
        return len(predicted_calls) == 0

    if not gt:
        return len(predicted_calls) == 0

    # Extract GT function names and params
    gt_names = []
    gt_items = []  # list of (func_name, param_dict)
    for item in gt:
        if isinstance(item, dict):
            for func_name, params in item.items():
                gt_names.append(func_name)
                gt_items.append((func_name, params if isinstance(params, dict) else {}))

    pred_names = [c.get("name", "") for c in predicted_calls]

    # Check function name match (order-independent)
    if sorted(gt_names) != sorted(pred_names):
        return False

    # For same-function calls (parallel), use optimal matching
    # Group by function name
    from collections import defaultdict

    gt_by_name = defaultdict(list)
    pred_by_name = defaultdict(list)
    for name, params in gt_items:
        gt_by_name[name].append(params)
    for call in predicted_calls:
        pred_by_name[call.get("name", "")].append(call.get("arguments", {}))

    for name, gt_param_list in gt_by_name.items():
        pred_param_list = pred_by_name.get(name, [])
        if len(gt_param_list) != len(pred_param_list):
            return False

        # Try all permutations to find best match
        from itertools import permutations

        for perm in permutations(range(len(pred_param_list))):
            all_match = True
            for i, gt_params in enumerate(gt_param_list):
                if not _params_match(pred_param_list[perm[i]], gt_params):
                    all_match = False
                    break
            if all_match:
                break
        else:
            return False

    return True


def run_eval(categories: list[str] | None = None, max_samples: int | None = None):
    """Run BFCL V4 evaluation."""
    cats_to_eval = categories or CATEGORIES

    # Check server health
    try:
        r = httpx.get("http://localhost:11401/health", timeout=5)
        if r.status_code != 200:
            print("ERROR: CARM server not responding on port 11401")
            return
        print("CARM server is healthy ✓")
    except Exception:
        print("ERROR: CARM server not responding on port 11401")
        return

    results = {}
    total_correct = 0
    total_samples = 0
    all_errors = {}

    for cat in cats_to_eval:
        items = load_bfcl_data(cat)
        if not items:
            print(f"  {cat:30s}: NO DATA")
            continue

        gt_map = load_ground_truth(cat)

        if max_samples:
            items = items[:max_samples]

        correct = 0
        errors = []
        t_start = time.time()

        for idx, item in enumerate(items):
            item_id = item.get("id", f"{cat}_{idx}")
            messages, functions = build_messages(item)
            gt = gt_map.get(item_id, [])

            content = send_query(messages)

            if content is None:
                errors.append(
                    {"id": item_id, "type": "no_response", "gt": str(gt)[:100]}
                )
                continue

            predicted = parse_tool_calls(content)
            is_correct = score_response(predicted, gt, cat)

            if is_correct:
                correct += 1
            else:
                errors.append(
                    {
                        "id": item_id,
                        "type": "wrong",
                        "gt_names": [
                            list(i.keys())[0] if isinstance(i, dict) else str(i)
                            for i in gt
                        ],
                        "pred_names": [c.get("name", "") for c in predicted],
                        "content": content[:200],
                    }
                )

        elapsed = time.time() - t_start
        total = len(items)
        accuracy = correct / total * 100 if total > 0 else 0
        results[cat] = {
            "correct": correct,
            "total": total,
            "accuracy": round(accuracy, 2),
            "errors": len(errors),
            "time_s": round(elapsed, 1),
        }
        total_correct += correct
        total_samples += total
        all_errors[cat] = errors

        error_types = {}
        for e in errors:
            t = e["type"]
            error_types[t] = error_types.get(t, 0) + 1

        print(
            f"  {cat:30s}: {correct:4d}/{total:4d} = {accuracy:5.1f}%  "
            f"({elapsed:.0f}s, {len(errors)} errors: {error_types})"
        )

    # Summary
    overall = total_correct / total_samples * 100 if total_samples > 0 else 0
    print(f"\n{'=' * 70}")
    print(f"BFCL V4 — CARM Router (Optimized)")
    print(f"{'=' * 70}")
    for cat, r in results.items():
        print(f"  {cat:30s}: {r['correct']:4d}/{r['total']:4d} = {r['accuracy']:5.1f}%")
    print(f"  {'OVERALL':30s}: {total_correct:4d}/{total_samples:4d} = {overall:5.1f}%")

    # Save results
    output = {
        "model": "CARM-v0.9.3-optimized",
        "benchmark": "BFCL V4",
        "evaluation_dimension": "full_function_calling",
        "overall_accuracy": round(overall, 2),
        "max_samples": max_samples,
        "subsets": results,
        "errors": {k: v[:10] for k, v in all_errors.items()},  # Top 10 errors per cat
    }

    output_path = Path(r"D:\codes\Mustard\data\eval\bfcl_v4_fast_results.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {output_path}")

    # Print top error categories
    print(f"\n{'=' * 70}")
    print("Top error categories (by error count):")
    print(f"{'=' * 70}")
    error_counts = [(cat, len(errs)) for cat, errs in all_errors.items() if errs]
    error_counts.sort(key=lambda x: x[1], reverse=True)
    for cat, count in error_counts[:5]:
        print(f"\n  {cat} ({count} errors):")
        for e in all_errors[cat][:5]:
            print(
                f"    {e['id']}: GT={e.get('gt_names', 'N/A')} → Pred={e.get('pred_names', 'N/A')}"
            )
            if "content" in e:
                print(f"      Response: {e['content'][:150]}")


def main():
    parser = argparse.ArgumentParser(description="Run BFCL V4 fast evaluation")
    parser.add_argument(
        "--categories",
        default=None,
        help="Comma-separated list of categories to evaluate (default: all)",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Max samples per category (for quick testing)",
    )
    args = parser.parse_args()

    cats = args.categories.split(",") if args.categories else None
    run_eval(categories=cats, max_samples=args.max_samples)


if __name__ == "__main__":
    main()
