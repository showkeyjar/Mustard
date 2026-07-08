"""Evaluate CARM on BFCL V3 with LLM-augmented function selection.

When CARM's lightweight keyword matching can't confidently select a function,
we fall back to Ollama for semantic matching. This tests the full CARM+LLM
stack against the BFCL leaderboard.
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("CARM_NO_EMBEDDING", "1")


# ── Helpers ───────────────────────────────────────────────────────────


def extract_user_query(turns) -> str:
    if isinstance(turns, str):
        turns = json.loads(turns)
    if isinstance(turns, list) and len(turns) > 0:
        first_turn = turns[0] if isinstance(turns[0], list) else turns
        for msg in first_turn:
            if isinstance(msg, dict) and msg.get("role") == "user":
                return msg.get("content", "")
    return ""


def extract_gt_functions(gt) -> list[str]:
    if isinstance(gt, str):
        gt = json.loads(gt)
    if isinstance(gt, list):
        return [list(item.keys())[0] for item in gt if isinstance(item, dict)]
    return []


def extract_available_functions(funcs) -> list[dict]:
    if isinstance(funcs, str):
        funcs = json.loads(funcs)
    return funcs if isinstance(funcs, list) else []


def tokenize(text: str) -> set[str]:
    return {w for w in re.split(r"[^a-zA-Z0-9]", text.lower()) if len(w) > 2}


# ── Lightweight keyword matching ─────────────────────────────────────


def _match_score(query: str, func_name: str, desc: str) -> float:
    query_tokens = tokenize(query)
    if not query_tokens:
        return 0.0
    name_tokens = tokenize(func_name.replace("_", " ").replace(".", " "))
    desc_tokens = tokenize(desc)

    name_overlap = (
        len(query_tokens & name_tokens) / max(len(name_tokens), 1) if name_tokens else 0
    )
    desc_overlap = (
        len(query_tokens & desc_tokens) / max(len(desc_tokens), 1) if desc_tokens else 0
    )
    name_parts = [
        p for p in func_name.replace("_", " ").replace(".", " ").split() if len(p) > 2
    ]
    substr_hits = sum(1 for p in name_parts if p.lower() in query.lower())
    substr_score = substr_hits / max(len(name_parts), 1) if name_parts else 0

    score = name_overlap * 2.0 + desc_overlap * 1.0 + substr_score * 1.5
    return min(score / 4.5, 1.0)


def route_keyword(query: str, available_funcs: list[dict]) -> tuple[str | None, float]:
    """Route using keyword matching. Returns (func_name, confidence)."""
    if not available_funcs:
        return None, 0.0
    if len(available_funcs) == 1:
        func = available_funcs[0]
        score = _match_score(query, func.get("name", ""), func.get("description", ""))
        return func.get("name", ""), score

    scored = []
    for func in available_funcs:
        name = func.get("name", "")
        desc = func.get("description", "")
        score = _match_score(query, name, desc)
        scored.append((name, score))

    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[0]


# ── LLM-augmented routing ────────────────────────────────────────────

_ollama_available = None
_last_llm_tokens: dict = {"eval": 0, "prompt_eval": 0}


def check_ollama() -> bool:
    global _ollama_available
    if _ollama_available is not None:
        return _ollama_available
    try:
        from urllib import request

        base_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
        with request.urlopen(f"{base_url}/api/ps", timeout=5) as r:
            _ollama_available = True
    except Exception:
        _ollama_available = False
    return _ollama_available


def route_llm(query: str, available_funcs: list[dict]) -> str | None:
    """Use Ollama to select the best function."""
    try:
        from urllib import request

        base_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
        model = os.environ.get("OLLAMA_MODEL", "qwen3-coder")

        # Build function list
        func_list = []
        for f in available_funcs:
            name = f.get("name", "")
            desc = f.get("description", "")
            func_list.append(f"- {name}: {desc}")

        prompt = (
            f"Select the best function for this query. Reply with ONLY the function name, nothing else.\n\n"
            f"Query: {query}\n\n"
            f"Available functions:\n" + "\n".join(func_list) + "\n\nFunction name:"
        )

        payload = {
            "model": model,
            "prompt": prompt,
            "stream": True,
            "options": {"temperature": 0.0, "num_predict": 32, "num_ctx": 1024},
            "think": False,
        }

        req = request.Request(
            f"{base_url}/api/generate",
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        full_text = ""
        eval_count = 0
        prompt_eval_count = 0
        with request.urlopen(req, timeout=30) as resp:
            for raw_line in resp:
                if not raw_line:
                    continue
                try:
                    chunk = json.loads(raw_line.decode("utf-8"))
                except (json.JSONDecodeError, UnicodeDecodeError):
                    continue
                if chunk.get("done"):
                    eval_count = chunk.get("eval_count", 0)
                    prompt_eval_count = chunk.get("prompt_eval_count", 0)
                    break
                full_text += chunk.get("response", "")

        # Extract function name from response
        result = full_text.strip()
        # Clean up: remove quotes, dots at end
        result = result.strip("\"'`.").strip()

        # Store token stats in module-level global for callers to read
        global _last_llm_tokens
        _last_llm_tokens = {"eval": eval_count, "prompt_eval": prompt_eval_count}

        # Verify the result matches one of the available functions
        available_names = {f.get("name", "") for f in available_funcs}
        if result in available_names:
            return result

        # Try partial match
        for name in available_names:
            if name.lower() in result.lower() or result.lower() in name.lower():
                return name

        return None

    except Exception:
        return None


# ── Hybrid router ────────────────────────────────────────────────────


def route_hybrid(
    query: str, available_funcs: list[dict], use_llm: bool = True
) -> str | None:
    """Route using keyword matching first, fall back to LLM if confidence is low."""
    best_name, best_score = route_keyword(query, available_funcs)

    # High confidence: return immediately (CARM's speed advantage)
    if best_score >= 0.6:
        return best_name

    # Medium confidence + single function: trust it
    if len(available_funcs) == 1 and best_score >= 0.3:
        return best_name

    # Low confidence: try LLM
    if use_llm and check_ollama() and best_score < 0.6:
        llm_result = route_llm(query, available_funcs)
        if llm_result:
            return llm_result

    # No LLM available or LLM failed: return best keyword match if above threshold
    if best_score >= 0.3:
        return best_name

    # Very low confidence: treat as irrelevant
    return None


# ── Main evaluation ──────────────────────────────────────────────────


def evaluate_bfcl(use_llm: bool = True):
    data_path = (
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
    if not data_path.exists():
        print(f"Data not found at {data_path}")
        return

    df = pd.read_parquet(str(data_path))

    eval_subsets = [
        ("simple", False),
        ("multiple", False),
        ("parallel", False),
        ("parallel_multiple", False),
        ("irrelevance", True),
        ("live_simple", False),
        ("live_multiple", False),
    ]

    results = {}
    total_correct = 0
    total_samples = 0
    llm_calls = 0
    total_time = 0

    mode_label = "CARM+LLM" if use_llm else "CARM-only"

    for subset, is_irr in eval_subsets:
        subset_df = df[df["subset"] == subset]
        if len(subset_df) == 0:
            continue

        correct = 0
        total = 0
        local_llm_calls = 0

        for idx, row in subset_df.iterrows():
            query = extract_user_query(row["turns"])
            if not query:
                continue

            gt_funcs = extract_gt_functions(row["ground_truth"])
            available_funcs = extract_available_functions(row["functions"])

            total += 1
            t0 = time.time()

            predicted = route_hybrid(query, available_funcs, use_llm=use_llm)

            elapsed = time.time() - t0
            total_time += elapsed

            # Track LLM usage
            best_name, best_score = route_keyword(query, available_funcs)
            if best_score < 0.6 and use_llm and check_ollama():
                local_llm_calls += 1

            # Score
            if is_irr:
                if predicted is None:
                    correct += 1
            else:
                gt_func_set = set(gt_funcs)
                if predicted and predicted in gt_func_set:
                    correct += 1
                elif not gt_funcs and predicted is None:
                    correct += 1

        accuracy = correct / total * 100 if total > 0 else 0
        results[subset] = {
            "correct": correct,
            "total": total,
            "accuracy": round(accuracy, 2),
            "llm_calls": local_llm_calls,
        }
        total_correct += correct
        total_samples += total
        llm_calls += local_llm_calls

        print(
            f"  {subset:25s}: {correct:4d}/{total:4d} = {accuracy:5.1f}% (LLM: {local_llm_calls})"
        )

    overall = total_correct / total_samples * 100 if total_samples > 0 else 0
    avg_time = total_time / total_samples if total_samples > 0 else 0

    print(f"\n{'=' * 60}")
    print(f"BFCL V3 - {mode_label} Function Selection")
    print(f"{'=' * 60}")
    for subset, r in results.items():
        print(
            f"  {subset:25s}: {r['correct']:4d}/{r['total']:4d} = {r['accuracy']:5.1f}%"
        )
    print(f"  {'OVERALL':25s}: {total_correct:4d}/{total_samples:4d} = {overall:5.1f}%")
    print(f"  {'Avg latency':25s}: {avg_time * 1000:.0f}ms")
    print(
        f"  {'LLM calls':25s}: {llm_calls}/{total_samples} ({llm_calls / total_samples * 100:.0f}%)"
    )

    # ── Leaderboard comparison ────────────────────────────────────────
    # BFCL V3 Overall Accuracy (from leaderboard, full FC = name+params)
    leaderboard_fc = {
        "GPT-4o (FC)": 89.52,
        "Claude-3.5-Sonnet (FC)": 88.85,
        "Gemini-1.5-Pro (FC)": 86.42,
        "GPT-4-Turbo (FC)": 85.03,
        "Qwen2.5-72B (FC)": 81.33,
        "Llama-3.1-70B (FC)": 78.67,
        "GPT-3.5-Turbo (FC)": 63.44,
        "Llama-3-8B (FC)": 52.89,
    }
    leaderboard_prompt = {
        "GPT-4o (Prompt)": 80.54,
        "Claude-3.5-Sonnet (Prompt)": 72.51,
        "Llama-3.1-70B (Prompt)": 61.27,
    }

    print(f"\n{'=' * 60}")
    print(f"BFCL V3 Leaderboard Comparison")
    print(f"{'=' * 60}")
    print(f"NOTE: CARM measures function SELECTION only.")
    print(f"BFCL leaderboard measures full function CALLING (name + params).")
    print(f"Selection accuracy is inherently higher than full FC accuracy.\n")

    carm_score = round(overall, 2)

    all_models = [(k, v, "FC") for k, v in leaderboard_fc.items()]
    all_models += [(k, v, "Prompt") for k, v in leaderboard_prompt.items()]
    all_models.sort(key=lambda x: x[1], reverse=True)

    carm_rank = 1
    for model, score, mode in all_models:
        if carm_score < score:
            carm_rank += 1

    # Print with CARM inserted
    printed = 0
    for model, score, mode in all_models:
        printed += 1
        if printed == carm_rank:
            print(f"  >>> CARM-v0.9.2 (Selection)      : {carm_score:.1f}% <<<")
        print(f"  #{printed:2d} {model:35s} ({mode:6s}): {score:.2f}%")

    if carm_rank > printed:
        print(f"  >>> CARM-v0.9.2 (Selection)      : {carm_score:.1f}% <<<")

    print(f"\n  CARM rank among FC models: #{carm_rank}/{len(all_models) + 1}")

    # Save
    output = {
        "model": f"CARM-v0.9.2-{mode_label}",
        "benchmark": "BFCL-V3",
        "evaluation_dimension": "function_selection_only",
        "overall_accuracy": round(overall, 2),
        "avg_latency_ms": round(avg_time * 1000, 1),
        "llm_call_rate": round(llm_calls / total_samples * 100, 1)
        if total_samples
        else 0,
        "subsets": results,
        "leaderboard_fc": leaderboard_fc,
        "leaderboard_prompt": leaderboard_prompt,
        "carm_rank_among_fc": carm_rank,
    }
    out_path = PROJECT_ROOT / "data" / "eval" / "bfcl_v3_carm_llm_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\nResults saved to {out_path}")

    return overall


if __name__ == "__main__":
    use_llm = "--no-llm" not in sys.argv
    evaluate_bfcl(use_llm=use_llm)
