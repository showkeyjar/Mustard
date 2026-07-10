"""Compute confidence intervals for evaluation results.

Uses Wilson score interval for proportions (more accurate than normal
approximation for small samples or extreme proportions).

Usage:
    python scripts/compute_ci.py --input data/eval/multidim_results/D5_carm_hybrid.json
    python scripts/compute_ci.py --summary data/eval/multidim_results/multidim_summary.json
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def wilson_ci(
    successes: int, total: int, z: float = 1.96
) -> tuple[float, float, float]:
    """Wilson score interval for a proportion.

    Args:
        successes: Number of successes.
        total: Total trials.
        z: Z-score for confidence level (1.96 = 95%).

    Returns:
        (point_estimate, lower, upper) as fractions.
    """
    if total == 0:
        return 0.0, 0.0, 0.0
    p = successes / total
    n = total
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    spread = z * math.sqrt((p * (1 - p) + z * z / (4 * n)) / n) / denom
    return p, max(0.0, center - spread), min(1.0, center + spread)


def format_pct(fraction: float) -> str:
    return f"{fraction * 100:.1f}%"


def format_ci(point: float, lower: float, upper: float) -> str:
    return f"{format_pct(point)} [{format_pct(lower)}, {format_pct(upper)}]"


def compute_ci_for_d1(filepath: str) -> dict:
    """Compute CI for D1 function selection results."""
    with open(filepath, encoding="utf-8") as f:
        data = json.load(f)

    # D1 data is stored as per-sample entries
    if isinstance(data, dict):
        # Check if it's the multidim format (case__0, case__1, ...)
        if any(k.startswith("case__") for k in data.keys()):
            total = len(data)
            correct = sum(
                1
                for k, v in data.items()
                if k.startswith("case__") and v.get("func_ok", v.get("orig_ok", False))
            )
        # Check if it has accuracy/total fields
        elif "accuracy" in data and "total" in data:
            correct = int(data["accuracy"] * data["total"] / 100)
            total = data["total"]
        else:
            return {"error": "unknown D1 format"}
    else:
        return {"error": "unknown D1 format"}

    p, lo, hi = wilson_ci(correct, total)
    return {
        "dimension": "D1 Function Selection",
        "correct": correct,
        "total": total,
        "point_estimate": f"{p * 100:.1f}%",
        "ci_95": f"[{lo * 100:.1f}%, {hi * 100:.1f}%]",
        "formatted": format_ci(p, lo, hi),
    }


def compute_ci_for_d5(filepath: str) -> dict:
    """Compute CI for D5 robustness results."""
    with open(filepath, encoding="utf-8") as f:
        data = json.load(f)

    total = 0
    correct = 0
    for k, v in data.items():
        if k.startswith("case__"):
            total += 1
            if v.get("pert_ok", False):
                correct += 1

    p, lo, hi = wilson_ci(correct, total)
    return {
        "dimension": "D5 Robustness",
        "correct": correct,
        "total": total,
        "point_estimate": f"{p * 100:.1f}%",
        "ci_95": f"[{lo * 100:.1f}%, {hi * 100:.1f}%]",
        "formatted": format_ci(p, lo, hi),
    }


def compute_ci_for_bfcl(category: str, accuracy: float, total: int) -> dict:
    """Compute CI for BFCL official results."""
    correct = int(accuracy / 100 * total)
    p, lo, hi = wilson_ci(correct, total)
    return {
        "dimension": f"BFCL {category}",
        "correct": correct,
        "total": total,
        "point_estimate": f"{accuracy:.2f}%",
        "ci_95": f"[{lo * 100:.1f}%, {hi * 100:.1f}%]",
        "formatted": format_ci(p, lo, hi),
    }


# BFCL V4 official results (from official scorer)
BFCL_RESULTS = {
    "simple_python": {"accuracy": 96.50, "total": 400},
    "multiple": {"accuracy": 95.00, "total": 200},
    "parallel": {"accuracy": 13.50, "total": 200},
    "parallel_multiple": {"accuracy": 3.50, "total": 200},
    "irrelevance": {"accuracy": 66.25, "total": 240},
    "live_simple": {"accuracy": 86.05, "total": 258},
    "live_multiple": {"accuracy": 77.97, "total": 1053},
    "live_parallel": {"accuracy": 0.00, "total": 16},
    "live_parallel_multiple": {"accuracy": 4.17, "total": 24},
    "live_irrelevance": {"accuracy": 38.24, "total": 882},
    "live_relevance": {"accuracy": 93.75, "total": 18},
    "multi_turn_base": {"accuracy": 0.00, "total": 200},
}


def main():
    parser = argparse.ArgumentParser(description="Compute confidence intervals")
    parser.add_argument("--input", help="Input JSON file for D1/D5")
    parser.add_argument(
        "--bfcl", action="store_true", help="Compute CI for BFCL results"
    )
    parser.add_argument("--output", help="Output JSON file")
    args = parser.parse_args()

    results = {}

    if args.bfcl:
        print("=== BFCL V4 Official Results with 95% CI ===\n")
        for cat, info in BFCL_RESULTS.items():
            ci = compute_ci_for_bfcl(cat, info["accuracy"], info["total"])
            results[cat] = ci
            print(f"  {cat:30s}  {ci['formatted']}  (n={ci['total']})")

    if args.input:
        p = Path(args.input)
        if "D1" in p.name:
            ci = compute_ci_for_d1(args.input)
            results["d1"] = ci
            print(f"\n=== {ci['dimension']} ===")
            print(f"  {ci['formatted']}  (n={ci['total']})")
        elif "D5" in p.name:
            ci = compute_ci_for_d5(args.input)
            results["d5"] = ci
            print(f"\n=== {ci['dimension']} ===")
            print(f"  {ci['formatted']}  (n={ci['total']})")

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
