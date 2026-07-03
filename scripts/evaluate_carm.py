"""CARM comprehensive evaluation framework.

Runs structured test queries through the full agent pipeline and scores
results across five dimensions:
  1. Routing accuracy   — did the agent pick the correct tool?
  2. Answer correctness — is the numerical/factual answer right?
  3. Output quality     — does the answer contain expected keywords?
  4. Robustness         — does the agent survive edge-case inputs?
  5. End-to-end success — route correct + answer non-empty + no repetition

Usage:
    python -m scripts.evaluate_carm
    python -m scripts.evaluate_carm --output data/eval/carm_eval_report.json
    python -m scripts.evaluate_carm --repeat 3   # run 3 times for stability
"""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

# ---------------------------------------------------------------------------
# Test case definitions
# ---------------------------------------------------------------------------

ToolName = Literal["calculator", "code_executor", "search", "bigmodel_proxy"]
AnswerType = Literal["numeric", "sorted_list", "keyword", "free_text", "none"]


@dataclass
class TestCase:
    """A single evaluation test case."""

    query: str
    expected_tool: ToolName
    category: str  # routing / quality / robustness
    difficulty: str = "normal"  # easy / normal / hard
    # --- Output validation ---
    expected_keywords: list[str] = field(default_factory=list)
    expected_answer: float | None = None  # for numeric answers
    answer_tolerance: float = 0.01  # relative tolerance for numeric comparison
    answer_type: AnswerType = "keyword"  # how to validate the answer
    # --- Routing flags ---
    allow_llm_escalation: bool = False
    skip_route_check: bool = False


# ---------------------------------------------------------------------------
# Test suite — 45 queries covering all 4 tools + edge cases + hard cases
# ---------------------------------------------------------------------------

TEST_SUITE: list[TestCase] = [
    # ══ Calculator — Easy ══════════════════════════════════════════════
    TestCase(
        query="2的10次方",
        expected_tool="calculator",
        category="routing",
        difficulty="easy",
        expected_answer=1024,
        answer_type="numeric",
    ),
    TestCase(
        query="100的平方根",
        expected_tool="calculator",
        category="routing",
        difficulty="easy",
        expected_answer=10,
        answer_type="numeric",
    ),
    TestCase(
        query="3.14乘以25",
        expected_tool="calculator",
        category="routing",
        difficulty="easy",
        expected_answer=78.5,
        answer_type="numeric",
    ),
    TestCase(
        query="0的平方",
        expected_tool="calculator",
        category="robustness",
        difficulty="easy",
        expected_answer=0,
        answer_type="numeric",
    ),
    # ══ Calculator — Normal ═════════════════════════════════════════════
    TestCase(
        query="100个席位每席位129元按年预算多少",
        expected_tool="calculator",
        category="routing",
        difficulty="normal",
        expected_answer=154800,
        answer_type="numeric",
    ),
    TestCase(
        query="计算 999 * 999",
        expected_tool="calculator",
        category="quality",
        difficulty="normal",
        expected_answer=998001,
        answer_type="numeric",
    ),
    # ══ Calculator — Hard ═══════════════════════════════════════════════
    TestCase(
        query="5加3乘2",
        expected_tool="calculator",
        category="routing",
        difficulty="hard",
        # 5+3*2 = 11 (standard precedence), also accept 16 ((5+3)*2)
        expected_keywords=["11", "16"],
        answer_type="keyword",
    ),
    TestCase(
        query="1万亿除以14亿等于多少",
        expected_tool="calculator",
        category="routing",
        difficulty="hard",
        expected_answer=714.2857,
        answer_tolerance=0.01,
        answer_type="numeric",
    ),
    TestCase(
        query="负3加5",
        expected_tool="calculator",
        category="quality",
        difficulty="hard",
        expected_answer=2,
        answer_type="numeric",
    ),
    TestCase(
        query="2的32次方减1",
        expected_tool="calculator",
        category="quality",
        difficulty="hard",
        expected_answer=4294967295,
        answer_type="numeric",
    ),
    TestCase(
        query="3亿加2亿",
        expected_tool="calculator",
        category="routing",
        difficulty="normal",
        expected_answer=500000000,
        answer_type="numeric",
    ),
    # ══ Code Executor — Normal ══════════════════════════════════════════
    TestCase(
        query="用快速排序排 5,3,8,1,9,2",
        expected_tool="code_executor",
        category="routing",
        difficulty="normal",
        expected_keywords=["1", "2", "3", "5", "8", "9"],
        answer_type="sorted_list",
    ),
    TestCase(
        query="帮我用冒泡排序排 9,5,7,1,3",
        expected_tool="code_executor",
        category="routing",
        difficulty="normal",
        expected_keywords=["1", "3", "5", "7", "9"],
        answer_type="sorted_list",
    ),
    TestCase(
        query="写一个斐波那契数列",
        expected_tool="code_executor",
        category="routing",
        difficulty="normal",
        expected_keywords=["0", "1"],
        answer_type="keyword",
    ),
    TestCase(
        query="用归并排序排 7,2,9,4",
        expected_tool="code_executor",
        category="routing",
        difficulty="normal",
        expected_keywords=["2", "4", "7", "9"],
        answer_type="sorted_list",
    ),
    TestCase(
        query="运行二分查找 7 在 1,3,5,7,9 中",
        expected_tool="code_executor",
        category="routing",
        difficulty="normal",
        expected_keywords=["3"],
        answer_type="keyword",
    ),
    TestCase(
        query="python代码 print('hello')",
        expected_tool="code_executor",
        category="quality",
        difficulty="easy",
        expected_keywords=["hello"],
        answer_type="keyword",
    ),
    TestCase(
        query="帮我写个冒泡排序",
        expected_tool="code_executor",
        category="quality",
        difficulty="normal",
        expected_keywords=["排序"],
        answer_type="keyword",
    ),
    # ══ Code Executor — Hard ═══════════════════════════════════════════
    TestCase(
        query="帮我写一个求阶乘的函数",
        expected_tool="code_executor",
        category="routing",
        difficulty="normal",
        expected_keywords=["120"],
        answer_type="keyword",
    ),
    TestCase(
        query="用插入排序排 4,1,3,2",
        expected_tool="code_executor",
        category="routing",
        difficulty="hard",
        expected_keywords=["1", "2", "3", "4"],
        answer_type="sorted_list",
    ),
    TestCase(
        query="计算10的阶乘",
        expected_tool="code_executor",
        category="routing",
        difficulty="hard",
        expected_keywords=["3628800"],
        answer_type="keyword",
    ),
    # ══ Search / Knowledge — Easy ══════════════════════════════════════
    TestCase(
        query="什么是机器学习",
        expected_tool="search",
        category="routing",
        difficulty="easy",
        expected_keywords=["学习"],
        answer_type="keyword",
        allow_llm_escalation=True,
    ),
    TestCase(
        query="Python是什么语言",
        expected_tool="search",
        category="routing",
        difficulty="easy",
        expected_keywords=["Python"],
        answer_type="keyword",
        allow_llm_escalation=True,
    ),
    TestCase(
        query="什么是深度学习",
        expected_tool="search",
        category="routing",
        difficulty="easy",
        expected_keywords=["学习"],
        answer_type="keyword",
        allow_llm_escalation=True,
    ),
    # ══ Search / Knowledge — Normal ════════════════════════════════════
    TestCase(
        query="解释一下什么是递归",
        expected_tool="search",
        category="routing",
        difficulty="normal",
        expected_keywords=["递归"],
        answer_type="keyword",
        allow_llm_escalation=True,
    ),
    TestCase(
        query="对比Python和Go的性能",
        expected_tool="search",
        category="routing",
        difficulty="normal",
        expected_keywords=["Python"],
        answer_type="keyword",
        allow_llm_escalation=True,
    ),
    TestCase(
        query="比较 PostgreSQL 和 MySQL 在中小团队里的适用性",
        expected_tool="search",
        category="routing",
        difficulty="normal",
        expected_keywords=["PostgreSQL", "MySQL"],
        answer_type="keyword",
        allow_llm_escalation=True,
    ),
    TestCase(
        query="量子计算的原理是什么",
        expected_tool="search",
        category="quality",
        difficulty="normal",
        expected_keywords=["量子"],
        answer_type="keyword",
        allow_llm_escalation=True,
    ),
    TestCase(
        query="REST和GraphQL的区别",
        expected_tool="search",
        category="routing",
        difficulty="normal",
        expected_keywords=["REST", "GraphQL"],
        answer_type="keyword",
        allow_llm_escalation=True,
    ),
    # ══ BigModel Proxy ═════════════════════════════════════════════════
    TestCase(
        query="给管理层写一份关于AI趋势的简要总结",
        expected_tool="bigmodel_proxy",
        category="routing",
        difficulty="hard",
        answer_type="free_text",
        allow_llm_escalation=True,
    ),
    TestCase(
        query="正式报告：总结本季度技术选型的优缺点",
        expected_tool="bigmodel_proxy",
        category="routing",
        difficulty="hard",
        answer_type="free_text",
        allow_llm_escalation=True,
    ),
    # ══ Confusion / Ambiguity cases ════════════════════════════════════
    TestCase(
        query="帮我算一下排序的时间复杂度",
        expected_tool="code_executor",
        category="routing",
        difficulty="hard",
        # "算" triggers calc, but "排序" is an algorithm → code_executor
        answer_type="free_text",
    ),
    TestCase(
        query="解释冒泡排序的原理",
        expected_tool="search",
        category="routing",
        difficulty="hard",
        expected_keywords=["排序"],
        answer_type="keyword",
        allow_llm_escalation=True,
    ),
    TestCase(
        query="比较冒泡排序和快速排序的效率",
        expected_tool="search",
        category="routing",
        difficulty="hard",
        expected_keywords=["排序"],
        answer_type="keyword",
        allow_llm_escalation=True,
    ),
    # ══ Robustness ═════════════════════════════════════════════════════
    TestCase(
        query="",
        expected_tool="search",
        category="robustness",
        difficulty="hard",
        answer_type="none",
        skip_route_check=True,
    ),
    TestCase(
        query="你好",
        expected_tool="search",
        category="robustness",
        difficulty="normal",
        answer_type="none",
        skip_route_check=True,
    ),
    TestCase(
        query="1234567890" * 20,
        expected_tool="search",
        category="robustness",
        difficulty="hard",
        answer_type="none",
        skip_route_check=True,
    ),
    TestCase(
        query="!@#$%^&*()",
        expected_tool="search",
        category="robustness",
        difficulty="hard",
        answer_type="none",
        skip_route_check=True,
    ),
    # ══ Multi-step / Long-tail ═════════════════════════════════════════
    TestCase(
        query="帮我计算圆的面积，半径是5",
        expected_tool="calculator",
        category="quality",
        difficulty="normal",
        expected_answer=78.54,
        answer_tolerance=0.01,
        answer_type="numeric",
    ),
    TestCase(
        query="一个长5宽3的矩形面积是多少",
        expected_tool="calculator",
        category="quality",
        difficulty="normal",
        expected_answer=15,
        answer_type="numeric",
    ),
    TestCase(
        query="50的阶乘是多少",
        expected_tool="code_executor",
        category="routing",
        difficulty="hard",
        # Very large number, just check routing and no crash
        answer_type="free_text",
    ),
    # ══ Advanced / Stress test ══════════════════════════════════════════
    # Cases designed to expose weaknesses in routing or computation
    TestCase(
        query="帮我用二分查找在 2,4,6,8,10 中找6",
        expected_tool="code_executor",
        category="quality",
        difficulty="normal",
        expected_keywords=["2"],
        answer_type="keyword",
    ),
    TestCase(
        query="运行插入排序 3,1,4,1,5",
        expected_tool="code_executor",
        category="quality",
        difficulty="normal",
        expected_keywords=["1", "3", "4", "5"],
        answer_type="sorted_list",
    ),
    TestCase(
        query="2的16次方",
        expected_tool="calculator",
        category="quality",
        difficulty="easy",
        expected_answer=65536,
        answer_type="numeric",
    ),
    TestCase(
        query="比较Java和Python的适用场景",
        expected_tool="search",
        category="routing",
        difficulty="normal",
        expected_keywords=["Python"],
        answer_type="keyword",
        allow_llm_escalation=True,
    ),
    TestCase(
        query="什么是微服务架构",
        expected_tool="search",
        category="routing",
        difficulty="normal",
        expected_keywords=["微服务"],
        answer_type="keyword",
        allow_llm_escalation=True,
    ),
    TestCase(
        query="负5乘以负3",
        expected_tool="calculator",
        category="quality",
        difficulty="hard",
        expected_answer=15,
        answer_type="numeric",
    ),
    TestCase(
        query="1加2加3加4加5",
        expected_tool="calculator",
        category="quality",
        difficulty="normal",
        expected_answer=15,
        answer_type="numeric",
    ),
    TestCase(
        query="帮我归纳一下K8s和Docker的区别",
        expected_tool="search",
        category="routing",
        difficulty="hard",
        expected_keywords=["Docker"],
        answer_type="keyword",
        allow_llm_escalation=True,
    ),
]


# ---------------------------------------------------------------------------
# Scoring logic
# ---------------------------------------------------------------------------


@dataclass
class EvalResult:
    """Result for a single test case."""

    query: str
    expected_tool: str
    actual_tool: str | None = None
    route_correct: bool = False
    output: str = ""
    keyword_hit: bool = False
    answer_correct: bool = False
    no_crash: bool = True
    no_dup: bool = True
    answer_nonempty: bool = False
    e2e_success: bool = False
    latency_ms: float = 0.0
    actions: list[str] = field(default_factory=list)
    error: str = ""


def _detect_tool_from_actions(actions: list[str], output: str) -> str | None:
    """Infer which tool was actually called from the action trace."""
    if "CALL_TOOL" not in actions:
        return None
    if any(kw in output for kw in ["计算结果", "calculator"]):
        return "calculator"
    if any(kw in output for kw in ["执行结果", "代码执行", "code_executor"]):
        return "code_executor"
    if any(
        kw in output for kw in ["基于大模型分析", "llm_escalation", "bigmodel_proxy"]
    ):
        return "bigmodel_proxy"
    if any(kw in output for kw in ["检索结果", "检索到", "fallback", "百科"]):
        return "search"
    return "search"


def _check_duplicates(output: str) -> bool:
    """Return True if there are substantial duplicate lines."""
    lines = [l.strip() for l in output.split("\n") if len(l.strip()) > 40]
    return len(lines) != len(set(lines))


def _check_numeric_answer(output: str, expected: float, tolerance: float) -> bool:
    """Check if the output contains a number close to the expected value."""
    # Extract all numbers from the output
    numbers = re.findall(r"[-+]?\d+\.?\d*", output)
    for num_str in numbers:
        try:
            num = float(num_str)
            if abs(num) > 0 and expected != 0:
                if abs(num - expected) / max(abs(expected), 1e-10) <= tolerance:
                    return True
            elif abs(num - expected) <= tolerance:
                return True
        except ValueError:
            continue
    return False


def evaluate_single(runner, test: TestCase) -> EvalResult:
    """Run a single test case and score it."""
    result = EvalResult(
        query=test.query,
        expected_tool=test.expected_tool,
    )

    try:
        t0 = time.perf_counter()
        answer, trace = runner.run(test.query)
        result.latency_ms = (time.perf_counter() - t0) * 1000
        result.output = answer
        result.actions = trace.actions
        result.no_crash = True
        result.answer_nonempty = len(answer.strip()) > 10

        # Detect which tool was called — prefer the step-level selected_tool
        # field (authoritative) over output-text heuristics.
        selected_tools = [
            step.selected_tool for step in trace.steps if step.selected_tool
        ]
        if selected_tools:
            result.actual_tool = selected_tools[-1]
        else:
            result.actual_tool = _detect_tool_from_actions(trace.actions, answer)

        # Route correctness
        if test.skip_route_check:
            result.route_correct = result.no_crash
        elif result.actual_tool == test.expected_tool:
            result.route_correct = True
        elif test.allow_llm_escalation and result.actual_tool == "bigmodel_proxy":
            result.route_correct = True

        # Keyword check
        if not test.expected_keywords:
            result.keyword_hit = True
        else:
            result.keyword_hit = any(kw in answer for kw in test.expected_keywords)

        # Answer correctness
        if test.answer_type == "numeric" and test.expected_answer is not None:
            result.answer_correct = _check_numeric_answer(
                answer, test.expected_answer, test.answer_tolerance
            )
        elif test.answer_type == "keyword":
            result.answer_correct = result.keyword_hit
        elif test.answer_type in ("sorted_list", "free_text", "none"):
            result.answer_correct = result.keyword_hit

        # Duplicate check
        result.no_dup = not _check_duplicates(answer)

        # E2E success
        if test.skip_route_check:
            result.e2e_success = result.no_crash
        else:
            result.e2e_success = (
                result.route_correct
                and result.answer_nonempty
                and result.no_dup
                and result.answer_correct
            )

    except Exception as e:
        result.no_crash = False
        result.error = f"{type(e).__name__}: {e}"
        result.e2e_success = False

    return result


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------


def generate_report(results: list[EvalResult], run_count: int = 1) -> dict:
    """Generate a structured evaluation report."""
    total = len(results)
    if total == 0:
        return {"error": "no results"}

    # Overall scores
    routing_correct = sum(1 for r in results if r.route_correct)
    answer_correct_count = sum(1 for r in results if r.answer_correct)
    keyword_hits = sum(1 for r in results if r.keyword_hit)
    no_crash_count = sum(1 for r in results if r.no_crash)
    no_dup_count = sum(1 for r in results if r.no_dup)
    e2e_success_count = sum(1 for r in results if r.e2e_success)
    answer_nonempty_count = sum(1 for r in results if r.answer_nonempty)

    # Per-dimension scores (0-100)
    routing_score = round(routing_correct / total * 100, 1)
    correctness_score = round(answer_correct_count / total * 100, 1)
    quality_score = round(keyword_hits / total * 100, 1)
    robustness_score = round(no_crash_count / total * 100, 1)
    e2e_score = round(e2e_success_count / total * 100, 1)

    # Composite score (weighted average)
    composite = round(
        routing_score * 0.25
        + correctness_score * 0.25
        + quality_score * 0.15
        + robustness_score * 0.10
        + e2e_score * 0.25,
        1,
    )

    # Latency statistics
    latencies = sorted([r.latency_ms for r in results if r.latency_ms > 0])
    avg_latency = round(statistics.mean(latencies), 0) if latencies else 0
    p50_latency = round(latencies[len(latencies) // 2], 0) if latencies else 0
    p95_latency = (
        round(latencies[int(len(latencies) * 0.95)], 0) if len(latencies) > 1 else 0
    )

    # Per-tool breakdown
    tool_stats: dict[str, dict] = {}
    for r in results:
        exp = r.expected_tool
        if exp not in tool_stats:
            tool_stats[exp] = {
                "total": 0,
                "route_correct": 0,
                "answer_correct": 0,
                "keyword_hit": 0,
                "e2e": 0,
            }
        tool_stats[exp]["total"] += 1
        if r.route_correct:
            tool_stats[exp]["route_correct"] += 1
        if r.answer_correct:
            tool_stats[exp]["answer_correct"] += 1
        if r.keyword_hit:
            tool_stats[exp]["keyword_hit"] += 1
        if r.e2e_success:
            tool_stats[exp]["e2e"] += 1

    # Per-difficulty breakdown
    difficulty_stats: dict[str, dict] = {}
    for r in results:
        cat = "unknown"
        for t in TEST_SUITE:
            if t.query == r.query:
                cat = t.difficulty
                break
        if cat not in difficulty_stats:
            difficulty_stats[cat] = {
                "total": 0,
                "route_correct": 0,
                "answer_correct": 0,
                "e2e": 0,
            }
        difficulty_stats[cat]["total"] += 1
        if r.route_correct:
            difficulty_stats[cat]["route_correct"] += 1
        if r.answer_correct:
            difficulty_stats[cat]["answer_correct"] += 1
        if r.e2e_success:
            difficulty_stats[cat]["e2e"] += 1

    # Per-category breakdown
    category_stats: dict[str, dict] = {}
    for r in results:
        cat = "unknown"
        for t in TEST_SUITE:
            if t.query == r.query:
                cat = t.category
                break
        if cat not in category_stats:
            category_stats[cat] = {
                "total": 0,
                "route_correct": 0,
                "answer_correct": 0,
                "e2e": 0,
            }
        category_stats[cat]["total"] += 1
        if r.route_correct:
            category_stats[cat]["route_correct"] += 1
        if r.answer_correct:
            category_stats[cat]["answer_correct"] += 1
        if r.e2e_success:
            category_stats[cat]["e2e"] += 1

    return {
        "version": "0.4.0",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "run_count": run_count,
        "summary": {
            "total_queries": total,
            "composite_score": composite,
            "routing_accuracy": routing_score,
            "answer_correctness": correctness_score,
            "output_quality": quality_score,
            "robustness": robustness_score,
            "e2e_success_rate": e2e_score,
            "avg_latency_ms": avg_latency,
            "p50_latency_ms": p50_latency,
            "p95_latency_ms": p95_latency,
        },
        "dimensions": {
            "routing": {
                "correct": routing_correct,
                "total": total,
                "score": routing_score,
                "description": "Did the agent route to the correct tool?",
            },
            "correctness": {
                "correct": answer_correct_count,
                "total": total,
                "score": correctness_score,
                "description": "Is the numerical/factual answer correct?",
            },
            "quality": {
                "keyword_hits": keyword_hits,
                "total": total,
                "score": quality_score,
                "description": "Does the output contain expected answer keywords?",
            },
            "robustness": {
                "no_crash": no_crash_count,
                "total": total,
                "score": robustness_score,
                "description": "Does the agent survive edge-case inputs?",
            },
            "e2e": {
                "success": e2e_success_count,
                "total": total,
                "score": e2e_score,
                "description": "Route correct + answer correct + no duplicates + non-empty",
            },
        },
        "per_tool": {
            tool: {
                "total": stats["total"],
                "routing_accuracy": round(
                    stats["route_correct"] / stats["total"] * 100, 1
                ),
                "answer_accuracy": round(
                    stats["answer_correct"] / stats["total"] * 100, 1
                ),
                "keyword_accuracy": round(
                    stats["keyword_hit"] / stats["total"] * 100, 1
                ),
                "e2e_rate": round(stats["e2e"] / stats["total"] * 100, 1),
            }
            for tool, stats in tool_stats.items()
        },
        "per_difficulty": {
            diff: {
                "total": stats["total"],
                "routing_accuracy": round(
                    stats["route_correct"] / stats["total"] * 100, 1
                ),
                "answer_accuracy": round(
                    stats["answer_correct"] / stats["total"] * 100, 1
                ),
                "e2e_rate": round(stats["e2e"] / stats["total"] * 100, 1),
            }
            for diff, stats in difficulty_stats.items()
        },
        "per_category": {
            cat: {
                "total": stats["total"],
                "routing_accuracy": round(
                    stats["route_correct"] / stats["total"] * 100, 1
                ),
                "answer_accuracy": round(
                    stats["answer_correct"] / stats["total"] * 100, 1
                ),
                "e2e_rate": round(stats["e2e"] / stats["total"] * 100, 1),
            }
            for cat, stats in category_stats.items()
        },
        "details": [
            {
                "query": r.query[:60] + ("..." if len(r.query) > 60 else ""),
                "expected_tool": r.expected_tool,
                "actual_tool": r.actual_tool,
                "route_correct": r.route_correct,
                "answer_correct": r.answer_correct,
                "keyword_hit": r.keyword_hit,
                "no_crash": r.no_crash,
                "no_dup": r.no_dup,
                "e2e_success": r.e2e_success,
                "latency_ms": round(r.latency_ms, 0),
                "error": r.error,
            }
            for r in results
        ],
    }


def print_report(report: dict) -> None:
    """Print a human-readable evaluation report."""
    s = report["summary"]
    d = report["dimensions"]

    print("\n" + "=" * 70)
    print("CARM Evaluation Report v" + report["version"])
    print("=" * 70)
    print(f"  Timestamp:   {report['timestamp']}")
    print(f"  Queries:     {s['total_queries']}")
    print(f"  Runs:        {report.get('run_count', 1)}")
    print()

    print(f"  ** Composite Score: {s['composite_score']}/100 **")
    print()
    print("  Dimension Scores:")
    print(
        f"    Routing Accuracy:    {s['routing_accuracy']:5.1f}%  ({d['routing']['correct']}/{d['routing']['total']})"
    )
    print(
        f"    Answer Correctness:  {s['answer_correctness']:5.1f}%  ({d['correctness']['correct']}/{d['correctness']['total']})"
    )
    print(
        f"    Output Quality:      {s['output_quality']:5.1f}%  ({d['quality']['keyword_hits']}/{d['quality']['total']})"
    )
    print(
        f"    Robustness:          {s['robustness']:5.1f}%  ({d['robustness']['no_crash']}/{d['robustness']['total']})"
    )
    print(
        f"    E2E Success Rate:    {s['e2e_success_rate']:5.1f}%  ({d['e2e']['success']}/{d['e2e']['total']})"
    )
    print()
    print(f"  Latency:")
    print(
        f"    Avg:  {s['avg_latency_ms']:.0f}ms  P50: {s['p50_latency_ms']:.0f}ms  P95: {s['p95_latency_ms']:.0f}ms"
    )
    print()

    # Per-tool breakdown
    print("  Per-Tool Breakdown:")
    print(
        f"    {'Tool':20s}  {'Routing':>8s}  {'Answer':>8s}  {'Quality':>8s}  {'E2E':>6s}  {'Count':>5s}"
    )
    for tool, stats in report["per_tool"].items():
        print(
            f"    {tool:20s}  {stats['routing_accuracy']:7.1f}%  "
            f"{stats['answer_accuracy']:7.1f}%  "
            f"{stats['keyword_accuracy']:7.1f}%  "
            f"{stats['e2e_rate']:5.1f}%  "
            f"({stats['total']})"
        )
    print()

    # Per-difficulty breakdown
    print("  Per-Difficulty Breakdown:")
    for diff in ("easy", "normal", "hard"):
        if diff in report["per_difficulty"]:
            stats = report["per_difficulty"][diff]
            print(
                f"    {diff:8s}  routing={stats['routing_accuracy']:5.1f}%  "
                f"answer={stats['answer_accuracy']:5.1f}%  "
                f"e2e={stats['e2e_rate']:5.1f}%  ({stats['total']} queries)"
            )
    print()

    # Failure details
    failures = [d for d in report["details"] if not d["e2e_success"]]
    if failures:
        print(f"  Failures ({len(failures)}):")
        for f in failures:
            markers = []
            if not f["route_correct"]:
                markers.append("route")
            if not f["answer_correct"]:
                markers.append("answer")
            if not f["keyword_hit"]:
                markers.append("kw")
            if not f["no_crash"]:
                markers.append("CRASH")
            if not f["no_dup"]:
                markers.append("dup")
            print(
                f"    [{f['expected_tool']:15s} -> {str(f['actual_tool']):15s}] "
                f"{'+'.join(markers):15s} | {f['query'][:50]}"
            )
        print()

    print("=" * 70)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description="CARM evaluation")
    parser.add_argument(
        "--output",
        type=str,
        default="data/eval/carm_eval_report.json",
        help="Path to save the evaluation report JSON",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Limit number of test cases (0 = all)",
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=1,
        help="Run each test case N times for stability measurement",
    )
    args = parser.parse_args()

    # Setup
    sys.stdout.reconfigure(encoding="utf-8")

    from tools.base import ToolManager
    from tools.calc_tool import CalculatorTool
    from tools.search_tool import SearchTool
    from tools.code_tool import CodeExecutorTool
    from tools.bigmodel_tool import BigModelProxyTool
    from carm.runner import AgentRunner

    tm = ToolManager(
        [CalculatorTool(), SearchTool(), CodeExecutorTool(), BigModelProxyTool()]
    )
    runner = AgentRunner(tm, max_steps=10)

    # Run evaluation
    test_cases = TEST_SUITE[: args.limit] if args.limit else TEST_SUITE
    all_results: list[EvalResult] = []

    for run_idx in range(args.repeat):
        if args.repeat > 1:
            print(f"\n--- Run {run_idx + 1}/{args.repeat} ---")
        for i, test in enumerate(test_cases, 1):
            label = test.query[:40] + ("..." if len(test.query) > 40 else "")
            print(f"  [{i:2d}/{len(test_cases)}] {label}", end="", flush=True)
            result = evaluate_single(runner, test)
            all_results.append(result)
            status = "PASS" if result.e2e_success else "FAIL"
            print(
                f" -> {status} (tool: {result.actual_tool}, "
                f"answer: {'Y' if result.answer_correct else 'N'}, "
                f"{result.latency_ms:.0f}ms)"
            )

    # Generate report
    report = generate_report(all_results, run_count=args.repeat)
    print_report(report)

    # Save report
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"Report saved to {output_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
