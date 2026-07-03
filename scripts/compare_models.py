#!/usr/bin/env python3
"""Cross-model comparison: CARM vs LLM baselines on the same test cases.

Runs the same benchmark queries through multiple models and compares:
1. Tool routing accuracy (did they pick the right tool?)
2. Answer quality (for calculator cases, is the answer numerically correct?)
3. Latency (how fast?)

Models tested:
- CARM: local policy routing + calc_tool
- Gemma3:12b: Ollama local model with tool-selection prompt
- Cloud models (if Ollama cloud endpoints available): kimi, glm, etc.

Usage:
    PYTHONPATH=. python scripts/compare_models.py
    PYTHONPATH=. python scripts/compare_models.py --models carm gemma3:12b
    PYTHONPATH=. python scripts/compare_models.py --benchmark smp  # only SMP2017
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Re-use the same test cases from the benchmark script
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.evaluate_carm_benchmark import (
    SMP2017_CASES,
    MATH23K_CASES,
    BFCL_CASES,
    MMLU_CN_CASES,
)


# ===========================================================================
# Tool-selection prompt for LLM baselines
# ===========================================================================

TOOL_PROMPT = """你是一个AI助手，需要根据用户问题选择最合适的工具。

可用工具：
1. calculator — 数学计算（加减乘除、面积、单位转换、折扣、百分比等）
2. code_executor — 运行代码（写算法、执行脚本、编程任务）
3. search — 搜索知识（查找信息、比较产品、解释概念、咨询建议）
4. bigmodel_proxy — 生成内容（写作、翻译、润色、总结、归纳、起草报告）

请仔细分析用户意图，选择最合适的工具。

用户问题：{query}

请按以下格式回答（不要输出其他内容）：
工具：[calculator/code_executor/search/bigmodel_proxy]
理由：[一句话解释为什么选这个工具]"""

CALC_ANSWER_PROMPT = """请计算以下问题，直接给出数字答案。

问题：{query}

请只输出最终数字答案，不要解释。如果无法计算，输出"无法计算"。"""


# ===========================================================================
# Model adapters
# ===========================================================================


@dataclass
class ModelResult:
    """Result from a single model on a single query."""

    query: str
    routed_tool: str | None = None
    answer: str | None = None
    answer_numeric: float | None = None
    latency_ms: float = 0.0
    error: str | None = None


def query_ollama(
    model: str, prompt: str, timeout: int = 30, host: str = "http://localhost:11434"
) -> str:
    """Send a prompt to an Ollama model and return the response text."""
    import urllib.request
    import urllib.error

    url = f"{host.rstrip('/')}/api/generate"
    payload = json.dumps(
        {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.1, "num_predict": 256},
        }
    ).encode("utf-8")

    req = urllib.request.Request(
        url,
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return data.get("response", "")
    except Exception as e:
        return f"ERROR: {e}"


def parse_tool_from_response(response: str) -> str | None:
    """Extract the selected tool from an LLM response."""
    # Look for explicit tool mention
    tool_map = {
        "calculator": "calculator",
        "code_executor": "code_executor",
        "code": "code_executor",
        "search": "search",
        "bigmodel_proxy": "bigmodel_proxy",
        "bigmodel": "bigmodel_proxy",
        "big model": "bigmodel_proxy",
    }
    lower = response.lower()

    # Try structured format first
    m = re.search(r"工具[：:]\s*(\S+)", response)
    if m:
        tool_raw = m.group(1).strip().lower()
        for key, val in tool_map.items():
            if key in tool_raw:
                return val

    # Fallback: scan for tool names in order of specificity
    for key, val in sorted(tool_map.items(), key=lambda x: -len(x[0])):
        if key in lower:
            return val

    return None


def parse_numeric_answer(response: str) -> float | None:
    """Extract a numeric answer from an LLM response."""
    # Try to find a standalone number (possibly with decimal point or minus)
    response = response.strip()
    # Remove common prefixes
    for prefix in ("答案是", "等于", "结果", "=", "：", ":"):
        response = response.replace(prefix, " ")
    # Find the last number in the response
    nums = re.findall(r"-?\d+\.?\d*", response)
    if nums:
        try:
            return float(nums[-1])
        except ValueError:
            return None
    return None


class CARMModel:
    """CARM model adapter — uses policy routing + calc_tool."""

    name = "CARM-v0.4"
    is_local = True

    def route(self, query: str) -> ModelResult:
        from carm.actions import Action
        from carm.memory import MemoryBoard, MemorySlot
        from carm.policy import OnlinePolicy
        from carm.state import AgentState

        t0 = time.perf_counter()
        try:
            policy = OnlinePolicy(
                state_path="data/carm/state.json",
                concept_state_path="data/carm/concept_state.json",
            )
            state = AgentState(step_idx=2, uncertainty=0.6, answer_ready=0.1)
            state.last_action = Action.WRITE_MEM.value
            memory = MemoryBoard()
            memory.write(
                MemorySlot(
                    slot_type="GOAL",
                    content=query,
                    confidence=0.9,
                    source="eval",
                    ttl=10,
                )
            )
            decision = policy.decide(state, memory, query)

            # Try up to 3 rounds of decision (like AgentRunner)
            tool = None
            for _ in range(3):
                if decision.action == Action.CALL_TOOL and decision.tool_call:
                    tool = decision.tool_call.tool_name
                    break
                elif decision.action == Action.CALL_BIGMODEL and decision.tool_call:
                    tool = decision.tool_call.tool_name
                    break
                elif decision.action == Action.WRITE_MEM:
                    memory.write(
                        MemorySlot(
                            slot_type=decision.target_slot or "PLAN",
                            content=query,
                            confidence=0.7,
                            source="eval",
                            ttl=10,
                        )
                    )
                    state.last_action = Action.WRITE_MEM.value
                    state.step_idx += 1
                    decision = policy.decide(state, memory, query)
                elif decision.action == Action.THINK:
                    state.last_action = Action.THINK.value
                    state.step_idx += 1
                    decision = policy.decide(state, memory, query)
                else:
                    break

            latency = (time.perf_counter() - t0) * 1000

            # If routed to calculator, also compute answer
            answer = None
            numeric = None
            if tool == "calculator":
                from tools.calc_tool import CalculatorTool

                calc = CalculatorTool()
                result = calc.execute(query, {})
                answer = result.result
                # Extract numeric result
                m = re.search(r"=\s*(-?\d+\.?\d*)", result.result)
                if m:
                    try:
                        numeric = float(m.group(1))
                        if numeric == int(numeric) and abs(numeric) < 1e15:
                            numeric = int(numeric)
                    except ValueError:
                        pass

            return ModelResult(
                query=query,
                routed_tool=tool,
                answer=answer,
                answer_numeric=numeric,
                latency_ms=latency,
            )
        except Exception as e:
            return ModelResult(
                query=query,
                error=str(e),
                latency_ms=(time.perf_counter() - t0) * 1000,
            )


class OllamaModel:
    """Ollama model adapter — uses chat prompt for tool selection."""

    def __init__(self, model: str, host: str = "http://localhost:11434"):
        self.model = model
        self.host = host
        self.name = model.replace(":latest", "").replace(":cloud", "")
        if host != "http://localhost:11434":
            # Tag remote models so we can distinguish in results
            short_host = (
                host.replace("http://", "").replace("https://", "").split(":")[0]
            )
            self.name = f"{self.name}@{short_host}"
        self.is_local = ":cloud" not in model and "localhost" in host

    def route(self, query: str) -> ModelResult:
        t0 = time.perf_counter()
        try:
            # Step 1: Tool selection
            prompt = TOOL_PROMPT.format(query=query)
            response = query_ollama(self.model, prompt, timeout=30, host=self.host)
            tool = parse_tool_from_response(response)

            latency = (time.perf_counter() - t0) * 1000

            # Step 2: If calculator, also get the answer
            answer = None
            numeric = None
            if tool == "calculator":
                calc_prompt = CALC_ANSWER_PROMPT.format(query=query)
                calc_response = query_ollama(
                    self.model, calc_prompt, timeout=30, host=self.host
                )
                answer = calc_response
                numeric = parse_numeric_answer(calc_response)

            return ModelResult(
                query=query,
                routed_tool=tool,
                answer=answer,
                answer_numeric=numeric,
                latency_ms=latency,
            )
        except Exception as e:
            return ModelResult(
                query=query,
                error=str(e),
                latency_ms=(time.perf_counter() - t0) * 1000,
            )
        except Exception as e:
            return ModelResult(
                query=query,
                error=str(e),
                latency_ms=(time.perf_counter() - t0) * 1000,
            )


# ===========================================================================
# Evaluation logic
# ===========================================================================


@dataclass
class BenchmarkComparison:
    benchmark: str
    models: list[str] = field(default_factory=list)
    # per-model, per-level stats
    stats: dict[str, dict[str, dict[str, int]]] = field(default_factory=dict)
    # per-case detail: {query: {model: {tool, correct, answer_correct, latency}}}
    details: list[dict] = field(default_factory=list)


def _get_expected_tool(case: dict) -> str | None:
    """Extract expected tool from a test case."""
    return case.get("expected_tool") or case.get("expected")


def _get_expected_answer(case: dict) -> float | None:
    """Extract expected numeric answer from a test case."""
    ans = case.get("expected_answer")
    if ans is not None:
        try:
            return float(ans)
        except (ValueError, TypeError):
            return None
    return None


def _is_routing_correct(actual: str | None, case: dict) -> bool:
    """Check if the routed tool matches expected."""
    expected = _get_expected_tool(case)
    if expected is None:
        return True  # no tool expectation

    # L4 cases: multi_intent/context_needed/multi_step are always wrong for single-tool models
    if expected in ("multi_intent", "context_needed", "multi_step"):
        return False

    acceptable = case.get("acceptable_tools", [expected])
    return actual in acceptable


def _is_answer_correct(actual: float | None, case: dict) -> bool | None:
    """Check if the numeric answer matches expected. Returns None if N/A."""
    expected = _get_expected_answer(case)
    if expected is None or actual is None:
        return None
    tolerance = case.get("tolerance", 0.01)
    return abs(actual - expected) <= tolerance * max(1, abs(expected))


def run_comparison(
    models: list[Any],
    cases: list[dict],
    benchmark_name: str,
) -> BenchmarkComparison:
    comp = BenchmarkComparison(benchmark=benchmark_name)
    comp.models = [m.name for m in models]

    for model in models:
        comp.stats[model.name] = {
            "L1": {"total": 0, "routing_correct": 0, "answer_correct": 0},
            "L2": {"total": 0, "routing_correct": 0, "answer_correct": 0},
            "L3": {"total": 0, "routing_correct": 0, "answer_correct": 0},
            "L4": {"total": 0, "routing_correct": 0, "answer_correct": 0},
        }

    for case in cases:
        level = case.get("level", "L1")
        row = {
            "query": case["query"],
            "level": level,
            "expected_tool": _get_expected_tool(case),
            "models": {},
        }

        for model in models:
            result = model.route(case["query"])
            routing_ok = _is_routing_correct(result.routed_tool, case)
            answer_ok = _is_answer_correct(result.answer_numeric, case)

            row["models"][model.name] = {
                "tool": result.routed_tool,
                "routing_correct": routing_ok,
                "answer": result.answer_numeric,
                "answer_correct": answer_ok,
                "latency_ms": round(result.latency_ms, 1),
                "error": result.error,
            }

            s = comp.stats[model.name][level]
            s["total"] += 1
            if routing_ok:
                s["routing_correct"] += 1
            if answer_ok is True:
                s["answer_correct"] += 1

        comp.details.append(row)

    return comp


def print_comparison(comp: BenchmarkComparison) -> None:
    print(f"\n{'=' * 80}")
    print(f"  {comp.benchmark} — Cross-Model Comparison")
    print(f"{'=' * 80}")

    # Per-level routing accuracy table
    for level in ("L1", "L2", "L3", "L4"):
        rows = []
        for model_name in comp.models:
            s = comp.stats[model_name][level]
            if s["total"] == 0:
                continue
            routing_pct = s["routing_correct"] / s["total"] * 100
            answer_total = sum(
                1
                for d in comp.details
                if d["level"] == level
                and d["models"].get(model_name, {}).get("answer_correct") is not None
            )
            answer_correct = sum(
                1
                for d in comp.details
                if d["level"] == level
                and d["models"].get(model_name, {}).get("answer_correct") is True
            )
            answer_pct = (
                (answer_correct / answer_total * 100) if answer_total > 0 else None
            )
            rows.append((model_name, routing_pct, answer_pct, s["total"]))

        if not rows:
            continue

        print(f"\n  {level} (Routing / Answer Accuracy):")
        print(f"  {'Model':20s}  {'Route%':>7s}  {'Answer%':>8s}  {'N':>3s}")
        print(f"  {'─' * 20}  {'─' * 7}  {'─' * 8}  {'─' * 3}")
        for name, rpct, apct, n in rows:
            apct_str = f"{apct:.0f}%" if apct is not None else "N/A"
            print(f"  {name:20s}  {rpct:6.1f}%  {apct_str:>8s}  {n:3d}")

    # Head-to-head: interesting disagreements
    if len(comp.models) >= 2:
        disagreements = []
        for d in comp.details:
            tools = {m: d["models"].get(m, {}).get("tool") for m in comp.models}
            unique_tools = set(tools.values())
            if len(unique_tools) > 1 and None not in unique_tools:
                disagreements.append(d)

        if disagreements:
            print(f"\n  Routing disagreements ({len(disagreements)} cases):")
            for d in disagreements[:10]:
                tools_str = " | ".join(
                    f"{m}→{d['models'][m]['tool']}" for m in comp.models
                )
                correct_str = " | ".join(
                    f"{'Y' if d['models'][m]['routing_correct'] else 'N'}"
                    for m in comp.models
                )
                print(f"    [{d['level']}] {d['query'][:45]}")
                print(f"      {tools_str}")
                print(f"      {correct_str}  (expected: {d['expected_tool']})")

    # Latency comparison
    print(f"\n  Latency (median ms):")
    for model_name in comp.models:
        latencies = [
            d["models"][model_name]["latency_ms"]
            for d in comp.details
            if d["models"].get(model_name, {}).get("latency_ms", 0) > 0
            and d["models"][model_name].get("error") is None
        ]
        if latencies:
            latencies.sort()
            median = latencies[len(latencies) // 2]
            print(f"    {model_name:20s}  {median:.0f} ms")


def save_comparison(comp: BenchmarkComparison, output_dir: Path) -> None:
    """Save comparison results to JSON."""
    output_dir.mkdir(parents=True, exist_ok=True)
    out = {
        "benchmark": comp.benchmark,
        "models": comp.models,
        "stats": comp.stats,
        "details": comp.details,
    }
    path = output_dir / f"compare_{comp.benchmark.lower().replace('-', '_')}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2, default=str)


# ===========================================================================
# Main
# ===========================================================================

BENCHMARK_MAP = {
    "smp": ("SMP2017-ECDT", SMP2017_CASES),
    "math23k": ("Math23K", MATH23K_CASES),
    "bfcl": ("BFCL-V3", BFCL_CASES),
    "mmlu": ("MMLU-CN", MMLU_CN_CASES),
}


def main():
    # Fix Windows terminal encoding
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description="Cross-model comparison")
    parser.add_argument(
        "--benchmark",
        choices=list(BENCHMARK_MAP.keys()) + ["all"],
        default="all",
        help="Which benchmark to run",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["carm", "gemma3:12b"],
        help="Models to compare. Format: 'carm' for CARM, or Ollama model names. "
        "Use 'model@host' syntax for remote Ollama, e.g. 'qwen3-coder@192.168.31.8'",
    )
    parser.add_argument(
        "--ollama-host",
        default="http://localhost:11434",
        help="Default Ollama host URL (overridden by @host syntax)",
    )
    parser.add_argument(
        "--output-dir",
        default="data/eval",
        help="Directory for JSON output",
    )
    args = parser.parse_args()

    # Build model adapters
    models: list[Any] = []
    for name in args.models:
        if name == "carm":
            models.append(CARMModel())
            continue
        # Support 'model@host' syntax for remote Ollama servers
        if "@" in name:
            model_name, host_addr = name.rsplit("@", 1)
            host = f"http://{host_addr}:11434"
        else:
            model_name = name
            host = args.ollama_host

        # Test if model is available
        print(f"  Checking {model_name} at {host}...", end=" ", flush=True)
        resp = query_ollama(model_name, "1+1", timeout=60, host=host)
        if resp.startswith("ERROR:"):
            print(f"UNAVAILABLE ({resp})")
            continue
        print("OK")
        models.append(OllamaModel(model_name, host=host))

    if len(models) < 2:
        print("Need at least 2 models for comparison. Available models:")
        print("  carm — CARM local policy")
        print("  gemma3:12b — Ollama local")
        print("  kimi-k2.6:cloud — Ollama cloud")
        print("  glm-5:cloud — Ollama cloud")
        sys.exit(1)

    # Select benchmarks
    if args.benchmark == "all":
        benchmarks = list(BENCHMARK_MAP.values())
    else:
        benchmarks = [BENCHMARK_MAP[args.benchmark]]

    print(f"\nComparing: {', '.join(m.name for m in models)}")
    print(f"Benchmarks: {', '.join(b[0] for b in benchmarks)}\n")

    output_dir = Path(args.output_dir)
    all_comparisons = []

    for bench_name, cases in benchmarks:
        print(f"Running {bench_name} ({len(cases)} cases)...", flush=True)
        comp = run_comparison(models, cases, bench_name)
        print_comparison(comp)
        save_comparison(comp, output_dir)
        all_comparisons.append(comp)

    # Summary
    print(f"\n{'=' * 80}")
    print(f"  OVERALL SUMMARY")
    print(f"{'=' * 80}")

    for model_name in models[0].name, models[-1].name if len(models) > 1 else None:
        if model_name is None:
            continue
        total_route = 0
        correct_route = 0
        total_answer = 0
        correct_answer = 0
        for comp in all_comparisons:
            for level in ("L1", "L2", "L3", "L4"):
                s = comp.stats.get(model_name, {}).get(level, {})
                total_route += s.get("total", 0)
                correct_route += s.get("routing_correct", 0)
                # Count answer-correct cases
                for d in comp.details:
                    if d["level"] == level:
                        ac = d["models"].get(model_name, {}).get("answer_correct")
                        if ac is not None:
                            total_answer += 1
                            if ac:
                                correct_answer += 1

        route_pct = correct_route / total_route * 100 if total_route else 0
        answer_pct = correct_answer / total_answer * 100 if total_answer else 0
        print(f"\n  {model_name}:")
        print(
            f"    Routing accuracy:   {correct_route}/{total_route} = {route_pct:.1f}%"
        )
        print(
            f"    Answer accuracy:    {correct_answer}/{total_answer} = {answer_pct:.1f}%"
        )

    # Key insight: CARM vs LLM tradeoff
    if len(models) >= 2:
        carm_name = models[0].name
        llm_name = models[1].name
        carm_total = sum(
            comp.stats[carm_name][l]["total"]
            for comp in all_comparisons
            for l in ("L1", "L2", "L3", "L4")
        )
        carm_correct = sum(
            comp.stats[carm_name][l]["routing_correct"]
            for comp in all_comparisons
            for l in ("L1", "L2", "L3", "L4")
        )
        llm_correct = sum(
            comp.stats[llm_name][l]["routing_correct"]
            for comp in all_comparisons
            for l in ("L1", "L2", "L3", "L4")
        )

        # Latency
        carm_lats = []
        llm_lats = []
        for comp in all_comparisons:
            for d in comp.details:
                cl = d["models"].get(carm_name, {}).get("latency_ms", 0)
                ll = d["models"].get(llm_name, {}).get("latency_ms", 0)
                if cl > 0:
                    carm_lats.append(cl)
                if ll > 0:
                    llm_lats.append(ll)
        carm_med = sorted(carm_lats)[len(carm_lats) // 2] if carm_lats else 0
        llm_med = sorted(llm_lats)[len(llm_lats) // 2] if llm_lats else 0

        print(f"\n  CARM vs {llm_name}:")
        print(
            f"    Routing:   {carm_name} {carm_correct / carm_total * 100:.1f}% vs {llm_name} {llm_correct / carm_total * 100:.1f}%"
        )
        print(
            f"    Latency:   {carm_name} {carm_med:.0f}ms vs {llm_name} {llm_med:.0f}ms"
        )
        if carm_med > 0 and llm_med > 0:
            print(f"    Speedup:   {llm_med / carm_med:.0f}x")

    print(f"\nResults saved to {output_dir}/")
    raise SystemExit(0)


if __name__ == "__main__":
    main()
