"""Test CARM's dynamic tool registration and routing.

Verifies that CARM can correctly route to user-registered tools
instead of only the 4 hardcoded default tools.
"""

from __future__ import annotations

import os
import sys

sys.stdout.reconfigure(encoding="utf-8")
os.environ["CARM_NO_EMBEDDING"] = "1"

from pathlib import Path

from carm.intent import IntentCategory
from carm.policy import OnlinePolicy
from carm.schemas import ToolResult
from tools.base import ToolManager


# ── Custom tools ──────────────────────────────────────────────────────


class WeatherAPI:
    """A custom weather API tool — handles SEARCH intents."""

    name = "weather_api"
    capability_tags = [IntentCategory.SEARCH]

    def execute(self, query: str, arguments: dict) -> ToolResult:
        return ToolResult(
            ok=True,
            tool_name=self.name,
            result=f"Weather data for: {query}",
            confidence=0.9,
            source="weather_api",
        )


class TranslationService:
    """A custom translation tool — handles CONSULT intents."""

    name = "translation_service"
    capability_tags = [IntentCategory.CONSULT]

    def execute(self, query: str, arguments: dict) -> ToolResult:
        return ToolResult(
            ok=True,
            tool_name=self.name,
            result=f"Translated: {query}",
            confidence=0.9,
            source="translation_service",
        )


class MathSolver:
    """A custom math solver — handles CALC intents."""

    name = "math_solver"
    capability_tags = [IntentCategory.CALC]

    def execute(self, query: str, arguments: dict) -> ToolResult:
        return ToolResult(
            ok=True,
            tool_name=self.name,
            result=f"Solved: {query}",
            confidence=0.9,
            source="math_solver",
        )


class DataAnalyzer:
    """A custom data analysis tool — handles CODE intents."""

    name = "data_analyzer"
    capability_tags = [IntentCategory.CODE]

    def execute(self, query: str, arguments: dict) -> ToolResult:
        return ToolResult(
            ok=True,
            tool_name=self.name,
            result=f"Analyzed: {query}",
            confidence=0.9,
            source="data_analyzer",
        )


# ── Test cases ────────────────────────────────────────────────────────

DYNAMIC_TEST_CASES = [
    # SEARCH intent → should route to weather_api (not "search")
    {
        "query": "今天天气怎么样",
        "expected_tool": "weather_api",
        "intent": IntentCategory.SEARCH,
    },
    {
        "query": "帮我搜索一下最近的新闻",
        "expected_tool": "weather_api",
        "intent": IntentCategory.SEARCH,
    },
    # CALC intent → should route to math_solver (not "calculator")
    {
        "query": "3加5等于多少",
        "expected_tool": "math_solver",
        "intent": IntentCategory.CALC,
    },
    {
        "query": "1万亿除以14亿",
        "expected_tool": "math_solver",
        "intent": IntentCategory.CALC,
    },
    # CODE intent → should route to data_analyzer (not "code_executor")
    {
        "query": "帮我写一个冒泡排序的代码",
        "expected_tool": "data_analyzer",
        "intent": IntentCategory.CODE,
    },
    {
        "query": "用python实现快速排序",
        "expected_tool": "data_analyzer",
        "intent": IntentCategory.CODE,
    },
    # CONSULT intent → should route to translation_service (not "bigmodel_proxy")
    {
        "query": "帮我翻译一下这段英文",
        "expected_tool": "translation_service",
        "intent": IntentCategory.CONSULT,
    },
    {
        "query": "写一篇关于AI的议论文",
        "expected_tool": "translation_service",
        "intent": IntentCategory.CONSULT,
    },
]


def _route_query(policy: OnlinePolicy, query: str) -> str | None:
    """Route a query using the policy and return the tool name."""
    from carm.actions import Action
    from carm.memory import MemoryBoard, MemorySlot
    from carm.session_memory import SessionMemoryManager
    from carm.state import AgentState

    SessionMemoryManager.reset_instance()

    state = AgentState(step_idx=0, uncertainty=0.6, answer_ready=0.1)
    state.last_action = Action.WRITE_MEM.value
    memory = MemoryBoard()
    memory.write(
        MemorySlot(
            slot_type="GOAL", content=query, confidence=0.9, source="test", ttl=10
        )
    )

    decision = policy.decide(state, memory, query)
    if decision.tool_call:
        return decision.tool_call.tool_name
    return decision.action.value


def main() -> None:
    # Register custom tools (NO default tools)
    tm = ToolManager()
    tm.register(WeatherAPI())
    tm.register(TranslationService())
    tm.register(MathSolver())
    tm.register(DataAnalyzer())

    # Create policy with the custom tool manager
    policy = OnlinePolicy(
        Path("data/experience/policy_state.json"),
        Path("data/experience/concept_state.json"),
        tool_manager=tm,
    )

    # Verify tool manager setup
    print("=" * 60)
    print("DYNAMIC TOOL REGISTRATION TEST")
    print("=" * 60)
    print(f"Registered tools: {tm.tool_names}")
    print(f"SEARCH  → {tm.find_by_capability(IntentCategory.SEARCH)}")
    print(f"CALC    → {tm.find_by_capability(IntentCategory.CALC)}")
    print(f"CODE    → {tm.find_by_capability(IntentCategory.CODE)}")
    print(f"CONSULT → {tm.find_by_capability(IntentCategory.CONSULT)}")
    print()

    passed = 0
    failed = 0
    for case in DYNAMIC_TEST_CASES:
        actual = _route_query(policy, case["query"])
        expected = case["expected_tool"]
        ok = actual == expected
        symbol = "✓" if ok else "✗"
        if ok:
            passed += 1
        else:
            failed += 1
        print(
            f"  {symbol} query='{case['query'][:30]}' "
            f"expected={expected} actual={actual}"
        )

    print()
    total = passed + failed
    print(f"Results: {passed}/{total} passed ({passed / total * 100:.0f}%)")
    if failed > 0:
        print(f"  FAILED: {failed} cases did not route to the expected custom tool")

    # Also test: default tools still work when no tool_manager is provided
    print()
    print("-" * 60)
    print("BACKWARD COMPAT TEST (no tool_manager → default tools)")
    print("-" * 60)

    default_policy = OnlinePolicy(
        Path("data/experience/policy_state.json"),
        Path("data/experience/concept_state.json"),
    )
    compat_cases = [
        ("3加5等于多少", "calculator"),
        ("帮我写一个排序代码", "code_executor"),
        ("今天天气怎么样", "search"),
        ("写一篇作文", "bigmodel_proxy"),
    ]
    compat_passed = 0
    for query, expected in compat_cases:
        actual = _route_query(default_policy, query)
        ok = actual == expected
        if ok:
            compat_passed += 1
        else:
            print(f"  ✗ query='{query}' expected={expected} actual={actual}")

    print(f"  Backward compat: {compat_passed}/{len(compat_cases)} passed")

    return 0 if failed == 0 and compat_passed == len(compat_cases) else 1


if __name__ == "__main__":
    sys.exit(main())
