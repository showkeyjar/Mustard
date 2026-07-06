"""Cross-validation test for CARM dynamic tool registration.

Tests realistic scenarios where users register custom tools alongside
or instead of default tools, verifying that CARM routes correctly
across all IntentCategory × tool registration combinations.

Test dimensions:
1. Single-category multi-tool: 2+ tools for same category, who wins?
2. Overlapping tags: tool with [SEARCH, CONSULT] vs [SEARCH] only
3. Category gap: missing CODE tool, where does code intent go?
4. Mixed default+custom: default tools + custom SEARCH tool
5. Multi-intent split: "先搜索再计算" with custom tools
6. Registration order: first vs last registered for same category
"""

from __future__ import annotations

import os
import sys

sys.stdout.reconfigure(encoding="utf-8")
os.environ["CARM_NO_EMBEDDING"] = "1"

from pathlib import Path
from typing import Any

from carm.intent import IntentCategory
from carm.policy import OnlinePolicy
from carm.schemas import ToolResult
from carm.session_memory import SessionMemoryManager
from tools.base import ToolManager
from tools.calc_tool import CalculatorTool
from tools.code_tool import CodeExecutorTool
from tools.search_tool import SearchTool
from tools.bigmodel_tool import BigModelProxyTool


# ── Test infrastructure ───────────────────────────────────────────────


def _make_tool(name: str, tags: list[IntentCategory]) -> type:
    """Create a minimal tool class with given name and capability_tags."""

    class DynTool:
        pass

    DynTool.name = name
    DynTool.capability_tags = tags

    def execute(self, query, arguments):
        return ToolResult(
            ok=True,
            tool_name=self.name,
            result=f"[{self.name}] {query}",
            confidence=0.9,
            source=self.name,
        )

    DynTool.execute = execute
    return DynTool


def _route(policy: OnlinePolicy, query: str) -> str | None:
    """Route a query and return the tool name."""
    from carm.actions import Action
    from carm.memory import MemoryBoard, MemorySlot
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


def _make_policy(tool_manager: ToolManager | None = None) -> OnlinePolicy:
    return OnlinePolicy(
        Path("data/experience/policy_state.json"),
        Path("data/experience/concept_state.json"),
        tool_manager=tool_manager,
    )


# ── Test 1: Single-category multi-tool ────────────────────────────────


def test_single_category_multi_tool():
    """When 2+ tools declare the same category, the FIRST registered wins."""
    tm = ToolManager()
    tm.register(_make_tool("weather_api", [IntentCategory.SEARCH])())
    tm.register(_make_tool("news_api", [IntentCategory.SEARCH])())
    tm.register(_make_tool("math_solver", [IntentCategory.CALC])())

    policy = _make_policy(tm)
    result = _route(policy, "今天天气怎么样")

    # First registered SEARCH tool should win
    assert result == "weather_api", (
        f"Expected 'weather_api' (first registered), got '{result}'"
    )
    # Verify ToolManager order
    search_tools = tm.tools_for_category(IntentCategory.SEARCH)
    assert search_tools == ["weather_api", "news_api"], (
        f"SEARCH tools order: {search_tools}"
    )
    print(f"  ✓ Multi-tool same category: first registered wins ({result})")


# ── Test 2: Overlapping tags ──────────────────────────────────────────


def test_overlapping_tags():
    """Tool with [SEARCH, CONSULT] vs [SEARCH] only — both handle SEARCH,
    first registered wins for SEARCH; overlapping tool also handles CONSULT."""
    tm = ToolManager()
    # Multi-capability tool registered first
    tm.register(
        _make_tool("knowledge_hub", [IntentCategory.SEARCH, IntentCategory.CONSULT])()
    )
    # Single-capability tool registered second
    tm.register(_make_tool("web_search", [IntentCategory.SEARCH])())

    policy = _make_policy(tm)

    search_result = _route(policy, "搜索一下Python教程")
    consult_result = _route(policy, "写一篇关于AI的议论文")

    assert search_result == "knowledge_hub", (
        f"SEARCH should route to 'knowledge_hub' (first registered), got '{search_result}'"
    )
    assert consult_result == "knowledge_hub", (
        f"CONSULT should route to 'knowledge_hub', got '{consult_result}'"
    )
    print(f"  ✓ Overlapping tags: SEARCH→{search_result}, CONSULT→{consult_result}")


# ── Test 3: Category gap (missing tool) ───────────────────────────────


def test_category_gap():
    """When no tool handles CODE, code-intent queries should fallback
    to DEFAULT_TOOL_MAP (code_executor) or degrade gracefully."""
    tm = ToolManager()
    tm.register(_make_tool("weather_api", [IntentCategory.SEARCH])())
    tm.register(_make_tool("math_solver", [IntentCategory.CALC])())
    # NO CODE tool registered

    policy = _make_policy(tm)
    result = _route(policy, "帮我写一个冒泡排序的代码")

    # _resolve_tool_name(CODE) should fallback to DEFAULT_TOOL_MAP → "code_executor"
    # But "code_executor" is not actually registered in ToolManager
    # This tests the fallback behavior
    print(f"  Category gap (no CODE tool): code intent → '{result}'")
    # Acceptable: either default fallback name or a graceful alternative
    assert result in ("code_executor", "search", "weather_api"), (
        f"Unexpected fallback for missing CODE tool: '{result}'"
    )
    print(f"  ✓ Category gap: code intent gracefully fell back to '{result}'")


# ── Test 4: Mixed default + custom ────────────────────────────────────


def test_mixed_default_custom():
    """Default 4 tools + 1 custom SEARCH tool — custom is registered
    AFTER defaults, so default 'search' should win for SEARCH intent."""
    tm = ToolManager()
    tm.register(CalculatorTool())
    tm.register(CodeExecutorTool())
    tm.register(SearchTool())
    tm.register(BigModelProxyTool())
    # Custom tool registered AFTER defaults
    tm.register(_make_tool("weather_api", [IntentCategory.SEARCH])())

    policy = _make_policy(tm)

    search_result = _route(policy, "今天天气怎么样")
    calc_result = _route(policy, "3加5等于多少")

    assert search_result == "search", (
        f"Default 'search' should win (registered first), got '{search_result}'"
    )
    assert calc_result == "calculator", (
        f"Default 'calculator' should handle calc, got '{calc_result}'"
    )

    # Verify: weather_api is second in SEARCH list
    search_tools = tm.tools_for_category(IntentCategory.SEARCH)
    assert "weather_api" in search_tools, (
        f"weather_api not in SEARCH tools: {search_tools}"
    )
    print(f"  ✓ Mixed default+custom: SEARCH→{search_result}, CALC→{calc_result}")


# ── Test 5: Custom tools ONLY (no defaults) ───────────────────────────


def test_custom_only():
    """Only custom tools, no defaults — verify all intent categories
    route to the correct custom tools."""
    tm = ToolManager()
    tm.register(_make_tool("weather_api", [IntentCategory.SEARCH])())
    tm.register(_make_tool("math_solver", [IntentCategory.CALC])())
    tm.register(_make_tool("code_runner", [IntentCategory.CODE])())
    tm.register(_make_tool("llm_service", [IntentCategory.CONSULT])())

    policy = _make_policy(tm)

    cases = [
        ("今天天气怎么样", "weather_api"),
        ("3加5等于多少", "math_solver"),
        ("帮我写一个排序的代码", "code_runner"),
        ("写一篇关于AI的议论文", "llm_service"),
        ("翻译一下这段英文", "llm_service"),
    ]

    all_ok = True
    for query, expected in cases:
        result = _route(policy, query)
        if result != expected:
            print(f"  ✗ '{query[:20]}' expected={expected} actual={result}")
            all_ok = False
        else:
            print(f"  ✓ '{query[:20]}' → {result}")

    assert all_ok, "Some custom-only routing tests failed"


# ── Test 6: Multi-intent with custom tools ────────────────────────────


def test_multi_intent_custom_tools():
    """Multi-intent query "先搜索再计算" with custom tools should
    split into sub-intents and each sub-intent routes to the right tool."""
    tm = ToolManager()
    tm.register(_make_tool("weather_api", [IntentCategory.SEARCH])())
    tm.register(_make_tool("math_solver", [IntentCategory.CALC])())

    policy = _make_policy(tm)

    # Test that multi-intent detection still works with custom tools
    from carm.signals import has_multi_intent_signal, split_multi_intent

    query = "帮我查一下北京天气顺便算一下3加5"
    assert has_multi_intent_signal(query), "Should detect multi-intent"

    splits = split_multi_intent(query)
    assert len(splits) >= 2, f"Should split into 2+ intents, got {len(splits)}"

    # Verify split intents use IntentCategory
    for s in splits:
        assert isinstance(s.primary_signal, (IntentCategory, str)), (
            f"Split signal should be IntentCategory or str, got {type(s.primary_signal)}"
        )

    # Route the full query — should hit multi_intent handler
    result = _route(policy, query)
    # The query should either be detected as multi_intent or route to one of the sub-tools
    print(f"  Multi-intent with custom tools: '{query[:30]}' → {result}")
    # Acceptable: multi_intent pseudo-tool, or one of the sub-tools
    assert result in ("multi_intent", "weather_api", "math_solver"), (
        f"Unexpected multi-intent routing: '{result}'"
    )
    print(f"  ✓ Multi-intent routing with custom tools: {result}")


# ── Test 7: Registration order matters ────────────────────────────────


def test_registration_order():
    """Registering custom SEARCH tool BEFORE default search should make
    the custom tool the primary handler for SEARCH intent."""
    tm = ToolManager()
    # Custom tool registered FIRST
    tm.register(_make_tool("enterprise_search", [IntentCategory.SEARCH])())
    # Default tools registered AFTER
    tm.register(CalculatorTool())
    tm.register(CodeExecutorTool())
    tm.register(BigModelProxyTool())

    policy = _make_policy(tm)
    result = _route(policy, "搜索一下Python教程")

    assert result == "enterprise_search", (
        f"Custom search registered first should win, got '{result}'"
    )
    print(f"  ✓ Registration order: custom-first → {result}")


# ── Test 8: Tool with no capability_tags (backward compat) ────────────


def test_no_capability_tags():
    """A tool without capability_tags should still be usable via
    ToolManager's name-based inference."""
    tm = ToolManager()

    # Simulate a legacy tool without capability_tags
    class LegacyTool:
        name = "my_legacy_tool"

        # No capability_tags attribute
        def execute(self, query, arguments):
            return ToolResult(
                ok=True,
                tool_name=self.name,
                result="legacy",
                confidence=0.9,
                source="legacy",
            )

    tm.register(LegacyTool())
    # Legacy tool with unknown name → no capability inference
    # But it should still be callable by name
    assert tm.has_tool("my_legacy_tool"), "Legacy tool should be registered"
    # No category should map to it
    for cat in IntentCategory:
        matches = tm.tools_for_category(cat)
        assert "my_legacy_tool" not in matches, (
            f"Legacy tool shouldn't appear in {cat}: {matches}"
        )

    # But it can still be executed by name
    result = tm.execute("my_legacy_tool", "test", {})
    assert result.ok, "Legacy tool should execute by name"
    print(f"  ✓ No capability_tags: legacy tool callable by name, not by category")


# ── Test 9: Policy without ToolManager (pure default) ─────────────────


def test_no_tool_manager():
    """Policy created without tool_manager should use DEFAULT_TOOL_MAP."""
    policy = _make_policy()  # No tool_manager

    cases = [
        ("3加5等于多少", "calculator"),
        ("帮我写一个排序代码", "code_executor"),
        ("今天天气怎么样", "search"),
        ("写一篇作文", "bigmodel_proxy"),
    ]

    all_ok = True
    for query, expected in cases:
        result = _route(policy, query)
        if result != expected:
            print(f"  ✗ No TM: '{query[:20]}' expected={expected} actual={result}")
            all_ok = False
        else:
            print(f"  ✓ No TM: '{query[:20]}' → {result}")

    assert all_ok, "Some no-ToolManager tests failed"


# ── Test 10: set_primary overrides registration order ──────────────────


def test_set_primary():
    """After registering defaults + custom, use set_primary to promote
    the custom tool without re-registering."""
    tm = ToolManager()
    tm.register(CalculatorTool())
    tm.register(SearchTool())
    tm.register(BigModelProxyTool())

    class WeatherAPI:
        name = "weather_api"
        capability_tags = [IntentCategory.SEARCH]

        def execute(self, q, a):
            return ToolResult(
                ok=True,
                tool_name=self.name,
                result="weather",
                confidence=0.9,
                source="weather_api",
            )

    tm.register(WeatherAPI())

    # Before set_primary: default 'search' wins
    assert tm.find_by_capability(IntentCategory.SEARCH) == "search", (
        "Default 'search' should win before set_primary"
    )

    # Promote weather_api
    tm.set_primary("weather_api", IntentCategory.SEARCH)

    # After set_primary: custom tool wins
    assert tm.find_by_capability(IntentCategory.SEARCH) == "weather_api", (
        "weather_api should win after set_primary"
    )

    # Verify routing through policy
    policy = _make_policy(tm)
    result = _route(policy, "今天天气怎么样")
    assert result == "weather_api", (
        f"Policy should route SEARCH to weather_api after set_primary, got '{result}'"
    )
    print(f"  ✓ set_primary: weather_api promoted to primary SEARCH tool → {result}")


# ── Test 11: Multiple categories, partial custom ──────────────────────


def test_partial_custom():
    """Custom tools for SEARCH+CALC, defaults for CODE+CONSULT.
    Verify each category routes to the right tool (custom or default)."""
    tm = ToolManager()
    # Custom SEARCH + CALC
    tm.register(_make_tool("enterprise_search", [IntentCategory.SEARCH])())
    tm.register(_make_tool("precision_calc", [IntentCategory.CALC])())
    # Default CODE + CONSULT
    tm.register(CodeExecutorTool())
    tm.register(BigModelProxyTool())

    policy = _make_policy(tm)

    cases = [
        ("今天天气怎么样", "enterprise_search"),
        ("3加5等于多少", "precision_calc"),
        ("帮我写一个排序代码", "code_executor"),
        ("写一篇作文", "bigmodel_proxy"),
    ]

    all_ok = True
    for query, expected in cases:
        result = _route(policy, query)
        if result != expected:
            print(f"  ✗ '{query[:20]}' expected={expected} actual={result}")
            all_ok = False
        else:
            print(f"  ✓ '{query[:20]}' → {result}")

    assert all_ok, "Some partial-custom tests failed"


# ── Run all tests ─────────────────────────────────────────────────────


def main():
    tests = [
        ("Single-category multi-tool", test_single_category_multi_tool),
        ("Overlapping tags", test_overlapping_tags),
        ("Category gap", test_category_gap),
        ("Mixed default + custom", test_mixed_default_custom),
        ("Custom tools only", test_custom_only),
        ("Multi-intent custom tools", test_multi_intent_custom_tools),
        ("Registration order", test_registration_order),
        ("No capability_tags", test_no_capability_tags),
        ("No ToolManager (pure default)", test_no_tool_manager),
        ("set_primary override", test_set_primary),
        ("Partial custom (mixed)", test_partial_custom),
    ]

    print("=" * 70)
    print("CARM DYNAMIC TOOL CROSS-VALIDATION")
    print("=" * 70)
    print()

    passed = 0
    failed = 0
    errors = []

    for name, test_fn in tests:
        print(f"[{name}]")
        try:
            test_fn()
            passed += 1
        except AssertionError as e:
            failed += 1
            errors.append((name, str(e)))
            print(f"  ✗ FAILED: {e}")
        except Exception as e:
            failed += 1
            errors.append((name, f"EXCEPTION: {e}"))
            print(f"  ✗ EXCEPTION: {e}")
        print()

    print("=" * 70)
    print(f"RESULTS: {passed}/{passed + failed} passed")
    if errors:
        print("FAILURES:")
        for name, err in errors:
            print(f"  [{name}] {err}")
    print("=" * 70)

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
