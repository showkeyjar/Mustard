"""CARM Router — the primary public API for CARM.

Usage:
    from carm.router import CARMRouter

    # Quick start with default tools
    router = CARMRouter()
    result = router.route("3加5等于多少")
    print(result.tool_name)   # "calculator"
    print(result.result)      # "计算结果: 3 + 5 = 8"

    # With custom tools
    from carm.intent import IntentCategory

    class WeatherAPI:
        name = "weather_api"
        capability_tags = [IntentCategory.SEARCH]
        def execute(self, query, arguments): ...

    router = CARMRouter()
    router.register_tool(WeatherAPI())
    result = router.route("今天天气怎么样")
    print(result.tool_name)   # "weather_api"

    # Multi-turn conversation
    result1 = router.route("帮我查一下GPU的性能参数", session_id="s1")
    result2 = router.route("它的性能怎么样", session_id="s1")
    # result2 has anaphora resolved from result1
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from carm.actions import Action
from carm.intent import IntentCategory
from carm.memory import MemoryBoard, MemorySlot
from carm.policy import OnlinePolicy
from carm.session_memory import SessionMemoryManager
from carm.schemas import ToolResult
from carm.state import AgentState
from tools.base import ToolManager
from tools.calc_tool import CalculatorTool
from tools.code_tool import CodeExecutorTool
from tools.search_tool import SearchTool
from tools.bigmodel_tool import BigModelProxyTool


@dataclass
class RouteResult:
    """Structured result from CARM routing."""

    query: str
    tool_name: str
    result: str
    confidence: float
    source: str
    ok: bool = True
    intent_category: IntentCategory | None = None
    session_id: str = ""

    def __str__(self) -> str:
        return self.result


class CARMRouter:
    """One-line routing API for CARM.

    Routes natural language queries to the best-matching registered tool,
    executes it, and returns a structured result.

    Args:
        policy_path: Path to policy state file (optional, uses default).
        tool_manager: Custom ToolManager (optional, creates default with 4 tools).
        embedding: Whether to enable sentence-transformers Tier 2 (default: auto-detect).
    """

    def __init__(
        self,
        policy_path: str | Path | None = None,
        tool_manager: ToolManager | None = None,
        embedding: bool | None = None,
    ) -> None:
        # Embedding control
        if embedding is False or (
            embedding is None and os.environ.get("CARM_NO_EMBEDDING", "0") == "1"
        ):
            os.environ["CARM_NO_EMBEDDING"] = "1"

        # Policy
        if policy_path is None:
            policy_path = Path("data/experience/policy_state.json")
        self._policy_path = Path(policy_path)
        concept_path = self._policy_path.with_name("concept_state.json")

        # Tool manager
        if tool_manager is not None:
            self._tool_manager = tool_manager
        else:
            self._tool_manager = ToolManager()
            self._register_defaults()

        # Policy with tool manager reference
        self._policy = OnlinePolicy(
            self._policy_path, concept_path, tool_manager=self._tool_manager
        )

    def _register_defaults(self) -> None:
        """Register the 4 built-in tools."""
        for tool_cls in (
            CalculatorTool,
            CodeExecutorTool,
            SearchTool,
            BigModelProxyTool,
        ):
            try:
                self._tool_manager.register(tool_cls())
            except Exception:
                pass  # Skip if tool init fails (e.g. no API key)

    def register_tool(self, tool: Any) -> None:
        """Register a custom tool.

        The tool must have:
        - ``name: str`` — unique identifier
        - ``capability_tags: list[IntentCategory]`` — which intents it handles
        - ``execute(query: str, arguments: dict) -> ToolResult`` — execution method
        """
        self._tool_manager.register(tool)
        # Re-create policy with updated tool manager
        concept_path = self._policy_path.with_name("concept_state.json")
        self._policy = OnlinePolicy(
            self._policy_path, concept_path, tool_manager=self._tool_manager
        )

    def set_primary(self, tool_name: str, category: IntentCategory) -> None:
        """Promote a tool to be the primary handler for a category.

        Example:
            router.register_tool(weather_api)
            router.set_primary("weather_api", IntentCategory.SEARCH)
        """
        self._tool_manager.set_primary(tool_name, category)

    def route(
        self,
        query: str,
        session_id: str | None = None,
        dry_run: bool = False,
        timeout: float | None = None,
        **kwargs: Any,
    ) -> RouteResult:
        """Route a query to the best-matching tool and execute it.

        Args:
            query: Natural language query (e.g. "3加5等于多少").
            session_id: Optional session ID for multi-turn context.
                When provided, anaphora resolution ("它", "上次") uses
                previous turns in this session.
            dry_run: If True, only decide routing without executing the tool.
                The result field will contain the resolved query instead.
            timeout: Max seconds for tool execution. None = no limit.

        Returns:
            RouteResult with tool_name, result text, confidence, etc.
        """
        # 1. Anaphora resolution (if session context available)
        resolved_query = query
        if session_id:
            session_mgr = SessionMemoryManager.get_instance()
            ctx = session_mgr.get_or_create(session_id)
            if ctx.turns:
                _, resolved_query = session_mgr.resolve_query(session_id, query)

        # 2. Policy decision
        state = AgentState(step_idx=0, uncertainty=0.6, answer_ready=0.1)
        state.last_action = Action.WRITE_MEM.value
        memory = MemoryBoard()
        memory.write(
            MemorySlot(
                slot_type="GOAL",
                content=resolved_query,
                confidence=0.9,
                source="router",
                ttl=10,
            )
        )

        decision = self._policy.decide(state, memory, resolved_query)

        # 3. Execute (or dry-run)
        if decision.tool_call:
            tool_name = decision.tool_call.tool_name

            if dry_run:
                result = RouteResult(
                    query=query,
                    tool_name=tool_name,
                    result=resolved_query,
                    confidence=0.0,
                    source="carm/router:dry_run",
                    ok=True,
                    intent_category=decision.tool_call.intent_category
                    if hasattr(decision.tool_call, "intent_category")
                    else None,
                    session_id=session_id or "",
                )
            else:
                import signal as _signal

                tool_result = None
                timed_out = False

                if timeout is not None and hasattr(_signal, "SIGALRM"):
                    # Unix: use SIGALRM
                    def _handler(signum, frame):
                        raise TimeoutError("tool execution timed out")

                    old = _signal.signal(_signal.SIGALRM, _handler)
                    _signal.alarm(int(timeout))
                    try:
                        tool_result = self._tool_manager.execute(
                            tool_name,
                            decision.tool_call.query or resolved_query,
                            decision.tool_call.arguments or {},
                        )
                    except TimeoutError:
                        timed_out = True
                    finally:
                        _signal.alarm(0)
                        _signal.signal(_signal.SIGALRM, old)
                else:
                    # Windows or no timeout: just execute
                    tool_result = self._tool_manager.execute(
                        tool_name,
                        decision.tool_call.query or resolved_query,
                        decision.tool_call.arguments or {},
                    )

                if timed_out:
                    result = RouteResult(
                        query=query,
                        tool_name=tool_name,
                        result=f"工具执行超时（{timeout}s）",
                        confidence=0.0,
                        source="carm/router:timeout",
                        ok=False,
                        session_id=session_id or "",
                    )
                else:
                    result = RouteResult(
                        query=query,
                        tool_name=tool_result.tool_name,
                        result=tool_result.result,
                        confidence=tool_result.confidence,
                        source=tool_result.source,
                        ok=tool_result.ok,
                        session_id=session_id or "",
                    )
        else:
            # No tool call — policy chose THINK/ANSWER/VERIFY/etc.
            result = RouteResult(
                query=query,
                tool_name="none",
                result=decision.reason or "无法确定合适的工具",
                confidence=0.3,
                source="carm/router:no_tool",
                ok=False,
                session_id=session_id or "",
            )

        # 4. Record in session memory
        if session_id:
            session_mgr = SessionMemoryManager.get_instance()
            session_mgr.append_turn(
                session_id=session_id,
                user_input=query,
                tool_name=result.tool_name,
                tool_result=result.result,
                confidence=result.confidence,
            )

        return result

    @property
    def tool_names(self) -> list[str]:
        """List all registered tool names."""
        return self._tool_manager.tool_names

    def find_tool_for(self, category: IntentCategory) -> str | None:
        """Find which tool handles a given intent category."""
        return self._tool_manager.find_by_capability(category)
