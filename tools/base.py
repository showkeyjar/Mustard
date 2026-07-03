from __future__ import annotations

from typing import Protocol

from carm.schemas import ToolResult


class Tool(Protocol):
    name: str

    def execute(self, query: str, arguments: dict) -> ToolResult: ...


class ToolManager:
    def __init__(self, tools: list[Tool]) -> None:
        self._tools = {tool.name: tool for tool in tools}

    def execute(
        self, tool_name: str, query: str, arguments: dict | None = None
    ) -> ToolResult:
        tool = self._tools.get(tool_name)
        if tool is None:
            return ToolResult(
                ok=False,
                tool_name=tool_name,
                result=f"工具 `{tool_name}` 不可用，请检查工具注册配置",
                confidence=0.1,
                source="tool_manager:fallback",
            )
        return tool.execute(query, arguments or {})
