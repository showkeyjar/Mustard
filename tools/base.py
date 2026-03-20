from __future__ import annotations

from typing import Protocol

from carm.schemas import ToolResult


class Tool(Protocol):
    name: str

    def execute(self, query: str, arguments: dict) -> ToolResult:
        ...


class ToolManager:
    def __init__(self, tools: list[Tool]) -> None:
        self._tools = {tool.name: tool for tool in tools}

    def execute(self, tool_name: str, query: str, arguments: dict | None = None) -> ToolResult:
        tool = self._tools[tool_name]
        return tool.execute(query, arguments or {})
