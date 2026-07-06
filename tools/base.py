from __future__ import annotations

from typing import Protocol

from carm.intent import IntentCategory
from carm.schemas import ToolResult


class Tool(Protocol):
    name: str
    capability_tags: list[IntentCategory]

    def execute(self, query: str, arguments: dict) -> ToolResult: ...


class ToolManager:
    def __init__(self, tools: list[Tool] | None = None) -> None:
        self._tools: dict[str, Tool] = {}
        # Index: IntentCategory → list of tool names that handle it
        self._capability_index: dict[IntentCategory, list[str]] = {
            cat: [] for cat in IntentCategory
        }
        if tools:
            for tool in tools:
                self.register(tool)

    def register(self, tool: Tool) -> None:
        """Register a tool with its capability tags."""
        self._tools[tool.name] = tool
        tags = getattr(tool, "capability_tags", [])
        if not tags:
            # Backward compatibility: infer from tool name
            tags = self._infer_tags(tool.name)
        for tag in tags:
            if tag not in self._capability_index:
                self._capability_index[tag] = []
            if tool.name not in self._capability_index[tag]:
                self._capability_index[tag].append(tool.name)

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

    def find_by_capability(
        self, category: IntentCategory, fallback: str | None = None
    ) -> str | None:
        """Find the best tool for a given intent category.

        Returns the name of the first registered tool that declares
        this capability.  If no tool matches, returns `fallback`.
        """
        candidates = self._capability_index.get(category, [])
        if candidates:
            return candidates[0]
        return fallback

    def tools_for_category(self, category: IntentCategory) -> list[str]:
        """Return all tool names that handle a given category."""
        return list(self._capability_index.get(category, []))

    def has_tool(self, tool_name: str) -> bool:
        return tool_name in self._tools

    @property
    def tool_names(self) -> list[str]:
        return list(self._tools.keys())

    @staticmethod
    def _infer_tags(name: str) -> list[IntentCategory]:
        """Infer capability tags from conventional tool names (backward compat)."""
        mapping: dict[str, list[IntentCategory]] = {
            "calculator": [IntentCategory.CALC],
            "code_executor": [IntentCategory.CODE],
            "search": [IntentCategory.SEARCH],
            "bigmodel_proxy": [IntentCategory.CONSULT, IntentCategory.SEARCH],
        }
        return mapping.get(name, [])
