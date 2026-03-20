from __future__ import annotations

from carm.schemas import ToolResult


class SearchTool:
    name = "search"

    def execute(self, query: str, arguments: dict) -> ToolResult:
        top_k = arguments.get("top_k", 3)
        summary = (
            f"检索到 {top_k} 条对比材料。综合来看，问题可从成本、性能、维护复杂度和生态角度分析。"
        )
        return ToolResult(
            ok=True,
            tool_name=self.name,
            result=summary,
            confidence=0.72,
            source="tool/search",
        )
