from __future__ import annotations

from carm.schemas import ToolResult


class BigModelProxyTool:
    name = "bigmodel_proxy"

    def execute(self, query: str, arguments: dict) -> ToolResult:
        return ToolResult(
            ok=True,
            tool_name=self.name,
            result=f"外部大模型建议: 先拆解任务，再补充缺失专业事实。原问题: {query[:60]}",
            confidence=0.8,
            source="tool/bigmodel_proxy",
        )
