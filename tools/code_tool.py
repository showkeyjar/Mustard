from __future__ import annotations

from carm.schemas import ToolResult


class CodeExecutorTool:
    name = "code_executor"

    def execute(self, query: str, arguments: dict) -> ToolResult:
        return ToolResult(
            ok=True,
            tool_name=self.name,
            result=f"代码执行完成，示例输出已生成。输入片段: {query[:40]}",
            confidence=0.68,
            source="tool/code_executor",
        )
