from __future__ import annotations

import re

from carm.schemas import ToolResult


class CalculatorTool:
    name = "calculator"

    def execute(self, query: str, arguments: dict) -> ToolResult:
        expression = "".join(re.findall(r"[0-9\.\+\-\*\/\(\) ]+", query)).strip()
        if not expression:
            result = "未找到可计算表达式，建议补充明确数字。"
            confidence = 0.2
        else:
            value = eval(expression, {"__builtins__": {}}, {})
            result = f"计算结果: {expression} = {value}"
            confidence = 0.95
        return ToolResult(
            ok=True,
            tool_name=self.name,
            result=result,
            confidence=confidence,
            source="tool/calculator",
        )
