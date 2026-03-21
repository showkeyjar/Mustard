from __future__ import annotations

import re

from carm.schemas import ToolResult


class CalculatorTool:
    name = "calculator"

    def execute(self, query: str, arguments: dict) -> ToolResult:
        candidates = re.findall(r"[0-9\.\+\-\*\/\(\) ]+", query)
        expressions = [item.strip() for item in candidates if re.search(r"\d", item) and re.search(r"[\+\-\*\/]", item)]
        expression = max(expressions, key=len, default="").strip()
        if not expression:
            result = "未找到可计算表达式，建议补充明确数字。"
            confidence = 0.2
        else:
            try:
                value = eval(expression, {"__builtins__": {}}, {})
                result = f"计算结果: {expression} = {value}"
                confidence = 0.95
            except Exception:
                result = "找到的表达式不完整，建议补充标准算式。"
                confidence = 0.25
        return ToolResult(
            ok=True,
            tool_name=self.name,
            result=result,
            confidence=confidence,
            source="tool/calculator",
        )
