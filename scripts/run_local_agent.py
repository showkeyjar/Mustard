from __future__ import annotations

import sys

from carm.runner import AgentRunner
from tools.base import ToolManager
from tools.bigmodel_tool import BigModelProxyTool
from tools.calc_tool import CalculatorTool
from tools.code_tool import CodeExecutorTool
from tools.search_tool import SearchTool


def main() -> int:
    prompt = " ".join(sys.argv[1:]).strip()
    if not prompt:
        print("Usage: python -m scripts.run_local_agent <prompt>")
        return 1

    tool_manager = ToolManager(
        [
            SearchTool(),
            CalculatorTool(),
            CodeExecutorTool(),
            BigModelProxyTool(),
        ]
    )
    runner = AgentRunner(tool_manager)
    answer, trace = runner.run(prompt)

    print(answer)
    print("\nTrace:")
    for action, note in zip(trace.actions, trace.notes):
        print(f"- {action}: {note}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
