from __future__ import annotations

from carm.runner import AgentRunner
from tools.base import ToolManager
from tools.bigmodel_tool import BigModelProxyTool
from tools.calc_tool import CalculatorTool
from tools.code_tool import CodeExecutorTool
from tools.search_tool import SearchTool


def build_runner() -> AgentRunner:
    return AgentRunner(
        ToolManager(
            [
                SearchTool(),
                CalculatorTool(),
                CodeExecutorTool(),
                BigModelProxyTool(),
            ]
        )
    )


def main() -> int:
    runner = build_runner()
    print("CARM interactive session. Type 'exit' to quit.")
    while True:
        user_input = input("You> ").strip()
        if not user_input:
            continue
        if user_input.lower() in {"exit", "quit"}:
            break

        answer, trace = runner.run(user_input)
        print(f"Agent>\n{answer}\n")
        print("Trace:")
        for step in trace.steps:
            print(f"- step={step.step_idx} action={step.action} reward={step.reward:.2f} reason={step.reason}")
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
