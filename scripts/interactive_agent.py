from __future__ import annotations

import json

from carm.evolution import EvolutionSignal
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
    print("Commands: /goal <text> | /prefer <tool> <query> | /evolve <json>")
    while True:
        user_input = input("You> ").strip()
        if not user_input:
            continue
        if user_input.lower() in {"exit", "quit"}:
            break
        if user_input.startswith("/goal "):
            goal = user_input[6:].strip()
            runner.apply_user_signal(
                EvolutionSignal(
                    source="interactive_cli",
                    query=goal,
                    goal=goal,
                    preferred_slot="PLAN",
                    reward=1.0,
                    note="interactive goal update",
                )
            )
            print("Agent>\n已记录当前目标，并将后续推理偏向结构化规划。\n")
            continue
        if user_input.startswith("/prefer "):
            payload = user_input[8:].strip().split(maxsplit=1)
            tool_name = payload[0] if payload else ""
            query = payload[1] if len(payload) > 1 else ""
            runner.apply_user_signal(
                EvolutionSignal(
                    source="interactive_cli",
                    query=query,
                    preferred_tool=tool_name,
                    reward=1.0,
                    note="interactive tool preference",
                )
            )
            print(f"Agent>\n已记录工具偏好: {tool_name or '未指定'}。\n")
            continue
        if user_input.startswith("/evolve "):
            payload = json.loads(user_input[8:].strip())
            runner.apply_user_signal(EvolutionSignal(**payload))
            print("Agent>\n已应用结构化在线进化信号。\n")
            continue

        answer, trace = runner.run(user_input)
        print(f"Agent>\n{answer}\n")
        print("Trace:")
        for step in trace.steps:
            print(f"- step={step.step_idx} action={step.action} reward={step.reward:.2f} reason={step.reason}")
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
