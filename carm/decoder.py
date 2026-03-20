from __future__ import annotations

from carm.memory import MemoryBoard
from carm.state import AgentState


class SimpleDecoder:
    def render(self, user_input: str, state: AgentState, memory: MemoryBoard) -> str:
        goal = memory.latest("GOAL")
        plan = memory.latest("PLAN")
        result = memory.latest("RESULT")
        draft = memory.latest("DRAFT")
        conflict = memory.latest("CONFLICT")

        parts: list[str] = []
        parts.append(f"任务: {goal.content if goal else user_input}")
        if plan:
            parts.append(f"计划: {plan.content}")
        if result:
            parts.append(f"外部结果: {result.content}")
        if draft:
            parts.append(f"结论: {draft.content}")
        else:
            parts.append("结论: 当前没有稳定草稿。")
        if conflict:
            parts.append(f"风险: {conflict.content}")
        parts.append(f"不确定度: {state.uncertainty:.2f}")
        return "\n".join(parts)
