from __future__ import annotations

from carm.memory import MemoryBoard
from carm.state import AgentState


class HeuristicReasoningCore:
    """A placeholder reasoning core that updates lightweight cognitive state."""

    def step(self, observation: dict[str, str], memory: MemoryBoard, state: AgentState) -> AgentState:
        next_state = state.snapshot()
        next_state.step_idx += 1
        next_state.phase = "REASONING"

        user_input = observation["input"]
        lower = user_input.lower()

        needs_precision = any(token in user_input for token in ("多少", "计算", "cost", "price", "sum", "数字"))
        comparative = any(token in user_input for token in ("比较", "区别", "优缺点", "vs", "对比"))
        code_related = any(token in lower for token in ("python", "code", "script", "代码"))

        if "goal_initialized" not in next_state.hidden:
            next_state.hidden["candidate"] = user_input
            next_state.hidden["slot_type"] = "GOAL"
            next_state.hidden["goal_initialized"] = "1"
            next_state.uncertainty = 0.9
            next_state.answer_ready = 0.0
            return next_state

        if comparative and memory.latest("PLAN") is None:
            next_state.hidden["candidate"] = "比较维度: 成本, 性能, 维护复杂度, 生态, 扩展性"
            next_state.hidden["slot_type"] = "PLAN"
            next_state.uncertainty = 0.7
            return next_state

        if needs_precision and memory.latest("RESULT") is None:
            next_state.hidden["candidate"] = "需要工具获取精确结果或外部事实"
            next_state.hidden["slot_type"] = "HYP"
            next_state.uncertainty = 0.8
            return next_state

        if code_related and memory.latest("RESULT") is None:
            next_state.hidden["candidate"] = "适合调用代码执行工具验证实现"
            next_state.hidden["slot_type"] = "PLAN"
            next_state.uncertainty = 0.75
            return next_state

        result = memory.latest("RESULT")
        if result and memory.latest("DRAFT") is None:
            next_state.hidden["candidate"] = (
                f"基于工具结果形成草稿: {result.content}"
            )
            next_state.hidden["slot_type"] = "DRAFT"
            next_state.uncertainty = 0.35
            next_state.answer_ready = 0.55
            return next_state

        plan = memory.latest("PLAN")
        if plan and memory.latest("RESULT") is None:
            next_state.hidden["candidate"] = f"按计划补充事实: {plan.content}"
            next_state.hidden["slot_type"] = "HYP"
            next_state.uncertainty = 0.82
            return next_state

        draft = memory.latest("DRAFT")
        if draft:
            next_state.hidden["candidate"] = draft.content
            next_state.uncertainty = 0.2
            next_state.answer_ready = 0.9
            return next_state

        next_state.hidden["candidate"] = "形成初步回答草稿，建议附上不确定性说明"
        next_state.hidden["slot_type"] = "DRAFT"
        next_state.uncertainty = 0.4
        next_state.answer_ready = 0.75
        return next_state
