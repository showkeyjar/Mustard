from __future__ import annotations

from carm.memory import MemoryBoard
from carm.state import AgentState


class SimpleDecoder:
    """Renders agent state and memory into user-facing output.

    Upgraded to surface structured confidence, verification, and risk signals
    from the intermediate representation.
    """

    def render(self, user_input: str, state: AgentState, memory: MemoryBoard) -> str:
        goal = memory.latest("GOAL")
        plan = memory.latest("PLAN")
        result = memory.latest("RESULT")
        draft = memory.latest("DRAFT")
        conflict = memory.latest("CONFLICT")
        plan_payload = memory.parse_content(plan)
        draft_payload = memory.parse_content(draft)
        result_payload = memory.parse_content(result)

        parts: list[str] = []
        parts.append(f"任务: {goal.content if goal else user_input}")
        if plan:
            parts.append(f"计划: {self._render_plan(plan_payload)}")
        if result:
            parts.append(f"外部结果: {result_payload.get('raw', result.content)}")
        if draft:
            parts.append(f"结论: {self._render_draft(draft_payload)}")
        else:
            parts.append("结论: 当前没有稳定草稿。")
        if conflict:
            parts.append(f"风险: {conflict.content}")
        parts.append(self._render_status_line(state, memory, draft_payload))
        return "\n".join(parts)

    def _render_plan(self, payload: dict[str, object]) -> str:
        if not payload:
            return ""
        summary = payload.get("summary", "")
        action_items = payload.get("action_items", [])
        unknowns = payload.get("unknowns", [])
        evidence_targets = payload.get("evidence_targets", [])
        confidence = payload.get("confidence_band", "")
        parts: list[str] = []
        if summary:
            parts.append(f"摘要={summary}")
        if action_items:
            parts.append(f"动作={', '.join(str(step) for step in action_items)}")
        if unknowns:
            parts.append(f"未知={', '.join(str(item) for item in unknowns)}")
        if evidence_targets:
            parts.append(f"证据={', '.join(str(item) for item in evidence_targets)}")
        if confidence:
            parts.append(f"置信={confidence}")
        if not parts and "raw" in payload:
            parts.append(str(payload["raw"]))
        return "；".join(parts)

    def _render_draft(self, payload: dict[str, object]) -> str:
        if not payload:
            return ""
        summary = payload.get("summary", payload.get("claim", ""))
        support = payload.get("support_items", payload.get("support", []))
        risks = payload.get("open_risks", [])
        confidence = payload.get("confidence_band", payload.get("status", ""))
        parts: list[str] = []
        if summary:
            parts.append(str(summary))
        if support:
            parts.append(f"依据={', '.join(str(item) for item in support)}")
        if risks:
            parts.append(f"风险={', '.join(str(item) for item in risks)}")
        if confidence:
            parts.append(f"置信={confidence}")
        if not parts and "raw" in payload:
            parts.append(str(payload["raw"]))
        return "；".join(parts)

    def _render_status_line(
        self,
        state: AgentState,
        memory: MemoryBoard,
        draft_payload: dict[str, object],
    ) -> str:
        verified = state.hidden.get("verified") == "1"
        has_result = memory.latest("RESULT") is not None
        has_conflict = memory.latest("CONFLICT") is not None
        confidence_band = str(draft_payload.get("confidence_band", "未知"))

        status_parts = [f"不确定度={state.uncertainty:.2f}"]
        if has_result:
            status_parts.append("有外部证据")
        if verified:
            status_parts.append("已通过验证")
        if has_conflict:
            status_parts.append("存在冲突")
        if confidence_band and confidence_band != "未知":
            status_parts.append(f"置信带={confidence_band}")
        return "状态: " + "，".join(status_parts)
