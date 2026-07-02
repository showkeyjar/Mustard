from __future__ import annotations

from carm.memory import MemoryBoard
from carm.state import AgentState


class SimpleDecoder:
    """Renders agent state and memory into user-facing output.

    Produces natural, readable responses with structured confidence,
    verification, and risk signals surfaced in a conversational format.
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

        # --- Main answer body ---
        parts: list[str] = []

        # Opening: acknowledge the task
        task_summary = goal.content if goal else user_input
        parts.append(f"关于「{task_summary}」：")

        # Evidence: external results
        if result:
            raw = result_payload.get("raw", result.content)
            parts.append(str(raw))

        # Core conclusion
        if draft:
            parts.append(self._render_draft_natural(draft_payload))
        else:
            parts.append("目前还没有形成稳定结论。")

        # Risk / conflict warning
        if conflict:
            parts.append(f"⚠ 注意：{conflict.content}")

        # Status footer (compact)
        parts.append(self._render_status_footer(state, memory, draft_payload))

        return "\n\n".join(parts)

    def _render_draft_natural(self, payload: dict[str, object]) -> str:
        """Render the draft conclusion in a natural, readable format."""
        if not payload:
            return ""
        summary = payload.get("summary", payload.get("claim", ""))
        support = payload.get("support_items", payload.get("support", []))
        risks = payload.get("open_risks", [])
        confidence = payload.get("confidence_band", payload.get("status", ""))

        parts: list[str] = []

        # Main claim
        if summary:
            parts.append(str(summary))

        # Supporting evidence
        if support and isinstance(support, list) and len(support) > 0:
            items = [str(item) for item in support if item]
            if items:
                parts.append("依据：" + "；".join(items))

        # Open risks
        if risks and isinstance(risks, list) and len(risks) > 0:
            items = [str(item) for item in risks if item]
            if items:
                parts.append("待确认风险：" + "；".join(items))

        # Confidence
        if confidence and str(confidence) not in ("未知", ""):
            label_map = {"high": "较高", "medium": "中等", "low": "较低"}
            label = label_map.get(str(confidence), str(confidence))
            parts.append(f"可信度：{label}")

        if not parts and "raw" in payload:
            parts.append(str(payload["raw"]))

        return "\n".join(parts)

    def _render_status_footer(
        self,
        state: AgentState,
        memory: MemoryBoard,
        draft_payload: dict[str, object],
    ) -> str:
        verified = state.hidden.get("verified") == "1"
        has_result = memory.latest("RESULT") is not None
        has_conflict = memory.latest("CONFLICT") is not None

        tags: list[str] = []
        if has_result:
            tags.append("有外部证据")
        if verified:
            tags.append("已验证")
        if has_conflict:
            tags.append("有冲突")

        if tags:
            return f"（{', '.join(tags)}）"
        return ""
