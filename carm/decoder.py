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

        # Core conclusion (draft always takes priority for the answer body)
        if draft:
            parts.append(self._render_draft_natural(draft_payload, result_payload))
        elif result:
            # No draft yet — show a compact result preview
            brief = self._result_brief(result_payload, result)
            parts.append(brief)
        else:
            parts.append("目前还没有形成稳定结论。")

        # Risk / conflict warning
        if conflict:
            parts.append(f"⚠ 注意：{conflict.content}")

        # Status footer (compact)
        parts.append(self._render_status_footer(state, memory, draft_payload))

        return "\n\n".join(parts)

    def _result_brief(self, result_payload: dict, result_slot: object) -> str:
        """Generate a brief preview of tool result without full-text dump."""
        raw = str(result_payload.get("raw", "")) if result_payload else ""
        if not raw and result_slot is not None:
            raw = str(getattr(result_slot, "content", ""))
        # For calculator/code results, the raw output IS the answer — show it
        source = str(result_payload.get("source", "")) if result_payload else ""
        if source.startswith("tool/calc") or source.startswith("tool/code"):
            return raw
        # For search results, show only a compact summary line
        lines = raw.split("\n")
        count = 0
        for line in lines:
            if line.strip() and line.strip()[0].isdigit():
                count += 1
        if count > 0:
            return f"已检索到 {count} 条外部结果（详见下方依据）"
        if len(raw) > 60:
            return raw[:60] + "…"
        return raw or "已获得外部结果"

    def _render_draft_natural(
        self, payload: dict[str, object], result_payload: dict | None = None
    ) -> str:
        """Render the draft conclusion in a natural, readable format."""
        if not payload:
            return ""
        summary = str(payload.get("summary", payload.get("claim", "")))
        support = payload.get("support_items", payload.get("support", []))
        risks = payload.get("open_risks", [])
        confidence = payload.get("confidence_band", payload.get("status", ""))

        parts: list[str] = []

        # Main claim
        if summary:
            parts.append(summary)

        # Supporting evidence — smart dedup against summary
        if support and isinstance(support, list) and len(support) > 0:
            items: list[str] = []
            raw_result = ""
            if result_payload:
                raw_result = str(result_payload.get("raw", ""))
            for item in support:
                item_str = str(item)
                # Skip if this item is already contained in the summary
                if item_str and len(item_str) > 10 and item_str in summary:
                    continue
                # Skip if this item is a prefix/suffix of the raw result
                # and the summary already references the result
                if raw_result and item_str in raw_result:
                    # If summary already says "基于检索结果" or similar,
                    # just note the source count, don't dump the full text
                    if "检索结果" in summary or "外部结果" in summary:
                        lines = [l for l in raw_result.split("\n") if l.strip()]
                        count_line = f"共{len(lines)}条检索记录"
                        if count_line not in items:
                            items.append(count_line)
                        continue
                    # Otherwise expand the brief reference into full content
                    items.append(raw_result)
                    raw_result = ""  # Only expand once
                else:
                    # Truncate overly long support items for readability
                    if len(item_str) > 300:
                        items.append(item_str[:300] + "…")
                    else:
                        items.append(item_str)
            if items:
                parts.append("依据：" + "；".join(items))

        # Open risks
        if risks and isinstance(risks, list) and len(risks) > 0:
            risk_items = [str(item) for item in risks if item]
            if risk_items:
                parts.append("待确认风险：" + "；".join(risk_items))

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
