from __future__ import annotations

from carm.memory import MemoryBoard, MemorySlot
from carm.state import AgentState


class SimpleVerifier:
    """Checks for minimal consistency before final answer.

    Upgraded from naive string-matching to structured draft payload inspection.
    Uses confidence_band, open_risks, support_items, and evidence grounding signals.
    """

    def check(self, state: AgentState, memory: MemoryBoard) -> tuple[bool, str]:
        draft = memory.latest("DRAFT")
        if draft is None:
            return False, "No draft available for verification."

        draft_payload = memory.parse_content(draft)
        confidence_band = str(draft_payload.get("confidence_band", "low"))
        open_risks = draft_payload.get("open_risks", [])
        support_items = draft_payload.get("support_items", [])
        has_result = memory.latest("RESULT") is not None
        has_conflict = memory.latest("CONFLICT") is not None
        if not isinstance(open_risks, list):
            open_risks = []
        if not isinstance(support_items, list):
            support_items = []

        if has_conflict:
            return (
                False,
                "Active conflict in working memory blocks answer release.",
            )

        if "无法验证" in draft.content:
            memory.write(
                MemorySlot(
                    slot_type="CONFLICT",
                    content="草稿包含未验证断言",
                    confidence=0.8,
                    source="verifier",
                )
            )
            return False, "Unverified assertion in draft."

        if confidence_band == "low" and not has_result:
            memory.write(
                MemorySlot(
                    slot_type="CONFLICT",
                    content="低置信草稿缺少外部证据支撑",
                    confidence=0.85,
                    source="verifier",
                )
            )
            return False, "Low-confidence draft without external backing evidence."

        if confidence_band in {"low", "medium"} and not support_items:
            return False, f"Draft confidence is '{confidence_band}' but has no support items."

        meaningful_risks = [
            risk
            for risk in open_risks
            if isinstance(risk, str)
            and risk.strip()
            and not risk.startswith("先")
        ]
        if meaningful_risks and not has_result:
            memory.write(
                MemorySlot(
                    slot_type="CONFLICT",
                    content=f"未解决的开放风险: {', '.join(meaningful_risks[:3])}",
                    confidence=0.8,
                    source="verifier",
                )
            )
            return (
                False,
                f"Draft has unresolved risks: {', '.join(meaningful_risks[:3])}.",
            )

        if has_result and confidence_band == "high" and support_items:
            return True, "Draft is well-grounded with high confidence."

        if state.uncertainty > 0.55:
            if not has_result:
                memory.write(
                    MemorySlot(
                        slot_type="CONFLICT",
                        content=f"高不确定性({state.uncertainty:.2f})且缺少外部结果",
                        confidence=0.85,
                        source="verifier",
                    )
                )
                return False, f"Uncertainty too high ({state.uncertainty:.2f}) without external evidence."

            if not support_items:
                return False, f"Uncertainty too high ({state.uncertainty:.2f}) without supporting evidence."

        if has_result and confidence_band in {"medium", "high"} and support_items:
            return True, "Draft is grounded by external result."

        if confidence_band == "medium" and support_items and state.uncertainty <= 0.45:
            return True, "Draft is supported and uncertainty is acceptable."

        return False, f"Draft needs stronger grounding: conf={confidence_band}, support={len(support_items)}, uncertainty={state.uncertainty:.2f}."
