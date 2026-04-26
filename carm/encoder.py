from __future__ import annotations

from carm.memory import MemoryBoard


class SimpleEncoder:
    """Compresses task input into a lightweight observation dict.

    Upgraded to expose structured intermediate representation signals
    alongside raw input, giving the core richer context for slot decisions.
    """

    def encode(self, user_input: str, memory: MemoryBoard) -> dict[str, str]:
        goal = memory.latest("GOAL")
        plan = memory.latest("PLAN")
        hyp = memory.latest("HYP")
        result = memory.latest("RESULT")
        draft = memory.latest("DRAFT")
        conflict = memory.latest("CONFLICT")

        plan_payload = memory.parse_content(plan)
        hyp_payload = memory.parse_content(hyp)
        draft_payload = memory.parse_content(draft)

        def _safe_list(value: object) -> str:
            if not isinstance(value, list):
                return ""
            return ", ".join(str(item) for item in value[:3])

        return {
            "input": user_input.strip(),
            "goal": goal.content if goal else "",
            "memory_summary": memory.summary(),
            "plan_summary": str(plan_payload.get("summary", "")),
            "plan_unknowns": _safe_list(plan_payload.get("unknowns")),
            "plan_evidence_targets": _safe_list(plan_payload.get("evidence_targets")),
            "hyp_summary": str(hyp_payload.get("summary", "")),
            "hyp_assumptions": _safe_list(hyp_payload.get("assumptions")),
            "draft_summary": str(draft_payload.get("summary", "")),
            "draft_confidence": str(draft_payload.get("confidence_band", "")),
            "draft_open_risks": _safe_list(draft_payload.get("open_risks")),
            "draft_support_items": _safe_list(draft_payload.get("support_items")),
            "has_external_result": "yes" if result is not None else "no",
            "has_conflict": "yes" if conflict is not None else "no",
            "has_plan_ready": "yes" if plan is not None else "no",
            "has_draft_ready": "yes" if draft is not None else "no",
        }
