from __future__ import annotations

from carm.memory import MemoryBoard, MemorySlot
from carm.state import AgentState


class SimpleVerifier:
    """Checks for minimal consistency before final answer."""

    def check(self, state: AgentState, memory: MemoryBoard) -> tuple[bool, str]:
        draft = memory.latest("DRAFT")
        if draft is None:
            return False, "No draft available for verification."

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

        if memory.latest("RESULT") is None and state.uncertainty > 0.5:
            memory.write(
                MemorySlot(
                    slot_type="CONFLICT",
                    content="高不确定性且缺少外部结果",
                    confidence=0.85,
                    source="verifier",
                )
            )
            return False, "High uncertainty without external evidence."

        return True, "Draft passed basic verification."
