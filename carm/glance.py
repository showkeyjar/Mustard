from __future__ import annotations

from dataclasses import dataclass, field

from carm.memory import MemoryBoard
from carm.runtime_controls import DEFAULT_CONTROLS
from carm.state import AgentState


@dataclass
class GlanceSignal:
    active: bool = False
    trigger: str = ""
    focus: list[str] = field(default_factory=list)
    suggestion: str = ""


class InternalGlance:
    """Bounded internal lookback over structured state only."""

    def __init__(self, controls: dict[str, float | int] | None = None) -> None:
        base = dict(DEFAULT_CONTROLS["glance"])
        if controls:
            base.update(controls)
        self.controls = base

    def inspect(self, state: AgentState, memory: MemoryBoard) -> GlanceSignal:
        if state.glance_budget <= 0:
            return GlanceSignal()
        if state.hidden.get("glance_cooldown") == "1":
            return GlanceSignal()

        has_result = memory.latest("RESULT") is not None
        has_draft = memory.latest("DRAFT") is not None
        has_conflict = memory.latest("CONFLICT") is not None
        has_plan = memory.latest("PLAN") is not None

        if has_conflict:
            return GlanceSignal(
                active=True,
                trigger="conflict",
                focus=self._focus(memory, ["CONFLICT", "DRAFT"]),
                suggestion="mark_conflict",
            )

        if has_result and not has_draft:
            return GlanceSignal(
                active=True,
                trigger="result_without_draft",
                focus=self._focus(memory, ["RESULT", "PLAN"]),
                suggestion="promote_draft",
            )

        threshold = float(self.controls.get("high_uncertainty_threshold", 0.78))
        if state.uncertainty >= threshold and not has_result:
            return GlanceSignal(
                active=True,
                trigger="high_uncertainty",
                focus=self._focus(memory, ["PLAN", "GOAL"]),
                suggestion="prefer_tool",
            )

        if state.answer_ready >= 0.75 and not has_draft and has_plan:
            return GlanceSignal(
                active=True,
                trigger="premature_answer_readiness",
                focus=self._focus(memory, ["PLAN"]),
                suggestion="delay_answer",
            )

        return GlanceSignal()

    def _focus(self, memory: MemoryBoard, slot_types: list[str]) -> list[str]:
        focused: list[str] = []
        for slot_type in slot_types:
            slot = memory.latest(slot_type)
            if slot is not None:
                focused.append(f"{slot_type}:{memory.slot_brief(slot)}")
        return focused[:2]
