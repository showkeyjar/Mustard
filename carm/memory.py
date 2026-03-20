from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Optional

from carm.state import AgentState


@dataclass
class MemorySlot:
    slot_type: str
    content: str
    confidence: float
    source: str
    ttl: int = 6


@dataclass
class MemoryBoard:
    max_slots: int = 16
    slots: list[MemorySlot] = field(default_factory=list)
    focus_slot: Optional[str] = None

    def read(self, slot_type: Optional[str] = None) -> list[MemorySlot]:
        if slot_type is None:
            return list(self.slots)
        return [slot for slot in self.slots if slot.slot_type == slot_type]

    def latest(self, slot_type: str) -> Optional[MemorySlot]:
        matches = self.read(slot_type)
        return matches[-1] if matches else None

    def write(self, slot: MemorySlot) -> None:
        if slot.slot_type == "GOAL":
            goal = self.latest("GOAL")
            if goal is not None:
                goal.content = slot.content
                goal.confidence = slot.confidence
                goal.source = slot.source
                goal.ttl = slot.ttl
                return

        self.slots.append(slot)
        self._trim()

    def focus(self, slot_type: str) -> list[MemorySlot]:
        self.focus_slot = slot_type
        return self.read(slot_type)

    def store_result(self, content: str, confidence: float, source: str) -> None:
        self.write(
            MemorySlot(
                slot_type="RESULT",
                content=content,
                confidence=confidence,
                source=source,
            )
        )

    def write_from_state(self, state: AgentState, slot_type: str, source: str) -> None:
        content = state.hidden.get("candidate", "").strip()
        if not content:
            return
        self.write(
            MemorySlot(
                slot_type=slot_type,
                content=content,
                confidence=max(0.1, 1.0 - state.uncertainty),
                source=source,
            )
        )

    def decay(self) -> None:
        remaining: list[MemorySlot] = []
        for slot in self.slots:
            slot.ttl -= 1
            if slot.ttl > 0 or slot.slot_type in {"GOAL", "FACT", "RESULT", "CONFLICT"}:
                remaining.append(slot)
        self.slots = remaining[-self.max_slots :]

    def _trim(self) -> None:
        if len(self.slots) <= self.max_slots:
            return

        sticky = [slot for slot in self.slots if slot.slot_type in {"GOAL", "FACT", "RESULT", "CONFLICT"}]
        other = [slot for slot in self.slots if slot.slot_type not in {"GOAL", "FACT", "RESULT", "CONFLICT"}]
        self.slots = (sticky + other)[-self.max_slots :]

    def summary(self) -> str:
        parts = [f"{slot.slot_type}:{slot.content}" for slot in self.slots]
        return " | ".join(parts)

    def restore(self, slots: Iterable[MemorySlot]) -> None:
        self.slots = [
            MemorySlot(
                slot_type=slot.slot_type,
                content=slot.content,
                confidence=slot.confidence,
                source=slot.source,
                ttl=slot.ttl,
            )
            for slot in slots
        ]
