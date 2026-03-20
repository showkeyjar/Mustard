from __future__ import annotations

import json
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
        if slot_type == "DRAFT":
            content = self._materialize_draft(content)
        self.write(
            MemorySlot(
                slot_type=slot_type,
                content=content,
                confidence=max(0.1, 1.0 - state.uncertainty),
                source=source,
            )
        )

    def parse_content(self, slot: Optional[MemorySlot]) -> dict[str, object]:
        if slot is None:
            return {}
        try:
            parsed = json.loads(slot.content)
        except json.JSONDecodeError:
            return {"raw": slot.content}
        return parsed if isinstance(parsed, dict) else {"raw": slot.content}

    def slot_brief(self, slot: Optional[MemorySlot]) -> str:
        payload = self.parse_content(slot)
        if not payload:
            return ""
        if "summary" in payload:
            return str(payload["summary"])
        if "action_items" in payload and isinstance(payload["action_items"], list):
            return ", ".join(str(item) for item in payload["action_items"][:2])
        if "assumptions" in payload and isinstance(payload["assumptions"], list):
            return ", ".join(str(item) for item in payload["assumptions"][:2])
        if "raw" in payload:
            return str(payload["raw"])
        return json.dumps(payload, ensure_ascii=False)

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
        parts = [f"{slot.slot_type}:{self.slot_brief(slot)}" for slot in self.slots]
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

    def _materialize_draft(self, content: str) -> str:
        payload = self._try_parse_json(content)
        if payload.get("kind") == "draft":
            return content

        result = self.latest("RESULT")
        result_text = self.slot_brief(result)
        if result_text:
            draft_payload = {
                "kind": "draft",
                "summary": "基于外部结果形成初步结论",
                "support_items": [result_text],
                "open_risks": [],
                "confidence_band": "high",
            }
            return json.dumps(draft_payload, ensure_ascii=False)

        if payload:
            draft_payload = {
                "kind": "draft",
                "summary": str(payload.get("summary") or payload.get("question") or payload.get("raw") or "形成暂定结论"),
                "support_items": [str(item) for item in payload.get("evidence_targets", [])] if isinstance(payload.get("evidence_targets"), list) else [],
                "open_risks": [str(item) for item in payload.get("assumptions", [])] if isinstance(payload.get("assumptions"), list) else [],
                "confidence_band": "medium" if payload.get("kind") == "hypothesis" else "low",
            }
            return json.dumps(draft_payload, ensure_ascii=False)

        draft_payload = {
            "kind": "draft",
            "summary": content,
            "support_items": [],
            "open_risks": [],
            "confidence_band": "low",
        }
        return json.dumps(draft_payload, ensure_ascii=False)

    def _try_parse_json(self, content: str) -> dict[str, object]:
        try:
            parsed = json.loads(content)
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}
