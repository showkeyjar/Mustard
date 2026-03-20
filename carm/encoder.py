from __future__ import annotations

from carm.memory import MemoryBoard


class SimpleEncoder:
    """Compresses task input into a lightweight observation dict."""

    def encode(self, user_input: str, memory: MemoryBoard) -> dict[str, str]:
        goal = memory.latest("GOAL")
        return {
            "input": user_input.strip(),
            "goal": goal.content if goal else "",
            "memory_summary": memory.summary(),
        }
