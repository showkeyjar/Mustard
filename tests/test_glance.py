import unittest

from carm.glance import InternalGlance
from carm.memory import MemoryBoard, MemorySlot
from carm.state import AgentState


class GlanceTests(unittest.TestCase):
    def test_promotes_draft_when_result_exists_without_draft(self) -> None:
        glance = InternalGlance()
        state = AgentState(uncertainty=0.55)
        memory = MemoryBoard()
        memory.write(MemorySlot("RESULT", "计算结果: 22", 0.9, "test"))

        signal = glance.inspect(state, memory)

        self.assertTrue(signal.active)
        self.assertEqual(signal.suggestion, "promote_draft")

    def test_prefers_tool_under_high_uncertainty(self) -> None:
        glance = InternalGlance()
        state = AgentState(uncertainty=0.9)
        memory = MemoryBoard()
        memory.write(MemorySlot("GOAL", "比较数据库", 0.8, "test"))

        signal = glance.inspect(state, memory)

        self.assertTrue(signal.active)
        self.assertEqual(signal.suggestion, "prefer_tool")

    def test_budget_prevents_repeated_glance(self) -> None:
        glance = InternalGlance()
        state = AgentState(uncertainty=0.9, glance_budget=0)
        memory = MemoryBoard()
        memory.write(MemorySlot("GOAL", "比较数据库", 0.8, "test"))

        signal = glance.inspect(state, memory)

        self.assertFalse(signal.active)


if __name__ == "__main__":
    unittest.main()
