import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from carm.core import AdaptiveReasoningCore
from carm.memory import MemoryBoard, MemorySlot
from carm.schemas import StepRecord
from carm.state import AgentState


class CoreAdaptationTests(unittest.TestCase):
    def test_learns_slot_preference_from_successful_write(self) -> None:
        with TemporaryDirectory() as temp_dir:
            core = AdaptiveReasoningCore(Path(temp_dir) / "core_state.json")
            step = StepRecord(
                step_idx=2,
                action="WRITE_MEM",
                reason="persist plan",
                score=1.0,
                feature_snapshot={"need_structure": 1.0, "need_external": 1.0},
                user_input="架构取舍建议",
                target_slot="PLAN",
                reward=1.0,
                high_value=True,
            )
            core.learn("架构取舍建议", [step], success=True)

            state = AgentState(hidden={"goal_initialized": "1"}, uncertainty=0.8)
            memory = MemoryBoard()
            memory.write(MemorySlot("GOAL", "架构取舍建议", 0.9, "test"))
            next_state = core.step({"input": "系统架构取舍建议"}, memory, state)

            self.assertEqual(next_state.hidden["slot_type"], "PLAN")
            self.assertEqual(len(next_state.latent), 6)


if __name__ == "__main__":
    unittest.main()
