import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from carm.actions import Action
from carm.memory import MemoryBoard, MemorySlot
from carm.policy import OnlinePolicy
from carm.schemas import StepRecord
from carm.state import AgentState


class PolicyAdaptationTests(unittest.TestCase):
    def test_learns_new_semantic_tool_preference(self) -> None:
        with TemporaryDirectory() as temp_dir:
            policy = OnlinePolicy(
                Path(temp_dir) / "policy_state.json",
                Path(temp_dir) / "concept_state.json",
            )

            learned_step = StepRecord(
                step_idx=2,
                action=Action.CALL_TOOL.value,
                reason="learn search preference",
                score=1.0,
                feature_snapshot={"bias": 1.0},
                user_input="数据库选型建议",
                selected_tool="search",
                reward=1.0,
                high_value=True,
            )
            policy.learn([learned_step])

            state = AgentState(
                hidden={"goal_initialized": "1", "candidate": "按计划补充事实", "slot_type": "HYP"},
                uncertainty=0.9,
            )
            memory = MemoryBoard()
            memory.write(MemorySlot("GOAL", "数据库选型建议", 0.8, "test"))
            decision = policy.decide(state, memory, "创业团队数据库选型怎么做")

            self.assertEqual(decision.action, Action.CALL_TOOL)
            self.assertIsNotNone(decision.tool_call)
            self.assertEqual(decision.tool_call.tool_name, "search")


if __name__ == "__main__":
    unittest.main()
