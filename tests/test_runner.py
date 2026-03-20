import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from carm.runner import AgentRunner
from tools.base import ToolManager
from tools.bigmodel_tool import BigModelProxyTool
from tools.calc_tool import CalculatorTool
from tools.code_tool import CodeExecutorTool
from tools.search_tool import SearchTool


def build_runner(temp_dir: str) -> AgentRunner:
    return AgentRunner(
        ToolManager(
            [
                SearchTool(),
                CalculatorTool(),
                CodeExecutorTool(),
                BigModelProxyTool(),
            ]
        ),
        experience_path=Path(temp_dir) / "episodes.jsonl",
        policy_state_path=Path(temp_dir) / "policy_state.json",
        concept_state_path=Path(temp_dir) / "concept_state.json",
        core_state_path=Path(temp_dir) / "core_state.json",
        review_path=Path(temp_dir) / "reviews.jsonl",
        controls_path=Path(temp_dir) / "runtime_controls.json",
    )


class RunnerTests(unittest.TestCase):
    def test_comparison_prompt_uses_search_and_answers(self) -> None:
        with TemporaryDirectory() as temp_dir:
            runner = build_runner(temp_dir)
            answer, trace = runner.run("比较 PostgreSQL 和 MySQL 在中小团队里的适用性")

            self.assertIn("外部结果", answer)
            self.assertIn("摘要=", answer)
            self.assertIn("动作=", answer)
            self.assertIn("CALL_TOOL", trace.actions)
            self.assertEqual(trace.actions[-1], "ANSWER")
            self.assertTrue(trace.steps[0].state_signature)
            self.assertTrue(trace.steps[0].memory_signature)
            self.assertNotEqual(trace.steps[-1].reward_reason, "")
            self.assertIn("glance_suggestion", trace.steps[0].state_signature)
            self.assertIn("glance_budget", trace.steps[0].state_signature)

    def test_numeric_prompt_uses_calculator(self) -> None:
        with TemporaryDirectory() as temp_dir:
            runner = build_runner(temp_dir)
            answer, trace = runner.run("请计算 12 + 30 / 3")

            self.assertIn("计算结果", answer)
            self.assertIn("依据=", answer)
            self.assertIn("置信=", answer)
            self.assertIn("CALL_TOOL", trace.actions)
            self.assertTrue(any(step.reward_reason for step in trace.steps))
            self.assertTrue(any(step.state_signature.get("glance_suggestion") for step in trace.steps))
            self.assertTrue(any(step.glance_used for step in trace.steps))

    def test_run_persists_experience_and_policy_state(self) -> None:
        with TemporaryDirectory() as temp_dir:
            runner = build_runner(temp_dir)
            runner.run("比较 PostgreSQL 和 MySQL 在中小团队里的适用性")

            self.assertTrue((Path(temp_dir) / "episodes.jsonl").exists())
            self.assertTrue((Path(temp_dir) / "policy_state.json").exists())
            self.assertTrue((Path(temp_dir) / "concept_state.json").exists())
            self.assertTrue((Path(temp_dir) / "core_state.json").exists())
            self.assertTrue((Path(temp_dir) / "reviews.jsonl").exists())


if __name__ == "__main__":
    unittest.main()
