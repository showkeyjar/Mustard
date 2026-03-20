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
    )


class RunnerTests(unittest.TestCase):
    def test_comparison_prompt_uses_search_and_answers(self) -> None:
        with TemporaryDirectory() as temp_dir:
            runner = build_runner(temp_dir)
            answer, trace = runner.run("比较 PostgreSQL 和 MySQL 在中小团队里的适用性")

            self.assertIn("外部结果", answer)
            self.assertIn("计划", answer)
            self.assertIn("CALL_TOOL", trace.actions)
            self.assertEqual(trace.actions[-1], "ANSWER")

    def test_numeric_prompt_uses_calculator(self) -> None:
        with TemporaryDirectory() as temp_dir:
            runner = build_runner(temp_dir)
            answer, trace = runner.run("请计算 12 + 30 / 3")

            self.assertIn("计算结果", answer)
            self.assertIn("CALL_TOOL", trace.actions)

    def test_run_persists_experience_and_policy_state(self) -> None:
        with TemporaryDirectory() as temp_dir:
            runner = build_runner(temp_dir)
            runner.run("比较 PostgreSQL 和 MySQL 在中小团队里的适用性")

            self.assertTrue((Path(temp_dir) / "episodes.jsonl").exists())
            self.assertTrue((Path(temp_dir) / "policy_state.json").exists())


if __name__ == "__main__":
    unittest.main()
