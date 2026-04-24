import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from carm.evolution import EvolutionSignal
from carm.runner import AgentRunner
from tools.base import ToolManager
from tools.bigmodel_tool import BigModelProxyTool
from tools.calc_tool import CalculatorTool
from tools.code_tool import CodeExecutorTool
from tools.search_tool import SearchTool


def build_runner(temp_dir: str, controls: dict[str, object] | None = None) -> AgentRunner:
    training_config_path = Path(temp_dir) / "training.json"
    controls_path = Path(temp_dir) / "runtime_controls.json"
    if controls is not None:
        controls_path.write_text(json.dumps(controls, ensure_ascii=False), encoding="utf-8")
    training_config_path.write_text(
        (
            '{'
            '"training":{'
            '"mode":"two_stage",'
            '"online_evolution":{'
            f'"signal_state_path":"{(Path(temp_dir) / "evolution_state.json").as_posix()}",'
            f'"signal_log_path":"{(Path(temp_dir) / "signals.jsonl").as_posix()}"'
            "}"
            "}"
            "}"
        ),
        encoding="utf-8",
    )
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
        controls_path=controls_path,
        training_config_path=training_config_path,
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

    def test_structured_signal_biases_future_tool_choice(self) -> None:
        with TemporaryDirectory() as temp_dir:
            runner = build_runner(temp_dir)
            runner.apply_user_signal(
                EvolutionSignal(
                    source="test",
                    query="数据库选型",
                    preferred_tool="search",
                    reward=1.0,
                )
            )

            _, trace = runner.run("数据库选型建议")
            self.assertIn("CALL_TOOL", trace.actions)
            self.assertTrue(any(step.selected_tool == "search" for step in trace.steps))

    def test_budget_prompt_prefers_calculator_even_with_natural_language(self) -> None:
        with TemporaryDirectory() as temp_dir:
            runner = build_runner(temp_dir)
            _, trace = runner.run("我们团队 9 个人，某 SaaS 每席位 129 元/月，如果按年预算估算请按 129 * 9 * 12 计算。")

            self.assertTrue(any(step.selected_tool == "calculator" for step in trace.steps))

    def test_formal_summary_prompt_can_use_bigmodel_proxy(self) -> None:
        with TemporaryDirectory() as temp_dir:
            runner = build_runner(temp_dir)
            _, trace = runner.run("我已经收集了官方文档、社区最佳实践和故障复盘，现在要写给负责人一份简洁但正式的结论，应该怎么组织？")

            self.assertTrue(any(step.selected_tool == "bigmodel_proxy" for step in trace.steps))

    def test_conflict_prompt_prefers_search_and_carries_risk(self) -> None:
        with TemporaryDirectory() as temp_dir:
            runner = build_runner(temp_dir)
            answer, trace = runner.run("两个来源对同一个数据库参数给出了相反建议，在冲突还没消解前应该直接下结论吗？")

            self.assertTrue(any(step.selected_tool == "search" for step in trace.steps))
            self.assertTrue(any(step.target_slot == "HYP" for step in trace.steps))
            self.assertIn("暂不直接下单边结论", answer)

    def test_candidate_conflict_verify_gate_verifies_before_answer(self) -> None:
        with TemporaryDirectory() as temp_dir:
            runner = build_runner(
                temp_dir,
                controls={
                    "policy": {
                        "require_conflict_verify_before_answer": 1,
                    }
                },
            )
            _, trace = runner.run("两篇数据库迁移文章对是否要先加只读窗口给出了相反建议，这时候应该怎么处理？")

            self.assertIn("VERIFY", trace.actions)
            self.assertLess(trace.actions.index("VERIFY"), trace.actions.index("ANSWER"))

    def test_candidate_mixed_numeric_code_gate_prefers_calculator(self) -> None:
        with TemporaryDirectory() as temp_dir:
            runner = build_runner(
                temp_dir,
                controls={
                    "policy": {
                        "prefer_calculator_for_mixed_numeric_code": 1,
                    }
                },
            )
            _, trace = runner.run("这个问题里既有一段 Python 代码，又问到一次性迁移 48000 条数据预计分几批处理，每批 6000 条。你会先走哪类工具？")

            self.assertTrue(any(step.selected_tool == "calculator" for step in trace.steps))


if __name__ == "__main__":
    unittest.main()
