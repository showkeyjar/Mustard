import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from scripts.evaluate_pretraining import compare_results, evaluate_runner, load_eval_prompts
from scripts.evaluate_real_prompts import evaluate_isolated_prompts


class FakeTrace:
    def __init__(self, actions, steps):
        self.actions = actions
        self.steps = steps


class FakeStep:
    def __init__(self, selected_tool="", target_slot=""):
        self.selected_tool = selected_tool
        self.target_slot = target_slot


class FakeRunner:
    def __init__(self, used_tool: str, target_slot: str = "PLAN") -> None:
        self.used_tool = used_tool
        self.target_slot = target_slot

    def run(self, prompt: str):
        return "任务: " + prompt, FakeTrace(
            ["WRITE_MEM", "CALL_TOOL", "ANSWER"],
            [FakeStep(target_slot=self.target_slot), FakeStep(selected_tool=self.used_tool), FakeStep()],
        )


class EvaluatePretrainingTests(unittest.TestCase):
    def test_load_eval_prompts(self) -> None:
        with TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "eval.json"
            path.write_text(
                json.dumps({"prompts": [{"id": "x", "prompt": "请计算 1 + 2", "expected_tool": "calculator", "logic_skill": "tool_selection"}]}, ensure_ascii=False),
                encoding="utf-8",
            )
            prompts = load_eval_prompts(path)
            self.assertEqual(len(prompts), 1)
            self.assertEqual(prompts[0]["expected_tool"], "calculator")
            self.assertEqual(prompts[0]["logic_skill"], "tool_selection")

    def test_evaluate_and_compare_results(self) -> None:
        prompts = [
            {"id": "calc", "prompt": "请计算 1 + 2", "expected_tool": "calculator", "logic_skill": "tool_selection"},
            {"id": "cmp", "prompt": "比较 A 和 B", "expected_tool": "search", "logic_skill": "comparison"},
        ]
        baseline = evaluate_runner(FakeRunner("search"), prompts)
        pretrained = evaluate_runner(FakeRunner("calculator"), prompts)
        comparison = compare_results(baseline, pretrained)

        self.assertIn("baseline", comparison)
        self.assertIn("pretrained", comparison)
        self.assertEqual(len(comparison["rows"]), 2)
        self.assertIn("by_logic_skill", baseline["summary"])

    def test_repo_eval_config_contains_hard_cases(self) -> None:
        prompts = load_eval_prompts("d:/codes/Mustard/configs/pretrain_eval.json")
        ids = {str(item.get("id", "")) for item in prompts}
        self.assertIn("code-looking-search", ids)
        self.assertIn("search-looking-calc", ids)
        self.assertIn("integrate-for-exec", ids)

    def test_evaluate_isolated_prompts_uses_per_prompt_runners(self) -> None:
        prompts = [
            {"id": "calc", "prompt": "请计算 1 + 2", "expected_tool": "calculator", "logic_skill": "tool_selection"},
            {"id": "summary", "prompt": "请整理成正式结论", "expected_tool": "bigmodel_proxy", "logic_skill": "result_integration"},
        ]
        with TemporaryDirectory() as temp_dir:
            result = evaluate_isolated_prompts(prompts, artifact_dir=Path(temp_dir))

        self.assertEqual(result["summary"]["prompt_count"], 2)
        self.assertEqual(len(result["rows"]), 2)
        self.assertIn("baseline_match_rate", result["summary"])
        self.assertIn("pretrained_match_rate", result["summary"])


if __name__ == "__main__":
    unittest.main()
