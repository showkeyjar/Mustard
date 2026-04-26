import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from scripts.build_learning_focus_eval import build_learning_focus_eval
from scripts.evaluate_learning_focus import write_learning_focus_report


class LearningFocusEvalTests(unittest.TestCase):
    def test_build_learning_focus_eval_writes_prompt_pack(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            input_path = root / "learning_intake_samples.jsonl"
            output_path = root / "learning_focus_eval.json"
            input_path.write_text(
                "\n".join(
                    [
                        json.dumps(
                            {
                                "user_input": "公开 agent 设计思想内化：主题=reasoning and acting loop。",
                                "expected_tool": "search",
                                "logic_skill": "result_integration",
                                "source_type": "learning_intake:public_idea",
                            },
                            ensure_ascii=False,
                        ),
                        json.dumps(
                            {
                                "user_input": "AttentionFlow 学习任务：premature_release_count=1。",
                                "expected_tool": "search",
                                "logic_skill": "conflict_detection",
                                "source_type": "learning_intake:attention_gap",
                            },
                            ensure_ascii=False,
                        ),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            payload = build_learning_focus_eval(input_path, output_path, limit=5)

            self.assertEqual(len(payload["prompts"]), 2)
            self.assertTrue(output_path.exists())
            self.assertEqual(payload["prompts"][0]["expected_tool"], "search")

    def test_evaluate_learning_focus_writes_report(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            eval_path = root / "learning_focus_eval.json"
            output_path = root / "learning_focus_eval_latest.json"
            eval_path.write_text(
                json.dumps(
                    {
                        "prompts": [
                            {
                                "id": "learning-focus-001",
                                "prompt": "请计算 12 + 30 / 3",
                                "expected_tool": "calculator",
                                "logic_skill": "tool_selection",
                            }
                        ]
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )

            current = Path.cwd()
            try:
                import os

                os.chdir("d:/codes/Mustard")
                result = write_learning_focus_report(eval_path, output_path)
            finally:
                os.chdir(current)

            self.assertTrue(output_path.exists())
            self.assertEqual(result["summary"]["prompt_count"], 1)
            self.assertIn("rows", result)


if __name__ == "__main__":
    unittest.main()
