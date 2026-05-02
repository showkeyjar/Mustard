import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from scripts.build_learning_focus_search_first_adversarial_eval import (
    build_learning_focus_search_first_adversarial_eval,
)
from scripts.evaluate_learning_focus_search_first_adversarial import (
    evaluate_learning_focus_search_first_adversarial,
)


class LearningFocusSearchFirstAdversarialTests(unittest.TestCase):
    def test_build_learning_focus_search_first_adversarial_eval_targets_failed_rows(self) -> None:
        with TemporaryDirectory(dir="D:/tmp") as temp_dir:
            root = Path(temp_dir)
            (root / "data" / "eval").mkdir(parents=True, exist_ok=True)
            (root / "artifacts").mkdir(parents=True, exist_ok=True)
            (root / "data" / "eval" / "learning_focus_eval.json").write_text(
                """
{
  "prompts": [
    {
      "id": "learning-focus-004",
      "prompt": "公开 agent 设计学习：主题=tool-use stability under ambiguity。 触发原因=frontier_zero_signal_persistence。",
      "expected_tool": "search",
      "logic_skill": "evidence_judgment"
    },
    {
      "id": "learning-focus-005",
      "prompt": "公开 agent 设计学习：主题=conflict-aware answer suppression。 触发原因=frontier_zero_signal_persistence。",
      "expected_tool": "search",
      "logic_skill": "evidence_judgment"
    }
  ]
}
""".strip(),
                encoding="utf-8",
            )
            (root / "artifacts" / "learning_focus_eval_latest.json").write_text(
                """
{
  "rows": [
    {
      "id": "learning-focus-004",
      "logic_skill": "evidence_judgment",
      "expected_tool": "search",
      "pretrained_match": false
    },
    {
      "id": "learning-focus-005",
      "logic_skill": "evidence_judgment",
      "expected_tool": "search",
      "pretrained_match": false
    }
  ]
}
""".strip(),
                encoding="utf-8",
            )

            payload = build_learning_focus_search_first_adversarial_eval(root)

            self.assertEqual(payload["summary"]["target_failure_count"], 2)
            self.assertEqual(payload["summary"]["prompt_count"], 4)
            self.assertTrue((root / "data" / "eval" / "learning_focus_search_first_adversarial_eval.json").exists())

    @patch("scripts.evaluate_learning_focus_search_first_adversarial.evaluate_isolated_prompts")
    @patch("scripts.evaluate_learning_focus_search_first_adversarial.load_eval_prompts")
    @patch("scripts.evaluate_learning_focus_search_first_adversarial.OfflinePretrainer")
    def test_evaluate_learning_focus_search_first_adversarial_writes_delta_report(
        self,
        trainer_cls,
        load_prompts_mock,
        eval_mock,
    ) -> None:
        with TemporaryDirectory(dir="D:/tmp") as temp_dir:
            root = Path(temp_dir)
            (root / "configs").mkdir(parents=True, exist_ok=True)
            (root / "data" / "eval").mkdir(parents=True, exist_ok=True)
            (root / "data" / "learning").mkdir(parents=True, exist_ok=True)
            (root / "artifacts").mkdir(parents=True, exist_ok=True)
            (root / "backlog" / "opportunities").mkdir(parents=True, exist_ok=True)
            (root / "configs" / "training.yaml").write_text(
                '{"training":{"pretraining":{"artifact_dir":"data/pretrain","experience_path":"data/experience/episodes.jsonl"}}}',
                encoding="utf-8",
            )
            (root / "data" / "eval" / "learning_focus_eval.json").write_text(
                '{"prompts":[{"id":"learning-focus-004","prompt":"公开 agent 设计学习：主题=tool-use stability under ambiguity。","expected_tool":"search","logic_skill":"evidence_judgment"}]}',
                encoding="utf-8",
            )
            (root / "artifacts" / "learning_focus_eval_latest.json").write_text(
                '{"rows":[{"id":"learning-focus-004","logic_skill":"evidence_judgment","expected_tool":"search","pretrained_match":false}]}',
                encoding="utf-8",
            )
            (root / "data" / "learning" / "reviewed_import_shadow_corpus.jsonl").write_text("", encoding="utf-8")

            load_prompts_mock.return_value = [{"id": "p1", "prompt": "x", "expected_tool": "search", "logic_skill": "evidence_judgment"}]
            eval_mock.side_effect = [
                {"summary": {"prompt_count": 2, "pretrained_match_rate": 0.0, "pretrained_avg_steps": 3.0}, "rows": []},
                {"summary": {"prompt_count": 2, "pretrained_match_rate": 0.5, "pretrained_avg_steps": 2.0}, "rows": []},
            ]

            payload = evaluate_learning_focus_search_first_adversarial(root)

            self.assertEqual(payload["target_failure_count"], 1)
            self.assertEqual(payload["delta"]["pretrained_match_rate"], 0.5)
            self.assertTrue((root / "artifacts" / "learning_focus_search_first_adversarial_latest.json").exists())
            self.assertTrue((root / "backlog" / "opportunities" / "learning_focus_search_first_adversarial.md").exists())


if __name__ == "__main__":
    unittest.main()
