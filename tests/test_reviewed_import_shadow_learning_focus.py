import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from scripts.evaluate_reviewed_import_shadow_learning_focus import evaluate_reviewed_import_shadow_learning_focus


class ReviewedImportShadowLearningFocusTests(unittest.TestCase):
    @patch("scripts.evaluate_reviewed_import_shadow_learning_focus.evaluate_isolated_prompts")
    @patch("scripts.evaluate_reviewed_import_shadow_learning_focus.load_eval_prompts")
    @patch("scripts.evaluate_reviewed_import_shadow_learning_focus.OfflinePretrainer")
    def test_evaluate_reviewed_import_shadow_learning_focus_writes_delta_report(
        self,
        trainer_cls,
        load_prompts_mock,
        eval_mock,
    ) -> None:
        with TemporaryDirectory(dir="D:/tmp") as temp_dir:
            root = Path(temp_dir)
            (root / "configs").mkdir(parents=True, exist_ok=True)
            (root / "data" / "learning").mkdir(parents=True, exist_ok=True)
            (root / "data" / "eval").mkdir(parents=True, exist_ok=True)
            (root / "artifacts").mkdir(parents=True, exist_ok=True)
            (root / "backlog" / "opportunities").mkdir(parents=True, exist_ok=True)
            (root / "configs" / "training.yaml").write_text(
                '{"training":{"pretraining":{"artifact_dir":"data/pretrain","experience_path":"data/experience/episodes.jsonl"}}}',
                encoding="utf-8",
            )
            (root / "data" / "learning" / "reviewed_import_shadow_corpus.jsonl").write_text("", encoding="utf-8")
            (root / "data" / "eval" / "learning_focus_eval.json").write_text('{"prompts":[]}', encoding="utf-8")

            load_prompts_mock.return_value = [{"id": "p1", "prompt": "x", "expected_tool": "search", "logic_skill": "evidence_judgment"}]
            eval_mock.side_effect = [
                {"summary": {"prompt_count": 1, "pretrained_match_rate": 0.5714, "pretrained_avg_steps": 3.0}, "rows": []},
                {"summary": {"prompt_count": 1, "pretrained_match_rate": 0.7143, "pretrained_avg_steps": 3.0}, "rows": []},
            ]

            payload = evaluate_reviewed_import_shadow_learning_focus(root)

            self.assertEqual(payload["delta"]["pretrained_match_rate"], 0.1429)
            self.assertTrue((root / "artifacts" / "reviewed_import_shadow_learning_focus_latest.json").exists())
            self.assertTrue((root / "backlog" / "opportunities" / "reviewed_import_shadow_learning_focus.md").exists())


if __name__ == "__main__":
    unittest.main()
