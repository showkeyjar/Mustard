import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from carm.pretrain_data import load_review_feedback
from scripts.set_learning_intake_human_review_decision import set_learning_intake_human_review_decision


class LearningIntakeHumanReviewDecisionTests(unittest.TestCase):
    def test_set_learning_intake_human_review_decision_maps_defer_to_pending(self) -> None:
        with TemporaryDirectory(dir="D:/tmp") as temp_dir:
            root = Path(temp_dir)
            (root / "data" / "learning").mkdir(parents=True, exist_ok=True)
            (root / "artifacts").mkdir(parents=True, exist_ok=True)
            (root / "backlog" / "opportunities").mkdir(parents=True, exist_ok=True)
            (root / "data" / "learning" / "candidate_pretrain_human_review_sheet.jsonl").write_text(
                json.dumps(
                    {
                        "sample_id": "attention-gap-001",
                        "user_input": "样本 A",
                        "human_review_status": "",
                        "human_review_note": "",
                    },
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )

            summary = set_learning_intake_human_review_decision("attention-gap-001", "defer", root=root)

            self.assertEqual(summary["applied_status"], "pending")
            rows = load_review_feedback(root / "data" / "learning" / "candidate_pretrain_human_review_sheet.jsonl")
            self.assertEqual(rows[0]["human_review_status"], "pending")
            self.assertTrue((root / "data" / "learning" / "candidate_pretrain_human_review_sheet.backup.jsonl").exists())


if __name__ == "__main__":
    unittest.main()
