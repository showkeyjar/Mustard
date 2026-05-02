import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from carm.pretrain_data import load_review_feedback
from scripts.sync_learning_intake_human_review_draft import sync_learning_intake_human_review_draft


class LearningIntakeHumanReviewDraftSyncTests(unittest.TestCase):
    def test_sync_learning_intake_human_review_draft_copies_draft_and_backs_up_sheet(self) -> None:
        with TemporaryDirectory(dir="D:/tmp") as temp_dir:
            root = Path(temp_dir)
            (root / "data" / "learning").mkdir(parents=True, exist_ok=True)
            (root / "artifacts").mkdir(parents=True, exist_ok=True)
            (root / "backlog" / "opportunities").mkdir(parents=True, exist_ok=True)

            (root / "data" / "learning" / "candidate_pretrain_human_review_sheet.jsonl").write_text(
                json.dumps({"user_input": "old", "human_review_status": ""}, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            (root / "data" / "learning" / "candidate_pretrain_human_review_sheet.draft.jsonl").write_text(
                "\n".join(
                    [
                        json.dumps({"user_input": "样本 A", "human_review_status": "accept"}, ensure_ascii=False),
                        json.dumps({"user_input": "样本 B", "human_review_status": ""}, ensure_ascii=False),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            summary = sync_learning_intake_human_review_draft(root)

            self.assertEqual(summary["synced_count"], 2)
            self.assertEqual(summary["nonempty_human_status_count"], 1)
            self.assertEqual(summary["preview_ready_to_apply_count"], 1)
            self.assertEqual(summary["preview_blank_decision_count"], 1)
            rows = load_review_feedback(root / "data" / "learning" / "candidate_pretrain_human_review_sheet.jsonl")
            self.assertEqual(rows[0]["user_input"], "样本 A")
            self.assertTrue((root / "data" / "learning" / "candidate_pretrain_human_review_sheet.backup.jsonl").exists())
            self.assertTrue((root / "artifacts" / "learning_intake_human_review_draft_sync_latest.json").exists())
            self.assertTrue((root / "artifacts" / "learning_intake_human_review_preview_latest.json").exists())


if __name__ == "__main__":
    unittest.main()
