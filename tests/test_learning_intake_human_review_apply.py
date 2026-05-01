import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from carm.pretrain_data import load_review_feedback
from scripts.apply_learning_intake_human_review_sheet import apply_learning_intake_human_review_sheet


class LearningIntakeHumanReviewApplyTests(unittest.TestCase):
    def test_apply_learning_intake_human_review_sheet_updates_review_pack_and_backup(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "data" / "learning").mkdir(parents=True, exist_ok=True)
            (root / "artifacts").mkdir(parents=True, exist_ok=True)
            (root / "backlog" / "opportunities").mkdir(parents=True, exist_ok=True)

            (root / "data" / "learning" / "candidate_pretrain_review_pack.jsonl").write_text(
                json.dumps(
                    {
                        "user_input": "样本 A",
                        "source_type": "learning_intake:learning_focus_stress",
                        "review_status": "pending",
                        "review_note": "",
                        "override_expected_tool": "",
                    },
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )
            (root / "data" / "learning" / "candidate_pretrain_human_review_sheet.jsonl").write_text(
                json.dumps(
                    {
                        "user_input": "样本 A",
                        "source_type": "learning_intake:learning_focus_stress",
                        "review_status": "pending",
                        "review_note": "",
                        "override_expected_tool": "search",
                        "sample_id": "a",
                        "suggested_decision": "approve",
                        "suggested_review_status": "accept",
                        "suggested_review_note": "still failing",
                        "suggested_why": ["still failing"],
                        "human_review_status": "edit",
                        "human_review_note": "tighten wording",
                    },
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )

            summary = apply_learning_intake_human_review_sheet(root)

            self.assertEqual(summary["applied_count"], 1)
            self.assertEqual(summary["edited_count"], 1)
            rows = load_review_feedback(root / "data" / "learning" / "candidate_pretrain_review_pack.jsonl")
            self.assertEqual(rows[0]["review_status"], "edit")
            self.assertEqual(rows[0]["review_note"], "tighten wording")
            self.assertEqual(rows[0]["override_expected_tool"], "search")
            self.assertNotIn("human_review_status", rows[0])
            self.assertTrue((root / "data" / "learning" / "candidate_pretrain_review_pack.backup.jsonl").exists())

    def test_apply_learning_intake_human_review_sheet_rejects_invalid_status(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "data" / "learning").mkdir(parents=True, exist_ok=True)
            (root / "artifacts").mkdir(parents=True, exist_ok=True)
            (root / "backlog" / "opportunities").mkdir(parents=True, exist_ok=True)

            (root / "data" / "learning" / "candidate_pretrain_review_pack.jsonl").write_text(
                json.dumps({"user_input": "样本 A", "review_status": "pending"}, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            (root / "data" / "learning" / "candidate_pretrain_human_review_sheet.jsonl").write_text(
                json.dumps(
                    {
                        "user_input": "样本 A",
                        "review_status": "pending",
                        "human_review_status": "maybe",
                    },
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )

            with self.assertRaises(ValueError):
                apply_learning_intake_human_review_sheet(root)


if __name__ == "__main__":
    unittest.main()
