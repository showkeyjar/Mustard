import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from scripts.preview_learning_intake_human_review_sheet import preview_learning_intake_human_review_sheet


class LearningIntakeHumanReviewPreviewTests(unittest.TestCase):
    def test_preview_learning_intake_human_review_sheet_counts_ready_and_blank(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "data" / "learning").mkdir(parents=True, exist_ok=True)
            (root / "artifacts").mkdir(parents=True, exist_ok=True)
            (root / "backlog" / "opportunities").mkdir(parents=True, exist_ok=True)

            (root / "data" / "learning" / "candidate_pretrain_review_pack.jsonl").write_text(
                "\n".join(
                    [
                        json.dumps({"user_input": "样本 A", "review_status": "pending"}, ensure_ascii=False),
                        json.dumps({"user_input": "样本 B", "review_status": "pending"}, ensure_ascii=False),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            (root / "data" / "learning" / "candidate_pretrain_human_review_sheet.jsonl").write_text(
                "\n".join(
                    [
                        json.dumps(
                            {
                                "user_input": "样本 A",
                                "sample_id": "a",
                                "suggested_review_status": "accept",
                                "human_review_status": "accept",
                                "human_review_note": "ok",
                            },
                            ensure_ascii=False,
                        ),
                        json.dumps(
                            {
                                "user_input": "样本 B",
                                "sample_id": "b",
                                "suggested_review_status": "edit",
                                "human_review_status": "",
                                "human_review_note": "",
                            },
                            ensure_ascii=False,
                        ),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            payload = preview_learning_intake_human_review_sheet(root)

            self.assertEqual(payload["summary"]["ready_to_apply_count"], 1)
            self.assertEqual(payload["summary"]["blank_decision_count"], 1)
            self.assertEqual(payload["summary"]["would_accept_count"], 1)
            self.assertEqual(payload["next_action"]["state"], "needs_more_review")
            self.assertEqual(payload["missing_samples"], ["b"])
            self.assertTrue((root / "artifacts" / "learning_intake_human_review_preview_latest.json").exists())
            self.assertTrue((root / "backlog" / "opportunities" / "learning_intake_human_review_preview.md").exists())

    def test_preview_learning_intake_human_review_sheet_uses_fallback_attention_id(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "data" / "learning").mkdir(parents=True, exist_ok=True)
            (root / "artifacts").mkdir(parents=True, exist_ok=True)
            (root / "backlog" / "opportunities").mkdir(parents=True, exist_ok=True)

            (root / "data" / "learning" / "candidate_pretrain_review_pack.jsonl").write_text(
                json.dumps({"user_input": "AttentionFlow task", "review_status": "pending"}, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            (root / "data" / "learning" / "candidate_pretrain_human_review_sheet.jsonl").write_text(
                json.dumps(
                    {
                        "user_input": "AttentionFlow task",
                        "source_type": "learning_intake:attention_gap",
                        "human_review_status": "",
                    },
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )

            payload = preview_learning_intake_human_review_sheet(root)

            self.assertEqual(payload["missing_samples"], ["attention-gap-001"])

    def test_preview_learning_intake_human_review_sheet_marks_ready_to_apply(self) -> None:
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
                        "sample_id": "a",
                        "suggested_review_status": "accept",
                        "human_review_status": "accept",
                        "human_review_note": "ok",
                    },
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )

            payload = preview_learning_intake_human_review_sheet(root)

            self.assertEqual(payload["next_action"]["state"], "ready_to_apply")
            self.assertIn("apply-human-review --export-import", payload["next_action"]["command"])


if __name__ == "__main__":
    unittest.main()
