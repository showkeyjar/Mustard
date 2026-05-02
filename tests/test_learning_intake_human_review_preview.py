import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from scripts.preview_learning_intake_human_review_sheet import preview_learning_intake_human_review_sheet


class LearningIntakeHumanReviewPreviewTests(unittest.TestCase):
    def test_preview_learning_intake_human_review_sheet_counts_ready_and_blank(self) -> None:
        with TemporaryDirectory(dir="D:/tmp") as temp_dir:
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
            self.assertEqual(payload["summary"]["actionable_decision_count"], 1)
            self.assertEqual(payload["summary"]["blank_decision_count"], 1)
            self.assertEqual(payload["summary"]["would_accept_count"], 1)
            self.assertEqual(payload["next_action"]["state"], "needs_one_more_review")
            self.assertIn("decide-human-review", payload["next_action"]["command"])
            self.assertEqual(payload["missing_samples"], ["b"])
            self.assertTrue((root / "artifacts" / "learning_intake_human_review_preview_latest.json").exists())
            self.assertTrue((root / "backlog" / "opportunities" / "learning_intake_human_review_preview.md").exists())

    def test_preview_learning_intake_human_review_sheet_uses_fallback_attention_id(self) -> None:
        with TemporaryDirectory(dir="D:/tmp") as temp_dir:
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
        with TemporaryDirectory(dir="D:/tmp") as temp_dir:
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
            self.assertEqual(payload["summary"]["actionable_decision_count"], 1)

    def test_preview_learning_intake_human_review_sheet_suggests_quick_decision_for_last_blank(self) -> None:
        with TemporaryDirectory(dir="D:/tmp") as temp_dir:
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
                        "sample_id": "attention-gap-001",
                        "source_type": "learning_intake:attention_gap",
                        "suggested_decision": "defer",
                        "suggested_review_status": "pending",
                        "human_review_status": "",
                    },
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )

            payload = preview_learning_intake_human_review_sheet(root)

            self.assertEqual(payload["next_action"]["state"], "needs_one_more_review")
            self.assertIn("decide-human-review", payload["next_action"]["command"])
            self.assertIn("--sample-id attention-gap-001", payload["next_action"]["command"])
            self.assertIn("--status defer", payload["next_action"]["command"])

    def test_preview_learning_intake_human_review_sheet_suggests_decide_top_for_multiple_blanks(self) -> None:
        with TemporaryDirectory(dir="D:/tmp") as temp_dir:
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
                                "priority_score": 111.0,
                                "suggested_decision": "approve_with_edit",
                                "suggested_review_status": "edit",
                                "human_review_status": "",
                            },
                            ensure_ascii=False,
                        ),
                        json.dumps(
                            {
                                "user_input": "样本 B",
                                "sample_id": "b",
                                "priority_score": 134.0,
                                "suggested_decision": "approve",
                                "suggested_review_status": "accept",
                                "human_review_status": "",
                            },
                            ensure_ascii=False,
                        ),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            payload = preview_learning_intake_human_review_sheet(root)

            self.assertEqual(payload["next_action"]["state"], "needs_more_review")
            self.assertIn("decide-top-human-review", payload["next_action"]["command"])

    def test_preview_learning_intake_human_review_sheet_marks_already_applied_when_sheet_matches_review_pack(self) -> None:
        with TemporaryDirectory(dir="D:/tmp") as temp_dir:
            root = Path(temp_dir)
            (root / "data" / "learning").mkdir(parents=True, exist_ok=True)
            (root / "artifacts").mkdir(parents=True, exist_ok=True)
            (root / "backlog" / "opportunities").mkdir(parents=True, exist_ok=True)

            (root / "data" / "learning" / "candidate_pretrain_review_pack.jsonl").write_text(
                json.dumps({"user_input": "样本 A", "review_status": "accept"}, ensure_ascii=False) + "\n",
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

            self.assertEqual(payload["summary"]["ready_to_apply_count"], 1)
            self.assertEqual(payload["summary"]["actionable_decision_count"], 0)
            self.assertEqual(payload["next_action"]["state"], "already_applied")
            self.assertIn("status", payload["next_action"]["command"])


if __name__ == "__main__":
    unittest.main()
