import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from carm.pretrain_data import load_review_feedback
from scripts.build_learning_intake_human_review_draft import build_learning_intake_human_review_draft


class LearningIntakeHumanReviewDraftTests(unittest.TestCase):
    def test_build_learning_intake_human_review_draft_prefills_suggested_statuses(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "data" / "learning").mkdir(parents=True, exist_ok=True)
            (root / "artifacts").mkdir(parents=True, exist_ok=True)
            (root / "backlog" / "opportunities").mkdir(parents=True, exist_ok=True)

            (root / "data" / "learning" / "candidate_pretrain_human_review_sheet.jsonl").write_text(
                "\n".join(
                    [
                        json.dumps(
                            {
                                "user_input": "样本 A",
                                "sample_id": "a",
                                "suggested_decision": "approve",
                                "suggested_review_status": "accept",
                                "human_review_status": "",
                                "human_review_note": "",
                            },
                            ensure_ascii=False,
                        ),
                        json.dumps(
                            {
                                "user_input": "样本 B",
                                "sample_id": "b",
                                "suggested_decision": "approve_with_edit",
                                "suggested_review_status": "edit",
                                "human_review_status": "",
                                "human_review_note": "",
                            },
                            ensure_ascii=False,
                        ),
                        json.dumps(
                            {
                                "user_input": "样本 C",
                                "sample_id": "c",
                                "suggested_decision": "defer",
                                "suggested_review_status": "pending",
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

            payload = build_learning_intake_human_review_draft(root)

            self.assertEqual(payload["summary"]["prefilled_accept_count"], 1)
            self.assertEqual(payload["summary"]["prefilled_edit_count"], 1)
            self.assertEqual(payload["summary"]["prefilled_pending_count"], 1)
            rows = load_review_feedback(root / "data" / "learning" / "candidate_pretrain_human_review_sheet.draft.jsonl")
            self.assertEqual(rows[0]["human_review_status"], "accept")
            self.assertEqual(rows[1]["human_review_status"], "edit")
            self.assertEqual(rows[2]["human_review_status"], "")
            self.assertTrue((root / "artifacts" / "learning_intake_human_review_draft_latest.json").exists())
            self.assertTrue((root / "backlog" / "opportunities" / "learning_intake_human_review_draft.md").exists())


if __name__ == "__main__":
    unittest.main()
