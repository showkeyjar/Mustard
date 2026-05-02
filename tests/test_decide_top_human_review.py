import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from carm.pretrain_data import load_review_feedback
from scripts.decide_top_learning_intake_human_review import decide_top_learning_intake_human_review


class DecideTopHumanReviewTests(unittest.TestCase):
    def test_decide_top_human_review_applies_highest_priority_suggestion(self) -> None:
        with TemporaryDirectory(dir="D:/tmp") as temp_dir:
            root = Path(temp_dir)
            (root / "data" / "learning").mkdir(parents=True, exist_ok=True)
            (root / "artifacts").mkdir(parents=True, exist_ok=True)
            (root / "backlog" / "opportunities").mkdir(parents=True, exist_ok=True)
            (root / "data" / "learning" / "candidate_pretrain_human_review_sheet.jsonl").write_text(
                "\n".join(
                    [
                        '{"sample_id":"a","source_type":"learning_intake:learning_focus_stress","priority_score":111.0,"suggested_review_status":"edit","human_review_status":"","human_review_note":""}',
                        '{"sample_id":"b","source_type":"learning_intake:search_first_adversarial_failure","priority_score":134.0,"suggested_review_status":"accept","human_review_status":"","human_review_note":""}',
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            payload = decide_top_learning_intake_human_review(root)

            self.assertEqual(payload["selected_sample_id"], "b")
            self.assertEqual(payload["selected_status"], "accept")
            rows = load_review_feedback(root / "data" / "learning" / "candidate_pretrain_human_review_sheet.jsonl")
            row_b = next(row for row in rows if row.get("sample_id") == "b")
            self.assertEqual(row_b["human_review_status"], "accept")
            self.assertEqual(row_b["human_review_note"], "auto:top_priority")
            self.assertTrue((root / "artifacts" / "decide_top_human_review_latest.json").exists())


if __name__ == "__main__":
    unittest.main()
