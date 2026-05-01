import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from carm.pretrain_data import load_review_feedback
from scripts.build_learning_intake_human_review_panel import build_learning_intake_human_review_panel


class LearningIntakeHumanReviewPanelTests(unittest.TestCase):
    def test_build_learning_intake_human_review_panel_creates_sheet_and_report(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "data" / "learning").mkdir(parents=True, exist_ok=True)
            (root / "artifacts").mkdir(parents=True, exist_ok=True)
            (root / "backlog" / "opportunities").mkdir(parents=True, exist_ok=True)

            (root / "data" / "learning" / "candidate_pretrain_review_pack.jsonl").write_text(
                "\n".join(
                    [
                        json.dumps(
                            {
                                "user_input": "样本 A",
                                "source_type": "learning_intake:learning_focus_stress",
                                "review_status": "pending",
                                "override_expected_tool": "",
                            },
                            ensure_ascii=False,
                        ),
                        json.dumps(
                            {
                                "user_input": "样本 B",
                                "source_type": "learning_intake:attention_gap",
                                "review_status": "pending",
                                "override_expected_tool": "",
                            },
                            ensure_ascii=False,
                        ),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            (root / "artifacts" / "learning_intake_human_gate_packet_latest.json").write_text(
                json.dumps(
                    {
                        "decisions": [
                            {
                                "prompt": "样本 A",
                                "sample_id": "a",
                                "decision": "approve",
                                "proposed_review_status": "accept",
                                "why": ["still failing"],
                            },
                            {
                                "prompt": "样本 B",
                                "sample_id": "",
                                "decision": "defer",
                                "proposed_review_status": "pending",
                                "why": ["deduped"],
                            },
                        ]
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

            panel = build_learning_intake_human_review_panel(root)

            self.assertEqual(panel["summary"]["total_candidates"], 2)
            self.assertEqual(panel["summary"]["recommend_accept"], 1)
            self.assertEqual(panel["summary"]["recommend_defer"], 1)
            sheet_rows = load_review_feedback(root / "data" / "learning" / "candidate_pretrain_human_review_sheet.jsonl")
            self.assertEqual(sheet_rows[0]["suggested_review_status"], "accept")
            self.assertEqual(sheet_rows[0]["human_review_status"], "")
            self.assertEqual(sheet_rows[1]["sample_id"], "attention-gap-001")
            self.assertTrue((root / "artifacts" / "learning_intake_human_review_panel_latest.json").exists())
            self.assertTrue((root / "backlog" / "opportunities" / "learning_intake_human_review_panel.md").exists())


if __name__ == "__main__":
    unittest.main()
