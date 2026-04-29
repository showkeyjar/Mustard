import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from scripts.export_learning_intake_candidate_import import export_learning_intake_candidate_import


class LearningIntakeCandidateImportTests(unittest.TestCase):
    def test_export_learning_intake_candidate_import_only_keeps_reviewed_rows(self) -> None:
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
                                "task_type": "fact_check",
                                "logic_skill": "evidence_judgment",
                                "source_type": "learning_intake:learning_focus_stress",
                                "expected_tool": "search",
                                "target_slot": "PLAN",
                                "quality_score": 0.99,
                                "plan_action_items": ["a", "b", "c"],
                                "plan_unknowns": ["u"],
                                "evidence_targets": ["e"],
                                "plan_summary": "plan",
                                "draft_summary": "draft",
                                "review_status": "accept",
                                "review_note": "looks good",
                                "override_task_type": "",
                                "override_expected_tool": "",
                                "override_target_slot": "",
                                "override_plan_summary": "",
                                "override_action_items": [],
                                "override_unknowns": [],
                                "override_evidence_targets": [],
                                "override_draft_summary": "",
                            },
                            ensure_ascii=False,
                        ),
                        json.dumps(
                            {
                                "user_input": "样本 B",
                                "task_type": "planning",
                                "logic_skill": "conflict_detection",
                                "source_type": "learning_intake:attention_gap",
                                "expected_tool": "search",
                                "target_slot": "PLAN",
                                "quality_score": 0.98,
                                "plan_action_items": ["a", "b", "c"],
                                "plan_unknowns": ["u"],
                                "evidence_targets": ["e"],
                                "plan_summary": "plan",
                                "draft_summary": "draft",
                                "review_status": "edit",
                                "review_note": "tighten target slot",
                                "override_task_type": "",
                                "override_expected_tool": "search",
                                "override_target_slot": "HYP",
                                "override_plan_summary": "",
                                "override_action_items": [],
                                "override_unknowns": [],
                                "override_evidence_targets": [],
                                "override_draft_summary": "",
                            },
                            ensure_ascii=False,
                        ),
                        json.dumps(
                            {
                                "user_input": "样本 C",
                                "task_type": "planning",
                                "logic_skill": "result_integration",
                                "source_type": "learning_intake:public_idea",
                                "expected_tool": "search",
                                "target_slot": "DRAFT",
                                "quality_score": 0.97,
                                "plan_action_items": ["a", "b", "c"],
                                "plan_unknowns": ["u"],
                                "evidence_targets": ["e"],
                                "plan_summary": "plan",
                                "draft_summary": "draft",
                                "review_status": "pending",
                                "review_note": "",
                                "override_task_type": "",
                                "override_expected_tool": "",
                                "override_target_slot": "",
                                "override_plan_summary": "",
                                "override_action_items": [],
                                "override_unknowns": [],
                                "override_evidence_targets": [],
                                "override_draft_summary": "",
                            },
                            ensure_ascii=False,
                        ),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            summary = export_learning_intake_candidate_import(root)

            self.assertEqual(summary["review_total"], 3)
            self.assertEqual(summary["approved_count"], 2)
            self.assertEqual(summary["pending_count"], 1)
            self.assertTrue((root / "data" / "learning" / "candidate_pretrain_import.jsonl").exists())
            self.assertTrue((root / "artifacts" / "learning_intake_candidate_import_latest.json").exists())
            self.assertTrue((root / "backlog" / "opportunities" / "learning_intake_candidate_import_report.md").exists())

            lines = [
                json.loads(line)
                for line in (root / "data" / "learning" / "candidate_pretrain_import.jsonl").read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            self.assertEqual(len(lines), 2)
            self.assertTrue(all(item["metadata"]["review_status"] in {"accept", "edit"} for item in lines))


if __name__ == "__main__":
    unittest.main()
