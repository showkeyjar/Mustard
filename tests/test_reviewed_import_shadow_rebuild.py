import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from carm.pretrain_data import PretrainSample, save_pretrain_samples
from scripts.build_reviewed_import_shadow_rebuild import build_reviewed_import_shadow_rebuild


class ReviewedImportShadowRebuildTests(unittest.TestCase):
    def test_build_reviewed_import_shadow_rebuild_writes_controlled_shadow_outputs(self) -> None:
        with TemporaryDirectory(dir="D:/tmp") as temp_dir:
            root = Path(temp_dir)
            (root / "configs").mkdir(parents=True, exist_ok=True)
            (root / "data" / "pretrain").mkdir(parents=True, exist_ok=True)
            (root / "data" / "learning").mkdir(parents=True, exist_ok=True)
            (root / "artifacts").mkdir(parents=True, exist_ok=True)
            (root / "backlog" / "opportunities").mkdir(parents=True, exist_ok=True)

            (root / "configs" / "training.yaml").write_text(
                json.dumps(
                    {
                        "training": {
                            "pretraining": {
                                "dataset_path": "data/pretrain/pretrain_corpus.jsonl",
                                "min_quality_score": 0.72,
                                "max_dataset_samples": 100,
                            }
                        }
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            save_pretrain_samples(
                root / "data" / "pretrain" / "pretrain_corpus.jsonl",
                [
                    PretrainSample(
                        user_input="base sample",
                        task_type="planning",
                        source_type="template_planning",
                        expected_tool="search",
                        target_slot="PLAN",
                        logic_skill="step_planning",
                        quality_score=0.9,
                    )
                ],
            )
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
                                "review_note": "ok",
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
                                "user_input": "base sample",
                                "task_type": "planning",
                                "logic_skill": "step_planning",
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
                                "review_note": "keep",
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
            (root / "data" / "learning" / "candidate_pretrain_import.jsonl").write_text(
                "\n".join(
                    [
                        json.dumps(
                            {
                                "prompt": "样本 A",
                                "source_type": "human_review_patch",
                                "logic_skill": "evidence_judgment",
                                "quality_score": 0.99,
                                "metadata": {"review_status": "accept"},
                            },
                            ensure_ascii=False,
                        ),
                        json.dumps(
                            {
                                "prompt": "base sample",
                                "source_type": "human_review_patch",
                                "logic_skill": "step_planning",
                                "quality_score": 0.98,
                                "metadata": {"review_status": "edit"},
                            },
                            ensure_ascii=False,
                        ),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            summary = build_reviewed_import_shadow_rebuild(root)

            self.assertEqual(summary["approved_review_count"], 2)
            self.assertEqual(summary["import_prompt_count"], 2)
            self.assertEqual(summary["added_count"], 1)
            self.assertEqual(summary["surviving_import_count"], 1)
            self.assertEqual(summary["deduped_import_count"], 1)
            self.assertTrue((root / "data" / "learning" / "reviewed_import_shadow_corpus.jsonl").exists())
            self.assertTrue((root / "artifacts" / "reviewed_import_shadow_rebuild_latest.json").exists())


if __name__ == "__main__":
    unittest.main()
