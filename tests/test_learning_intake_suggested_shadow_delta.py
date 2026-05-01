import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from carm.pretrain_data import PretrainSample, save_pretrain_samples
from scripts.compare_learning_intake_suggested_shadow import compare_learning_intake_suggested_shadow


class LearningIntakeSuggestedShadowDeltaTests(unittest.TestCase):
    def test_compare_learning_intake_suggested_shadow_reports_surviving_and_deduped_imports(self) -> None:
        with TemporaryDirectory() as temp_dir:
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
            save_pretrain_samples(
                root / "data" / "learning" / "candidate_pretrain_suggested_corpus.jsonl",
                [
                    PretrainSample(
                        user_input="base sample",
                        task_type="planning",
                        source_type="template_planning",
                        expected_tool="search",
                        target_slot="PLAN",
                        logic_skill="step_planning",
                        quality_score=0.9,
                    ),
                    PretrainSample(
                        user_input="new sample A",
                        task_type="fact_check",
                        source_type="human_review_patch",
                        expected_tool="search",
                        target_slot="PLAN",
                        logic_skill="evidence_judgment",
                        quality_score=0.95,
                        metadata={"candidate_source_type": "learning_intake:learning_focus_stress", "review_status": "accept"},
                    ),
                    PretrainSample(
                        user_input="new sample B",
                        task_type="planning",
                        source_type="human_review_patch",
                        expected_tool="search",
                        target_slot="PLAN",
                        logic_skill="conflict_detection",
                        quality_score=0.95,
                        metadata={"candidate_source_type": "learning_intake:attention_gap", "review_status": "edit"},
                    ),
                ],
            )
            (root / "data" / "learning" / "candidate_pretrain_suggested_import.jsonl").write_text(
                "\n".join(
                    [
                        json.dumps(
                            {
                                "prompt": "new sample A",
                                "source_type": "human_review_patch",
                                "logic_skill": "evidence_judgment",
                                "quality_score": 0.95,
                                "metadata": {
                                    "candidate_source_type": "learning_intake:learning_focus_stress",
                                    "review_status": "accept",
                                },
                            },
                            ensure_ascii=False,
                        ),
                        json.dumps(
                            {
                                "prompt": "new sample B",
                                "source_type": "human_review_patch",
                                "logic_skill": "conflict_detection",
                                "quality_score": 0.95,
                                "metadata": {
                                    "candidate_source_type": "learning_intake:attention_gap",
                                    "review_status": "edit",
                                },
                            },
                            ensure_ascii=False,
                        ),
                        json.dumps(
                            {
                                "prompt": "base sample",
                                "source_type": "human_review_patch",
                                "logic_skill": "step_planning",
                                "quality_score": 0.95,
                                "metadata": {
                                    "candidate_source_type": "learning_intake:learning_focus_stress",
                                    "review_status": "edit",
                                },
                            },
                            ensure_ascii=False,
                        ),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            payload = compare_learning_intake_suggested_shadow(root)

            self.assertEqual(payload["summary"]["base_sample_count"], 1)
            self.assertEqual(payload["summary"]["shadow_sample_count"], 3)
            self.assertEqual(payload["summary"]["added_count"], 2)
            self.assertEqual(payload["summary"]["suggested_import_count"], 3)
            self.assertEqual(payload["summary"]["surviving_import_count"], 2)
            self.assertEqual(payload["summary"]["deduped_import_count"], 1)
            self.assertTrue((root / "artifacts" / "learning_intake_suggested_shadow_delta_latest.json").exists())
            self.assertTrue((root / "backlog" / "opportunities" / "learning_intake_suggested_shadow_delta.md").exists())


if __name__ == "__main__":
    unittest.main()
