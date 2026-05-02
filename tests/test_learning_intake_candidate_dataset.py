import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from carm.pretrain_data import PretrainSample, save_pretrain_samples
from scripts.build_learning_intake_candidate_dataset import build_learning_intake_candidate_dataset


class LearningIntakeCandidateDatasetTests(unittest.TestCase):
    def test_build_learning_intake_candidate_dataset_writes_preview_outputs(self) -> None:
        with TemporaryDirectory(dir="D:/tmp") as temp_dir:
            root = Path(temp_dir)
            (root / "configs").mkdir(parents=True, exist_ok=True)
            (root / "data" / "pretrain").mkdir(parents=True, exist_ok=True)
            (root / "data" / "learning").mkdir(parents=True, exist_ok=True)
            (root / "backlog" / "opportunities").mkdir(parents=True, exist_ok=True)
            (root / "artifacts").mkdir(parents=True, exist_ok=True)

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
                        user_input="如何规划一次低风险的数据库迁移",
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
                root / "data" / "learning" / "learning_intake_samples.jsonl",
                [
                    PretrainSample(
                        user_input="Learning-focus routing stress 学习：样本=stress-002。请把这个误路由场景转成监督任务。",
                        task_type="fact_check",
                        source_type="learning_intake:learning_focus_stress",
                        expected_tool="search",
                        target_slot="PLAN",
                        logic_skill="evidence_judgment",
                        quality_score=0.95,
                    ),
                    PretrainSample(
                        user_input="AttentionFlow 学习任务：先把残差流向 VERIFY。",
                        task_type="planning",
                        source_type="learning_intake:attention_gap",
                        expected_tool="search",
                        target_slot="PLAN",
                        logic_skill="conflict_detection",
                        quality_score=0.93,
                    ),
                    PretrainSample(
                        user_input="Search-first adversarial failure 学习：样本=search-first-adversarial-002。",
                        task_type="fact_check",
                        source_type="learning_intake:search_first_adversarial_failure",
                        expected_tool="search",
                        target_slot="PLAN",
                        logic_skill="evidence_judgment",
                        quality_score=0.97,
                    ),
                    PretrainSample(
                        user_input="公开 agent 设计思想内化：把推理与行动交替展开。",
                        task_type="planning",
                        source_type="learning_intake:public_idea",
                        expected_tool="search",
                        target_slot="DRAFT",
                        logic_skill="result_integration",
                        quality_score=0.94,
                    ),
                ],
            )

            payload = build_learning_intake_candidate_dataset(root)

            self.assertEqual(payload["base_sample_count"], 1)
            self.assertEqual(payload["selected_candidate_count"], 3)
            self.assertTrue((root / "data" / "learning" / "candidate_pretrain_corpus.jsonl").exists())
            self.assertTrue((root / "data" / "learning" / "candidate_pretrain_review_pack.jsonl").exists())
            self.assertTrue((root / "artifacts" / "learning_intake_candidate_dataset_latest.json").exists())
            self.assertTrue((root / "backlog" / "opportunities" / "learning_intake_candidate_dataset_report.md").exists())
            self.assertIn("learning_intake:learning_focus_stress", payload["selected_source_counts"])
            self.assertIn("learning_intake:search_first_adversarial_failure", payload["selected_source_counts"])
            self.assertNotIn("learning_intake:public_idea", payload["selected_source_counts"])


if __name__ == "__main__":
    unittest.main()
