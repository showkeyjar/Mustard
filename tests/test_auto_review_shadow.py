import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from carm.pretrain_data import PretrainSample, save_pretrain_samples
from scripts.run_learning_intake_auto_review_shadow import run_learning_intake_auto_review_shadow


class AutoReviewShadowTests(unittest.TestCase):
    def test_run_learning_intake_auto_review_shadow_builds_shadow_outputs(self) -> None:
        with TemporaryDirectory(dir="D:/tmp") as temp_dir:
            root = Path(temp_dir)
            (root / "configs").mkdir(parents=True, exist_ok=True)
            (root / "data" / "learning").mkdir(parents=True, exist_ok=True)
            (root / "data" / "pretrain").mkdir(parents=True, exist_ok=True)
            (root / "artifacts").mkdir(parents=True, exist_ok=True)
            (root / "backlog" / "opportunities").mkdir(parents=True, exist_ok=True)

            (root / "configs" / "training.yaml").write_text(
                '{"training":{"pretraining":{"dataset_path":"data/pretrain/pretrain_corpus.jsonl","min_quality_score":0.72,"max_dataset_samples":100}}}',
                encoding="utf-8",
            )
            save_pretrain_samples(
                root / "data" / "pretrain" / "pretrain_corpus.jsonl",
                [
                    PretrainSample(
                        user_input="base",
                        task_type="planning",
                        source_type="template",
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
                        '{"user_input":"样本 A","source_type":"learning_intake:search_first_adversarial_failure","review_status":"pending","task_type":"fact_check","logic_skill":"evidence_judgment","expected_tool":"search","target_slot":"PLAN","quality_score":0.99,"plan_action_items":[],"plan_unknowns":[],"evidence_targets":[],"plan_summary":"p","draft_summary":"d"}',
                        '{"user_input":"样本 B","source_type":"learning_intake:attention_gap","review_status":"pending","task_type":"planning","logic_skill":"conflict_detection","expected_tool":"search","target_slot":"PLAN","quality_score":0.99,"plan_action_items":[],"plan_unknowns":[],"evidence_targets":[],"plan_summary":"p","draft_summary":"d"}',
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            (root / "artifacts" / "learning_intake_human_gate_packet_latest.json").write_text(
                '{"decisions":[{"prompt":"样本 A","sample_id":"a","decision":"approve","proposed_review_status":"accept","priority_score":134.0},{"prompt":"样本 B","sample_id":"b","decision":"defer","proposed_review_status":"pending","priority_score":117.0}]}',
                encoding="utf-8",
            )

            payload = run_learning_intake_auto_review_shadow(root)

            self.assertEqual(payload["auto_accept_count"], 1)
            self.assertEqual(payload["auto_pending_count"], 1)
            self.assertEqual(payload["auto_import_count"], 1)
            self.assertEqual(payload["top_priority_sample_id"], "a")
            self.assertTrue((root / "data" / "learning" / "candidate_pretrain_auto_review_shadow.jsonl").exists())
            self.assertTrue((root / "data" / "learning" / "candidate_pretrain_auto_import_shadow.jsonl").exists())
            self.assertTrue((root / "data" / "learning" / "candidate_pretrain_auto_shadow_corpus.jsonl").exists())


if __name__ == "__main__":
    unittest.main()
