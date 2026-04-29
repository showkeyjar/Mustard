import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from scripts.build_learning_intake_review_queue import build_learning_intake_review_queue


class LearningIntakeReviewQueueTests(unittest.TestCase):
    def test_build_learning_intake_review_queue_prioritizes_active_failures(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "data" / "learning").mkdir(parents=True, exist_ok=True)
            (root / "data" / "evolution").mkdir(parents=True, exist_ok=True)
            (root / "artifacts").mkdir(parents=True, exist_ok=True)
            (root / "backlog" / "opportunities").mkdir(parents=True, exist_ok=True)

            (root / "data" / "learning" / "candidate_pretrain_review_pack.jsonl").write_text(
                "\n".join(
                    [
                        json.dumps(
                            {
                                "user_input": "Learning-focus routing stress 学习：样本=stress-learning-focus-evidence-routing-002。 期望工具=search。",
                                "task_type": "fact_check",
                                "logic_skill": "evidence_judgment",
                                "source_type": "learning_intake:learning_focus_stress",
                                "expected_tool": "search",
                                "target_slot": "PLAN",
                                "quality_score": 0.99,
                                "review_status": "pending",
                            },
                            ensure_ascii=False,
                        ),
                        json.dumps(
                            {
                                "user_input": "Learning-focus routing stress 学习：样本=stress-learning-focus-evidence-routing-003。 期望工具=search。",
                                "task_type": "fact_check",
                                "logic_skill": "evidence_judgment",
                                "source_type": "learning_intake:learning_focus_stress",
                                "expected_tool": "search",
                                "target_slot": "PLAN",
                                "quality_score": 0.99,
                                "review_status": "pending",
                            },
                            ensure_ascii=False,
                        ),
                        json.dumps(
                            {
                                "user_input": "AttentionFlow 学习任务：残差必须先流向 VERIFY。",
                                "task_type": "planning",
                                "logic_skill": "conflict_detection",
                                "source_type": "learning_intake:attention_gap",
                                "expected_tool": "search",
                                "target_slot": "PLAN",
                                "quality_score": 0.99,
                                "review_status": "pending",
                            },
                            ensure_ascii=False,
                        ),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            (root / "data" / "evolution" / "learning_focus_evidence_routing_eval_result.json").write_text(
                json.dumps(
                    {
                        "rows": [
                            {
                                "id": "stress-learning-focus-evidence-routing-002",
                                "baseline_used_tool": "calculator",
                                "pretrained_used_tool": "calculator",
                                "baseline_match": False,
                                "pretrained_match": False,
                            },
                            {
                                "id": "stress-learning-focus-evidence-routing-003",
                                "baseline_used_tool": "calculator",
                                "pretrained_used_tool": "search",
                                "baseline_match": False,
                                "pretrained_match": True,
                            },
                        ]
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            (root / "artifacts" / "attention_flow_latest.json").write_text(
                json.dumps({"summary": {"premature_release_count": 1}}, ensure_ascii=False),
                encoding="utf-8",
            )
            (root / "artifacts" / "attention_training_views_latest.json").write_text(
                json.dumps({"summary": {"conflict_to_verification_rate": 0.25}}, ensure_ascii=False),
                encoding="utf-8",
            )

            summary = build_learning_intake_review_queue(root)

            self.assertEqual(summary["queue_count"], 3)
            self.assertEqual(summary["recommend_accept"], 2)
            self.assertEqual(summary["recommend_edit"], 1)
            artifact = json.loads((root / "artifacts" / "learning_intake_review_queue_latest.json").read_text(encoding="utf-8"))
            queue = artifact["queue"]
            self.assertEqual(queue[0]["sample_id"], "stress-learning-focus-evidence-routing-002")
            self.assertEqual(queue[0]["recommended_status"], "accept")
            self.assertTrue((root / "backlog" / "opportunities" / "learning_intake_review_queue.md").exists())


if __name__ == "__main__":
    unittest.main()
