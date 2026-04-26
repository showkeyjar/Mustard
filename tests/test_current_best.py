import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from scripts.current_best import build_current_best_payload, write_current_best


class CurrentBestTests(unittest.TestCase):
    def test_build_current_best_prefers_latest_real_prompt_eval(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "data" / "train_runs").mkdir(parents=True, exist_ok=True)
            (root / "data" / "eval").mkdir(parents=True, exist_ok=True)

            (root / "data" / "train_runs" / "auto_train_latest.json").write_text(
                json.dumps(
                    {
                        "run_id": "run-123",
                        "dataset": {
                            "path": "data/pretrain/pretrain_corpus.jsonl",
                            "sample_count": 42,
                            "teacher_sample_count": 9,
                            "real_prompt_candidate_count": 3,
                        },
                        "pretraining": {"artifact_dir": "data/pretrain"},
                        "evaluation": {
                            "pretrain_eval": {"pretrained": {"tool_match_rate": 0.95}},
                            "real_prompt_eval": {
                                "summary": {
                                    "prompt_count": 12,
                                    "pretrained_match_rate": 0.75,
                                    "pretrained_avg_steps": 3.0,
                                },
                                "rows": [
                                    {
                                        "id": "old-fail",
                                        "logic_skill": "comparison",
                                        "expected_tool": "search",
                                        "pretrained_used_tool": "calculator",
                                        "pretrained_match": False,
                                    }
                                ],
                            },
                        },
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            (root / "data" / "eval" / "real_prompt_eval_latest.json").write_text(
                json.dumps(
                    {
                        "summary": {
                            "prompt_count": 20,
                            "pretrained_match_rate": 1.0,
                            "pretrained_avg_steps": 3.25,
                        },
                        "rows": [
                            {
                                "id": "fresh-pass",
                                "logic_skill": "conflict_detection",
                                "expected_tool": "search",
                                "pretrained_used_tool": "search",
                                "pretrained_match": True,
                            }
                        ],
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            (root / "artifacts").mkdir(parents=True, exist_ok=True)
            (root / "artifacts" / "reasoning_pattern_codec_latest.json").write_text(
                json.dumps(
                    {
                        "summary": {
                            "hard_logic_count": 7,
                            "hard_logic_avg_fit_score": 0.81,
                            "residual_explanation_rate": 0.6,
                            "verify_when_residual_risky_rate": 0.5,
                        },
                        "hard_eval": {
                            "pass_rate": 0.75,
                            "failed_case_ids": ["hard-conflict"],
                        },
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            (root / "artifacts" / "attention_flow_latest.json").write_text(
                json.dumps(
                    {
                        "summary": {
                            "focus_continuity": 1.0,
                            "evidence_grounding": 0.8,
                            "residual_resolution": 0.5,
                            "premature_release_rate": 0.25,
                        }
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            (root / "artifacts" / "attention_training_views_latest.json").write_text(
                json.dumps(
                    {
                        "summary": {
                            "release_allowed_rate": 0.62,
                            "conflict_to_verification_rate": 0.75,
                            "tool_boundary_block_rate": 1.0,
                        }
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

            payload = build_current_best_payload(root)
            self.assertEqual(payload["best_run_id"], "run-123")
            self.assertEqual(payload["summary"]["real_prompt_count"], 20)
            self.assertEqual(payload["summary"]["real_prompt_match_rate"], 1.0)
            self.assertEqual(payload["summary"]["critical_failure_count"], 0)
            self.assertEqual(payload["summary"]["hard_logic_count"], 7)
            self.assertEqual(payload["summary"]["residual_explanation_rate"], 0.6)
            self.assertEqual(payload["summary"]["hard_eval_pass_rate"], 0.75)
            self.assertEqual(payload["summary"]["attention_focus_continuity"], 1.0)
            self.assertEqual(payload["summary"]["attention_premature_release_rate"], 0.25)
            self.assertEqual(payload["summary"]["attention_release_allowed_rate"], 0.62)
            self.assertEqual(payload["summary"]["attention_conflict_to_verification_rate"], 0.75)
            self.assertEqual(payload["summary"]["attention_tool_boundary_block_rate"], 1.0)
            self.assertTrue(payload["sources"]["attention_flow"].endswith("attention_flow_latest.json"))
            self.assertTrue(payload["sources"]["attention_training_views"].endswith("attention_training_views_latest.json"))
            self.assertEqual(payload["hard_eval_failures"], ["hard-conflict"])
            self.assertEqual(payload["status"], "needs_attention")

    def test_write_current_best_creates_file(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "data" / "train_runs").mkdir(parents=True, exist_ok=True)
            (root / "data" / "train_runs" / "auto_train_latest.json").write_text(
                json.dumps(
                    {
                        "run_id": "run-001",
                        "dataset": {"path": "data/pretrain/pretrain_corpus.jsonl"},
                        "pretraining": {"artifact_dir": "data/pretrain"},
                        "evaluation": {},
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

            payload = write_current_best(root)
            target = root / "artifacts" / "current_best.json"
            self.assertTrue(target.exists())
            self.assertEqual(payload["best_run_id"], "run-001")


if __name__ == "__main__":
    unittest.main()
