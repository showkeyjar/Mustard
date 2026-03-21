import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from scripts.auto_train import run_auto_train


class AutoTrainTests(unittest.TestCase):
    def test_run_auto_train_writes_report_and_artifacts(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "configs").mkdir(parents=True, exist_ok=True)
            (root / "data" / "experience").mkdir(parents=True, exist_ok=True)
            (root / "data" / "review").mkdir(parents=True, exist_ok=True)
            (root / "data" / "evolution").mkdir(parents=True, exist_ok=True)

            (root / "configs" / "pretrain_eval.json").write_text(
                json.dumps({"prompts": [{"id": "x", "prompt": "请计算 1 + 2", "expected_tool": "calculator", "logic_skill": "tool_selection"}]}, ensure_ascii=False),
                encoding="utf-8",
            )
            (root / "configs" / "real_prompt_eval.json").write_text(
                json.dumps({"prompts": [{"id": "y", "prompt": "比较 A 和 B", "expected_tool": "search", "logic_skill": "comparison"}]}, ensure_ascii=False),
                encoding="utf-8",
            )
            (root / "configs" / "training.yaml").write_text(
                json.dumps(
                    {
                        "training": {
                            "pretraining": {
                                "artifact_dir": str(root / "data" / "pretrain"),
                                "dataset_path": str(root / "data" / "pretrain" / "pretrain_corpus.jsonl"),
                                "review_pack_path": str(root / "data" / "pretrain" / "review_pack.jsonl"),
                                "count_per_task_type": 1,
                                "seed": 1,
                                "max_dataset_samples": 50,
                                "experience_path": str(root / "data" / "experience" / "episodes.jsonl"),
                                "real_prompt_candidate_path": str(root / "data" / "eval" / "real_prompt_candidates.json"),
                            },
                            "online_evolution": {
                                "signal_log_path": str(root / "data" / "evolution" / "signals.jsonl")
                            },
                            "automation": {
                                "report_dir": str(root / "data" / "train_runs"),
                                "pretrain_eval_path": str(root / "configs" / "pretrain_eval.json"),
                                "real_eval_path": str(root / "configs" / "real_prompt_eval.json"),
                            },
                        }
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

            current = Path.cwd()
            try:
                import os
                os.chdir(root)
                report = run_auto_train(root / "configs" / "training.yaml")
            finally:
                os.chdir(current)

            latest = root / "data" / "train_runs" / "auto_train_latest.json"
            self.assertTrue(latest.exists())
            self.assertEqual(report["dataset"]["sample_count"], 8)
            self.assertGreaterEqual(report["dataset"]["teacher_sample_count"], 1)
            self.assertIn("pretrain_eval", report["evaluation"])
            self.assertIn("real_prompt_eval", report["evaluation"])


if __name__ == "__main__":
    unittest.main()
