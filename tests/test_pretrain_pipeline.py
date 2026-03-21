import json
import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory


class PretrainPipelineTests(unittest.TestCase):
    def test_build_dataset_supports_import_and_review_pack(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            raw_path = root / "public_tasks.jsonl"
            config_dir = root / "configs"
            data_dir = root / "data" / "pretrain"
            experience_dir = root / "data" / "experience"
            eval_dir = root / "data" / "eval"
            config_dir.mkdir(parents=True, exist_ok=True)
            data_dir.mkdir(parents=True, exist_ok=True)
            experience_dir.mkdir(parents=True, exist_ok=True)
            eval_dir.mkdir(parents=True, exist_ok=True)

            raw_path.write_text(
                json.dumps({"prompt": "比较 Redis 和 Memcached 在缓存服务里的成本"}, ensure_ascii=False)
                + "\n"
                + json.dumps({"question": "如何规划一次低风险的数据库迁移"}, ensure_ascii=False)
                + "\n",
                encoding="utf-8",
            )
            (experience_dir / "episodes.jsonl").write_text(
                json.dumps(
                    {
                        "user_input": "我们团队 9 个人，某 SaaS 每席位 129 元/月，如果按年预算估算请按 129 * 9 * 12 计算。",
                        "answer": "ok",
                        "summary": "summary",
                        "success": True,
                        "value_score": 0.91,
                        "episode_features": {
                            "used_tool": "calculator",
                            "plan_summary": "先确认预算口径。",
                            "plan_action_items": ["识别变量", "执行计算", "核对结果"],
                            "plan_unknowns": ["是否含税未知"],
                            "evidence_targets": ["精确数值"],
                            "draft_summary": "输出年度预算结果。",
                            "action_sequence": ["CALL_TOOL", "WRITE_MEM", "ANSWER"]
                        },
                        "steps": [
                            {"step_idx": 1, "action": "CALL_TOOL", "reason": "use calc", "score": 1.0, "selected_tool": "calculator"},
                            {"step_idx": 2, "action": "WRITE_MEM", "reason": "write hyp", "score": 0.8, "target_slot": "HYP"},
                            {"step_idx": 3, "action": "ANSWER", "reason": "finish", "score": 0.9}
                        ]
                    },
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )
            (config_dir / "training.yaml").write_text(
                json.dumps(
                    {
                        "training": {
                            "pretraining": {
                                "dataset_path": str(data_dir / "pretrain_corpus.jsonl"),
                                "review_pack_path": str(data_dir / "review_pack.jsonl"),
                                "count_per_task_type": 1,
                                "seed": 1,
                                "min_quality_score": 0.72,
                                "max_dataset_samples": 100,
                                "experience_path": str(experience_dir / "episodes.jsonl"),
                                "real_prompt_candidate_path": str(eval_dir / "real_prompt_candidates.json"),
                            }
                        }
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

            from scripts.build_pretrain_dataset import main as build_main

            current = Path.cwd()
            try:
                os.chdir(root)
                os.environ["CARM_PRETRAIN_IMPORT_PATHS"] = str(raw_path)
                build_main()
            finally:
                os.environ.pop("CARM_PRETRAIN_IMPORT_PATHS", None)
                os.chdir(current)

            dataset_path = data_dir / "pretrain_corpus.jsonl"
            review_pack_path = data_dir / "review_pack.jsonl"
            real_prompt_candidate_path = eval_dir / "real_prompt_candidates.json"
            self.assertTrue(dataset_path.exists())
            self.assertTrue(review_pack_path.exists())
            self.assertTrue(real_prompt_candidate_path.exists())
            dataset_records = dataset_path.read_text(encoding="utf-8").splitlines()
            self.assertGreaterEqual(len(dataset_records), 2)
            self.assertIn("calculator", dataset_path.read_text(encoding="utf-8"))

    def test_apply_review_feedback_script_updates_dataset(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config_dir = root / "configs"
            data_dir = root / "data" / "pretrain"
            config_dir.mkdir(parents=True, exist_ok=True)
            data_dir.mkdir(parents=True, exist_ok=True)

            dataset_path = data_dir / "pretrain_corpus.jsonl"
            review_pack_path = data_dir / "review_pack.jsonl"
            dataset_path.write_text(
                json.dumps(
                    {
                        "user_input": "如何规划一次低风险的数据库迁移",
                        "task_type": "planning",
                        "source_type": "template_planning",
                        "expected_tool": "search",
                        "target_slot": "PLAN",
                        "plan_summary": "围绕任务 `planning` 建立求解路径。",
                        "plan_action_items": ["拆解阶段目标", "识别依赖与风险", "输出执行顺序与里程碑"],
                        "plan_unknowns": ["约束条件不完整"],
                        "evidence_targets": ["执行约束"],
                        "draft_summary": "形成可执行、可追踪的行动计划。",
                        "quality_score": 0.86,
                        "metadata": {},
                    },
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )
            review_pack_path.write_text(
                json.dumps(
                    {
                        "user_input": "如何规划一次低风险的数据库迁移",
                        "task_type": "planning",
                        "source_type": "template_planning",
                        "expected_tool": "search",
                        "target_slot": "PLAN",
                        "quality_score": 0.86,
                        "plan_action_items": ["拆解阶段目标", "识别依赖与风险", "输出执行顺序与里程碑"],
                        "plan_unknowns": ["约束条件不完整"],
                        "evidence_targets": ["执行约束"],
                        "plan_summary": "围绕任务 `planning` 建立求解路径。",
                        "draft_summary": "形成可执行、可追踪的行动计划。",
                        "review_status": "edit",
                        "review_note": "人工确认需要更强工具偏好",
                        "override_task_type": "",
                        "override_expected_tool": "code_executor",
                        "override_target_slot": "",
                        "override_plan_summary": "",
                        "override_action_items": ["先检查脚本", "再安排迁移步骤"],
                        "override_unknowns": [],
                        "override_evidence_targets": [],
                        "override_draft_summary": "",
                    },
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )
            (config_dir / "training.yaml").write_text(
                json.dumps(
                    {
                        "training": {
                            "pretraining": {
                                "dataset_path": str(dataset_path),
                                "review_pack_path": str(review_pack_path),
                                "min_quality_score": 0.72,
                                "max_dataset_samples": 100,
                            }
                        }
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

            from scripts.apply_pretrain_review_feedback import main as apply_main

            current = Path.cwd()
            try:
                os.chdir(root)
                apply_main()
            finally:
                os.chdir(current)

            merged = [json.loads(line) for line in dataset_path.read_text(encoding="utf-8").splitlines() if line.strip()]
            self.assertEqual(merged[0]["expected_tool"], "code_executor")


if __name__ == "__main__":
    unittest.main()
