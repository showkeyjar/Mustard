import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from scripts.build_learning_intake import build_learning_intake


class LearningIntakeTests(unittest.TestCase):
    def test_build_learning_intake_collects_multi_source_candidates(self) -> None:
        with TemporaryDirectory(dir="D:/tmp") as temp_dir:
            root = Path(temp_dir)
            (root / "data" / "experience").mkdir(parents=True, exist_ok=True)
            (root / "data" / "desktop").mkdir(parents=True, exist_ok=True)
            (root / "data" / "research").mkdir(parents=True, exist_ok=True)
            (root / "artifacts").mkdir(parents=True, exist_ok=True)
            (root / "backlog" / "opportunities").mkdir(parents=True, exist_ok=True)

            (root / "data" / "experience" / "episodes.jsonl").write_text(
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
                            "action_sequence": ["CALL_TOOL", "WRITE_MEM", "ANSWER"],
                        },
                        "steps": [
                            {"step_idx": 1, "action": "CALL_TOOL", "reason": "use calc", "score": 1.0, "selected_tool": "calculator"},
                            {"step_idx": 2, "action": "WRITE_MEM", "reason": "write hyp", "score": 0.8, "target_slot": "HYP"},
                            {"step_idx": 3, "action": "ANSWER", "reason": "finish", "score": 0.9},
                        ],
                    },
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )

            (root / "data" / "desktop" / "bridge_events.jsonl").write_text(
                json.dumps(
                    {
                        "event_id": "evt-1",
                        "timestamp_utc": "2026-04-26T00:00:00+00:00",
                        "kind": "question",
                        "summary": "用户正在比较数据库方案",
                        "prompt": "要不要先比较 PostgreSQL 和 MySQL 的适用性",
                        "source": "desktop",
                        "status": "open",
                        "metadata": {},
                    },
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )
            (root / "data" / "desktop" / "bridge_feedback.jsonl").write_text(
                json.dumps(
                    {
                        "timestamp_utc": "2026-04-26T00:01:00+00:00",
                        "event_id": "evt-1",
                        "feedback_type": "misread",
                        "note": "这里应该先查官方资料再比较",
                        "metadata": {},
                    },
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )

            (root / "data" / "research" / "frontier_observations.jsonl").write_text(
                json.dumps(
                    {
                        "id": "frontier-1",
                        "topic": "tool-use stability under ambiguity",
                        "source": "manual",
                        "label": "borrow",
                        "reason": "close attention handoff gap",
                        "created_at_utc": "2026-04-26T00:00:00+00:00",
                    },
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )
            (root / "data" / "research" / "public_agent_ideas.jsonl").write_text(
                json.dumps(
                    {
                        "topic": "reasoning + acting loop",
                        "insight": "让推理轨迹与工具调用交替显式展开，而不是只模仿最终答案。",
                        "source_name": "ReAct",
                        "source_url": "https://arxiv.org/abs/2210.03629",
                    },
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )

            (root / "artifacts" / "attention_flow_latest.json").write_text(
                json.dumps({"summary": {"premature_release_count": 1}}, ensure_ascii=False),
                encoding="utf-8",
            )
            (root / "artifacts" / "attention_training_views_latest.json").write_text(
                json.dumps(
                    {"summary": {"conflict_to_verification_rate": 0.25, "tool_boundary_block_rate": 0.5}},
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            (root / "data" / "evolution").mkdir(parents=True, exist_ok=True)
            (root / "data" / "evolution" / "learning_focus_evidence_routing_eval_result.json").write_text(
                json.dumps(
                    {
                        "rows": [
                            {
                                "id": "stress-learning-focus-evidence-routing-002",
                                "logic_skill": "evidence_judgment",
                                "expected_tool": "search",
                                "baseline_used_tool": "calculator",
                                "pretrained_used_tool": "calculator",
                            },
                            {
                                "id": "stress-learning-focus-evidence-routing-003",
                                "logic_skill": "evidence_judgment",
                                "expected_tool": "search",
                                "baseline_used_tool": "calculator",
                                "pretrained_used_tool": "search",
                            },
                        ]
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            (root / "data" / "eval").mkdir(parents=True, exist_ok=True)
            (root / "data" / "eval" / "learning_focus_search_first_adversarial_eval.json").write_text(
                json.dumps(
                    {
                        "prompts": [
                            {
                                "id": "search-first-adversarial-002",
                                "prompt": "围绕 tool-use stability under ambiguity，先检索公开资料再区分可借鉴/待验证要点。",
                                "expected_tool": "search",
                                "logic_skill": "evidence_judgment",
                                "mutation": "evidence_gate_before_takeaway",
                                "source_learning_focus_id": "learning-focus-004",
                                "source_prompt": "公开 agent 设计学习：主题=tool-use stability under ambiguity。",
                            }
                        ]
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            (root / "artifacts" / "learning_focus_search_first_adversarial_latest.json").write_text(
                json.dumps(
                    {
                        "current_rows": [
                            {
                                "id": "search-first-adversarial-002",
                                "logic_skill": "evidence_judgment",
                                "expected_tool": "search",
                                "pretrained_match": False,
                                "pretrained_used_tool": "calculator",
                            }
                        ],
                        "shadow_rows": [
                            {
                                "id": "search-first-adversarial-002",
                                "logic_skill": "evidence_judgment",
                                "expected_tool": "search",
                                "pretrained_match": False,
                                "pretrained_used_tool": "calculator",
                            }
                        ],
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

            manifest = build_learning_intake(root, max_per_source=5)

            self.assertGreaterEqual(manifest["candidate_count"], 4)
            self.assertTrue((root / "data" / "learning" / "learning_intake_samples.jsonl").exists())
            self.assertTrue((root / "data" / "learning" / "learning_intake_import.jsonl").exists())
            self.assertTrue((root / "data" / "learning" / "learning_intake_review_pack.jsonl").exists())
            self.assertTrue((root / "backlog" / "opportunities" / "learning_intake_report.md").exists())
            source_counts = manifest["source_counts"]
            self.assertIn("learning_intake:experience", source_counts)
            self.assertIn("learning_intake:bridge_feedback", source_counts)
            self.assertIn("learning_intake:frontier", source_counts)
            self.assertIn("learning_intake:public_idea", source_counts)
            self.assertIn("learning_intake:learning_focus_stress", source_counts)
            self.assertIn("learning_intake:search_first_adversarial_failure", source_counts)


if __name__ == "__main__":
    unittest.main()
