import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from scripts.team_conductor import (
    _build_delivery_decision,
    _classify_delivery_paths,
    bootstrap_workspace,
    run_cycle,
)


class TeamConductorTests(unittest.TestCase):
    def test_build_delivery_decision_groups_core_and_volatile_paths(self) -> None:
        changed_paths = [
            "scripts/team_conductor.py",
            "configs/real_prompt_eval.json",
            "configs/team_github.json",
            "data/team/role_content_history.jsonl",
            "backlog/opportunities/research_latest.md",
            "team/GITHUB_AUTOMATION.md",
            "docs/plans/2026-03-27-auto-delivery-lane-plan.md",
        ]
        groups = _classify_delivery_paths(changed_paths)
        self.assertIn("scripts/team_conductor.py", groups["core"])
        self.assertIn("configs/real_prompt_eval.json", groups["core"])
        self.assertIn("configs/team_github.json", groups["core"])
        self.assertIn("data/team/role_content_history.jsonl", groups["volatile"])
        self.assertIn("backlog/opportunities/research_latest.md", groups["artifacts"])
        self.assertIn("team/GITHUB_AUTOMATION.md", groups["artifacts"])
        self.assertIn("docs/plans/2026-03-27-auto-delivery-lane-plan.md", groups["core"])

        decision = _build_delivery_decision({}, {}, changed_paths, {})
        self.assertIn("should_commit", decision)
        self.assertIn("should_push", decision)
        self.assertIn("should_open_pr", decision)
        self.assertIn("delivery_lane", decision)
        self.assertIn("file_groups", decision)
        self.assertEqual(decision["delivery_lane"], "sync_only")

    def test_build_delivery_decision_selects_pr_lane_when_config_and_direction_allow(self) -> None:
        changed_paths = ["scripts/team_conductor.py"]
        decision = _build_delivery_decision(
            {"alerts": []},
            {"verdict": "direction_correct", "escalate_to_human": False},
            changed_paths,
            {"github_delivery": {"enabled": True, "require_direction_correct": True, "require_clean_alerts": True}},
        )
        self.assertEqual(decision["delivery_lane"], "pr_delivery")
        self.assertTrue(decision["should_open_pr"])

    def test_run_cycle_executes_expand_eval_set_operator(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "configs").mkdir(parents=True, exist_ok=True)
            (root / "data" / "review").mkdir(parents=True, exist_ok=True)
            (root / "data" / "control").mkdir(parents=True, exist_ok=True)
            (root / "data" / "train_runs").mkdir(parents=True, exist_ok=True)
            (root / "data" / "eval").mkdir(parents=True, exist_ok=True)

            (root / "configs" / "team_cycle.json").write_text(
                json.dumps(
                    {
                        "team_name": "mustard-claw",
                        "heartbeat": {"max_new_proposals_per_cycle": 1},
                        "risk_policy": {"human_gate_required_change_types": ["runtime_control"]},
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            (root / "configs" / "real_prompt_eval.json").write_text(
                json.dumps(
                    {
                        "prompts": [
                            {
                                "id": "seed-1",
                                "prompt": "请计算 1 + 2",
                                "expected_tool": "calculator",
                                "logic_skill": "tool_selection",
                            }
                        ]
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            (root / "data" / "eval" / "real_prompt_candidates.json").write_text(
                json.dumps(
                    {
                        "prompts": [
                            {
                                "id": "cand-search",
                                "prompt": "比较 PostgreSQL 和 MySQL 在中小团队里的适用性",
                                "expected_tool": "search",
                                "logic_skill": "comparison",
                            },
                            {
                                "id": "cand-summary",
                                "prompt": "请基于三份材料生成正式结论摘要",
                                "expected_tool": "bigmodel_proxy",
                                "logic_skill": "result_integration",
                            },
                        ]
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            (root / "data" / "review" / "reviews.jsonl").write_text('{"x":1}\n', encoding="utf-8")
            (root / "data" / "control" / "control_state.json").write_text("{}", encoding="utf-8")
            (root / "data" / "train_runs" / "auto_train_latest.json").write_text(
                json.dumps(
                    {
                        "run_id": "run_2",
                        "dataset": {"sample_count": 8},
                        "evaluation": {
                            "real_prompt_eval": {
                                "summary": {
                                    "prompt_count": 1,
                                    "baseline_match_rate": 1.0,
                                    "pretrained_match_rate": 1.0,
                                },
                                "rows": [
                                    {"id": "seed-1", "pretrained_match": True}
                                ],
                            }
                        },
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

            result = run_cycle(root=root, config_path=Path("configs/team_cycle.json"))

            self.assertIn("delivery_decision", result)
            self.assertIn("delivery_lane", result["delivery_decision"])
            self.assertIn("file_groups", result["delivery_decision"])
            self.assertTrue(result["operator_result"]["executed"])
            self.assertEqual(result["operator_result"]["reason"], "expanded_eval_set")
            self.assertTrue(Path(result["operator_result"]["quality_report_path"]).exists())
            self.assertTrue(Path(result["operator_result"]["quality_json_path"]).exists())
            updated_payload = json.loads((root / "configs" / "real_prompt_eval.json").read_text(encoding="utf-8"))
            self.assertGreaterEqual(len(updated_payload.get("prompts", [])), 2)

    def test_bootstrap_workspace_creates_team_directories(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            bootstrap_workspace(root)

            self.assertTrue((root / "team").exists())
            self.assertTrue((root / "memory" / "daily").exists())
            self.assertTrue((root / "backlog" / "proposals").exists())

    def test_run_cycle_writes_digest_and_proposal_for_candidate_rollout(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "configs").mkdir(parents=True, exist_ok=True)
            (root / "data" / "review").mkdir(parents=True, exist_ok=True)
            (root / "data" / "control").mkdir(parents=True, exist_ok=True)
            (root / "data" / "train_runs").mkdir(parents=True, exist_ok=True)

            (root / "configs" / "team_cycle.json").write_text(
                json.dumps(
                    {
                        "team_name": "mustard-claw",
                        "heartbeat": {"max_new_proposals_per_cycle": 3},
                        "risk_policy": {"human_gate_required_change_types": ["runtime_control"]},
                        "focus_areas": ["runtime_control_safety"],
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            (root / "data" / "review" / "reviews.jsonl").write_text('{"x":1}\n', encoding="utf-8")
            (root / "data" / "control" / "control_state.json").write_text(
                json.dumps(
                    {
                        "rollout_status": "candidate",
                        "candidate_version": "v2",
                        "candidate_baseline_version": "v1",
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            (root / "data" / "train_runs" / "auto_train_latest.json").write_text(
                json.dumps(
                    {
                        "run_id": "run_1",
                        "dataset": {"sample_count": 12},
                        "evaluation": {
                            "real_prompt_eval": {
                                "delta_tool_match_rate": -0.15,
                                "summary": {
                                    "prompt_count": 12,
                                    "baseline_match_rate": 0.9,
                                    "pretrained_match_rate": 1.0,
                                },
                                "rows": [
                                    {
                                        "id": "real-1",
                                        "pretrained_match": True,
                                    }
                                ],
                            },
                        },
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

            result = run_cycle(root=root, config_path=Path("configs/team_cycle.json"))

            digest_path = Path(result["digest_path"])
            self.assertTrue(digest_path.exists())
            self.assertIn("candidate_rollout_active", result["alerts"])
            self.assertGreaterEqual(result["proposal_count"], 2)

            proposal_paths = [Path(item) for item in result["proposal_paths"]]
            self.assertEqual(len(proposal_paths), result["proposal_count"])
            self.assertTrue(any("needs_human_approval: True" in path.read_text(encoding="utf-8") for path in proposal_paths))

            self.assertEqual(result["evolution_track"], "real_prompt_coverage")
            self.assertIn(result["operator_result"]["reason"], {"insufficient_bucket_coverage", "no_valid_candidates", "expanded_eval_set"})
            self.assertTrue(Path(result["operator_result"]["quality_json_path"]).exists())
            self.assertTrue(Path(result["quality_stabilization_report_path"]).exists())
            self.assertTrue(Path(result["quality_stabilization_json_path"]).exists())
            if "repair_result" in result["operator_result"]:
                self.assertTrue(Path(result["operator_result"]["repair_result"]["repair_candidates_path"]).exists())
            if result["quality_operator_result"].get("executed"):
                self.assertTrue(Path(result["quality_operator_result"]["quality_hard_variants_path"]).exists())
                self.assertTrue(Path(result["quality_operator_result"]["quality_focus_eval_path"]).exists())
                focus_eval_result = result["quality_operator_result"].get("quality_focus_eval_result", {})
                if focus_eval_result.get("executed"):
                    self.assertTrue(Path(focus_eval_result["result_path"]).exists())
                    benchmark_result = result["quality_operator_result"].get("quality_benchmark_result", {})
                    if benchmark_result:
                        self.assertTrue(Path(benchmark_result["retained_path"]).exists())
                        self.assertTrue(Path(benchmark_result["pruned_path"]).exists())
                        self.assertTrue(Path(benchmark_result["promotion"]["config_path"]).exists())
                        promoted_payload = json.loads(Path(benchmark_result["promotion"]["config_path"]).read_text(encoding="utf-8"))
                        prompts = promoted_payload.get("prompts", []) if isinstance(promoted_payload, dict) else []
                        self.assertTrue(all(str(item.get("prompt", "")).strip() for item in prompts if isinstance(item, dict)))
            self.assertTrue(Path(result["evolution_candidate_path"]).exists())
            self.assertTrue(Path(result["evolution_lineage_path"]).exists())
            self.assertTrue(Path(result["evolution_run_path"]).exists())


if __name__ == "__main__":
    unittest.main()
