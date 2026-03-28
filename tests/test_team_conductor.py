import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from scripts.team_conductor import (
    _build_delivery_decision,
    _classify_delivery_paths,
    _dedupe_and_prioritize_proposals,
    _evaluate_research_quality,
    _load_active_failure_pattern_ids,
    bootstrap_workspace,
    build_proposals,
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


    def test_evaluate_research_quality_flags_degraded_when_patterns_empty_and_zero_streaks(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "configs").mkdir(parents=True, exist_ok=True)
            (root / "memory").mkdir(parents=True, exist_ok=True)
            (root / "backlog" / "opportunities").mkdir(parents=True, exist_ok=True)
            (root / "data" / "team").mkdir(parents=True, exist_ok=True)

            (root / "memory" / "failure_patterns.md").write_text(
                "# Failure Patterns\n\n## Sampling insufficiency / blind spots\n- blind_spot: eval coverage still narrow\n",
                encoding="utf-8",
            )
            (root / "backlog" / "opportunities" / "research_latest.md").write_text(
                "# Research Artifact\n\nOnly summary metrics, no weakness diagnosis.\n",
                encoding="utf-8",
            )
            (root / "data" / "team" / "research_quality_state.json").write_text(
                json.dumps(
                    {
                        "rounds_without_new_failure_pattern": 1,
                        "coverage_only_gap_streak": 1,
                        "zero_bridge_feedback_streak": 2,
                        "zero_frontier_observation_streak": 2,
                        "last_failure_pattern_ids": [],
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

            path, payload = _evaluate_research_quality(
                root,
                {
                    "bridge_feedback": 0,
                    "frontier_observation_count": 0,
                    "real_prompt_eval": {"summary": {"prompt_count": 10}},
                },
                {
                    "research_quality_policy": {
                        "enabled": True,
                        "max_rounds_without_new_failure_pattern": 2,
                        "max_rounds_with_coverage_only_top_gap": 2,
                        "require_nonempty_failure_patterns": True,
                        "flag_zero_bridge_feedback_rounds": 3,
                        "flag_zero_frontier_observation_rounds": 3,
                    }
                },
                root / "memory" / "failure_patterns.md",
                root / "backlog" / "opportunities" / "research_latest.md",
            )

            self.assertTrue(path.exists())
            self.assertTrue(payload["degraded"])
            self.assertIn("failure_patterns_empty", payload["reasons"])
            self.assertIn("coverage_only_top_gap_repetition", payload["reasons"])
            self.assertIn("bridge_zero_feedback_persistence", payload["reasons"])
            self.assertIn("frontier_zero_signal_persistence", payload["reasons"])

    def test_run_cycle_surfaces_research_quality_in_digest_and_result(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "configs").mkdir(parents=True, exist_ok=True)
            (root / "data" / "review").mkdir(parents=True, exist_ok=True)
            (root / "data" / "control").mkdir(parents=True, exist_ok=True)
            (root / "data" / "train_runs").mkdir(parents=True, exist_ok=True)
            (root / "memory").mkdir(parents=True, exist_ok=True)
            (root / "backlog" / "opportunities").mkdir(parents=True, exist_ok=True)

            (root / "configs" / "team_cycle.json").write_text(
                json.dumps(
                    {
                        "team_name": "mustard-claw",
                        "heartbeat": {"max_new_proposals_per_cycle": 1},
                        "risk_policy": {"human_gate_required_change_types": ["runtime_control"]},
                        "research_quality_policy": {
                            "enabled": True,
                            "max_rounds_without_new_failure_pattern": 1,
                            "max_rounds_with_coverage_only_top_gap": 1,
                            "require_nonempty_failure_patterns": True,
                            "flag_zero_bridge_feedback_rounds": 1,
                            "flag_zero_frontier_observation_rounds": 1,
                        },
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            (root / "memory" / "failure_patterns.md").write_text("# Failure Patterns\n", encoding="utf-8")
            (root / "data" / "review" / "reviews.jsonl").write_text('{"x":1}\n', encoding="utf-8")
            (root / "data" / "control" / "control_state.json").write_text("{}", encoding="utf-8")
            (root / "data" / "train_runs" / "auto_train_latest.json").write_text(
                json.dumps(
                    {
                        "run_id": "run_3",
                        "dataset": {"sample_count": 8},
                        "evaluation": {
                            "real_prompt_eval": {
                                "summary": {
                                    "prompt_count": 10,
                                    "baseline_match_rate": 0.8,
                                    "pretrained_match_rate": 1.0,
                                },
                                "rows": [{"id": "seed-1", "pretrained_match": True}],
                            }
                        },
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

            result = run_cycle(root=root, config_path=Path("configs/team_cycle.json"))
            self.assertIn("research_quality", result)
            self.assertTrue(result["research_quality"]["degraded"])
            self.assertIn("research_quality_degraded", result["alerts"])

            digest_text = Path(result["digest_path"]).read_text(encoding="utf-8")
            self.assertIn("- research_quality: degraded", digest_text)
            self.assertIn("- research_reasons:", digest_text)

    def test_run_cycle_writes_real_failure_patterns_into_memory_file(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "configs").mkdir(parents=True, exist_ok=True)
            (root / "data" / "review").mkdir(parents=True, exist_ok=True)
            (root / "data" / "control").mkdir(parents=True, exist_ok=True)
            (root / "data" / "train_runs").mkdir(parents=True, exist_ok=True)
            (root / "memory").mkdir(parents=True, exist_ok=True)

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
            (root / "data" / "review" / "reviews.jsonl").write_text('{"x":1}\n', encoding="utf-8")
            (root / "data" / "control" / "control_state.json").write_text("{}", encoding="utf-8")
            (root / "data" / "train_runs" / "auto_train_latest.json").write_text(
                json.dumps(
                    {
                        "run_id": "run_4",
                        "dataset": {"sample_count": 8},
                        "evaluation": {
                            "real_prompt_eval": {
                                "summary": {
                                    "prompt_count": 20,
                                    "baseline_match_rate": 0.8,
                                    "pretrained_match_rate": 1.0,
                                },
                                "rows": [{"id": "seed-1", "pretrained_match": True}],
                            }
                        },
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

            run_cycle(root=root, config_path=Path("configs/team_cycle.json"))
            failure_text = (root / "memory" / "failure_patterns.md").read_text(encoding="utf-8")
            self.assertIn("pattern_id: sampling_blind_spot", failure_text)
            self.assertIn("pattern_id: frontier_research_blindspot", failure_text)
            self.assertIn("pattern_id: no_tool_feedback_loop", failure_text)

    def test_build_proposals_prioritizes_new_failure_patterns(self) -> None:
        proposals = build_proposals(
            {
                "signals": {
                    "bridge_feedback": 0,
                    "frontier_observation_count": 0,
                    "dataset_sample_count": 186,
                    "real_prompt_eval": {"summary": {"prompt_count": 20}},
                },
                "research_quality": {
                    "new_failure_pattern_ids": [
                        "sampling_blind_spot",
                        "frontier_research_blindspot",
                        "no_tool_feedback_loop",
                    ]
                },
            },
            {
                "heartbeat": {"max_new_proposals_per_cycle": 5},
                "risk_policy": {"human_gate_required_change_types": ["runtime_control"]},
            },
            {"last_mode": "max_landing", "stagnation_rounds": 10},
        )
        titles = [item.get("title", "") for item in proposals]
        self.assertIn("Exploit sampling blind spot with high-information real prompts", titles)
        self.assertIn("Reopen frontier research intake with minimum observation quota", titles)
        self.assertIn("Bootstrap bridge feedback capture without changing defaults", titles)
        self.assertTrue(any(item.get("architect_handoff") for item in proposals))

    def test_dedupe_and_prioritize_prefers_richer_failure_pattern_proposal(self) -> None:
        proposals = _dedupe_and_prioritize_proposals(
            [
                {
                    "title": "Kickstart desktop-bridge feedback loop",
                    "problem": "桌面桥梁暂无反馈样本",
                    "change_type": "desktop_behavior",
                    "proposed_change": "采集 bridge useful/misread 反馈",
                    "needs_human_approval": True,
                },
                {
                    "title": "Bootstrap bridge feedback capture without changing defaults",
                    "problem": "bridge feedback 持续为 0",
                    "from_failure_pattern": "no_tool_feedback_loop",
                    "from_top_gap": "research_quality_degraded",
                    "change_type": "desktop_behavior",
                    "proposed_change": "先补反馈采样与整理入口，只记录 useful/misread",
                    "expected_metric_delta": "bridge_feedback > 0",
                    "relative_to_last_round": "从抽象提桥梁闭环，改为先把反馈入口打通。",
                    "scenario_fit": "用户在真实桌面协作里纠偏系统误读/误触发的场景。",
                    "architect_handoff": "guardian reviews scope, then failure_miner defines feedback capture schema before any rollout",
                    "needs_human_approval": True,
                    "evaluation_plan": ["python -m scripts.desktop_agent_control snapshot"],
                },
            ]
        )
        self.assertEqual(len(proposals), 1)
        self.assertEqual(proposals[0]["title"], "Bootstrap bridge feedback capture without changing defaults")

    def test_load_active_failure_pattern_ids_reads_auto_incidents(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "backlog" / "incidents").mkdir(parents=True, exist_ok=True)
            (root / "backlog" / "incidents" / "auto_failure_patterns.json").write_text(
                json.dumps(
                    {
                        "patterns": [
                            {"id": "frontier_research_blindspot"},
                            {"id": "no_tool_feedback_loop"},
                        ]
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            active = _load_active_failure_pattern_ids(root)
            self.assertEqual(active, {"frontier_research_blindspot", "no_tool_feedback_loop"})

    def test_build_proposals_uses_persistent_research_quality_reasons(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "backlog" / "incidents").mkdir(parents=True, exist_ok=True)
            (root / "backlog" / "incidents" / "auto_failure_patterns.json").write_text(
                json.dumps(
                    {
                        "patterns": [
                            {"id": "frontier_research_blindspot"},
                            {"id": "no_tool_feedback_loop"},
                            {"id": "sampling_blind_spot"},
                        ]
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            proposals = build_proposals(
                {
                    "workspace_root": str(root),
                    "signals": {
                        "bridge_feedback": 0,
                        "frontier_observation_count": 0,
                        "dataset_sample_count": 186,
                        "real_prompt_eval": {"summary": {"prompt_count": 20}},
                    },
                    "research_quality": {
                        "new_failure_pattern_ids": [],
                        "reasons": [
                            "no_new_failure_pattern",
                            "bridge_zero_feedback_persistence",
                            "frontier_zero_signal_persistence",
                        ],
                    },
                },
                {
                    "heartbeat": {"max_new_proposals_per_cycle": 5},
                    "risk_policy": {"human_gate_required_change_types": ["desktop_behavior"]},
                },
                {"last_mode": "max_landing", "stagnation_rounds": 10},
            )
            titles = [item.get("title", "") for item in proposals]
            self.assertIn("Exploit sampling blind spot with high-information real prompts", titles)
            self.assertIn("Reopen frontier research intake with minimum observation quota", titles)
            self.assertIn("Bootstrap bridge feedback capture without changing defaults", titles)

    def test_run_cycle_writes_architect_handoff_into_outputs(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "configs").mkdir(parents=True, exist_ok=True)
            (root / "data" / "review").mkdir(parents=True, exist_ok=True)
            (root / "data" / "control").mkdir(parents=True, exist_ok=True)
            (root / "data" / "train_runs").mkdir(parents=True, exist_ok=True)
            (root / "memory").mkdir(parents=True, exist_ok=True)

            (root / "configs" / "team_cycle.json").write_text(
                json.dumps(
                    {
                        "team_name": "mustard-claw",
                        "heartbeat": {"max_new_proposals_per_cycle": 1},
                        "risk_policy": {"human_gate_required_change_types": ["desktop_behavior"]},
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            (root / "data" / "review" / "reviews.jsonl").write_text('{"x":1}\n', encoding="utf-8")
            (root / "data" / "control" / "control_state.json").write_text("{}", encoding="utf-8")
            (root / "data" / "train_runs" / "auto_train_latest.json").write_text(
                json.dumps(
                    {
                        "run_id": "run_5",
                        "dataset": {"sample_count": 186},
                        "evaluation": {
                            "real_prompt_eval": {
                                "summary": {
                                    "prompt_count": 20,
                                    "baseline_match_rate": 0.8,
                                    "pretrained_match_rate": 1.0,
                                },
                                "rows": [{"id": "seed-1", "pretrained_match": True}],
                            }
                        },
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

            result = run_cycle(root=root, config_path=Path("configs/team_cycle.json"))
            proposal_text = Path(result["proposal_paths"][0]).read_text(encoding="utf-8")
            architect_text = (root / "backlog" / "role_outputs" / "architect_latest.md").read_text(encoding="utf-8")
            self.assertIn("- architect_handoff:", proposal_text)
            self.assertNotIn("- architect_handoff: direct_execute_if_format_passes", proposal_text)
            self.assertIn("- first_architect_handoff:", architect_text)
            self.assertTrue("- first_from_failure_pattern:" in architect_text or "- first_from_top_gap:" in architect_text)


if __name__ == "__main__":
    unittest.main()
