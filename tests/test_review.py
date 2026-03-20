import json
import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from carm.runner import AgentRunner
from carm.review import ReviewStore
from tools.base import ToolManager
from tools.bigmodel_tool import BigModelProxyTool
from tools.calc_tool import CalculatorTool
from tools.code_tool import CodeExecutorTool
from tools.search_tool import SearchTool


class ReviewTests(unittest.TestCase):
    def test_runner_emits_review_record(self) -> None:
        with TemporaryDirectory() as temp_dir:
            runner = AgentRunner(
                ToolManager([SearchTool(), CalculatorTool(), CodeExecutorTool(), BigModelProxyTool()]),
                experience_path=Path(temp_dir) / "episodes.jsonl",
                policy_state_path=Path(temp_dir) / "policy_state.json",
                concept_state_path=Path(temp_dir) / "concept_state.json",
                core_state_path=Path(temp_dir) / "core_state.json",
                review_path=Path(temp_dir) / "reviews.jsonl",
            )
            runner.run("请计算 12 + 30 / 3")

            reviews = ReviewStore(Path(temp_dir) / "reviews.jsonl").load_all()
            self.assertEqual(len(reviews), 1)
            self.assertTrue(reviews[0].issue_tags)
            self.assertTrue(reviews[0].recommendations)

    def test_consolidation_payload_shape(self) -> None:
        with TemporaryDirectory() as temp_dir:
            review_path = Path(temp_dir) / "reviews.jsonl"
            review_path.write_text(
                json.dumps(
                    {
                        "user_input": "x",
                        "success": True,
                        "value_score": 0.9,
                        "issue_tags": ["stable_path"],
                        "strengths": ["ok"],
                        "weaknesses": [],
                        "recommendations": ["keep"],
                        "target_modules": ["policy"],
                        "evidence": {"action_sequence": ["WRITE_MEM", "ANSWER"]},
                    },
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )
            output_path = Path(temp_dir) / "consolidated_recommendations.json"

            from scripts.consolidate_reviews import main as consolidate_main

            current = Path.cwd()
            try:
                import os

                os.chdir(temp_dir)
                os.environ["CARM_REVIEW_PATH"] = str(review_path)
                os.environ["CARM_REVIEW_OUTPUT"] = str(output_path)
                consolidate_main()
            finally:
                os.environ.pop("CARM_REVIEW_PATH", None)
                os.environ.pop("CARM_REVIEW_OUTPUT", None)
                os.chdir(current)

            payload = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["review_count"], 1)
            self.assertTrue(payload["top_issue_tags"])
            self.assertIn("slow_path_actions", payload)

    def test_build_payload_emits_actionable_guidance(self) -> None:
        from scripts.consolidate_reviews import build_payload
        from carm.schemas import ReviewRecord

        reviews = [
            ReviewRecord(
                user_input="x",
                success=False,
                value_score=0.3,
                issue_tags=["idle_drift", "tool_underuse", "weak_grounding"],
                strengths=[],
                weaknesses=[],
                recommendations=["raise tool usage"],
                target_modules=["policy", "core"],
                evidence={"glance_triggers": ["high_uncertainty"], "glance_help_rate": 0.2},
            ),
            ReviewRecord(
                user_input="y",
                success=False,
                value_score=0.4,
                issue_tags=["idle_drift", "tool_underuse", "weak_grounding"],
                strengths=[],
                weaknesses=[],
                recommendations=["raise tool usage"],
                target_modules=["policy", "core"],
                evidence={"glance_triggers": ["high_uncertainty"], "glance_help_rate": 0.1},
            ),
        ]
        payload = build_payload(reviews)

        self.assertTrue(payload["slow_path_actions"])
        self.assertTrue(any(action["target_module"] in {"policy", "core", "glance"} for action in payload["slow_path_actions"]))

    def test_system_status_reports_counts(self) -> None:
        with TemporaryDirectory() as temp_dir:
            data_dir = Path(temp_dir) / "data"
            (data_dir / "experience").mkdir(parents=True, exist_ok=True)
            (data_dir / "review").mkdir(parents=True, exist_ok=True)
            (data_dir / "control").mkdir(parents=True, exist_ok=True)
            (data_dir / "experience" / "episodes.jsonl").write_text('{"x":1}\n{"x":2}\n', encoding="utf-8")
            (data_dir / "review" / "reviews.jsonl").write_text('{"x":1}\n', encoding="utf-8")
            (data_dir / "experience" / "policy_state.json").write_text('{"action_weights":{"A":{}}}', encoding="utf-8")
            (data_dir / "experience" / "concept_state.json").write_text('{"token_action_weights":{"x":{}}}', encoding="utf-8")
            (data_dir / "experience" / "core_state.json").write_text('{"feature_weights":{"PLAN":{}}}', encoding="utf-8")
            (data_dir / "review" / "consolidated_recommendations.json").write_text(
                json.dumps({"glance_summary": {"average_help_rate": 0.5}, "slow_path_actions": [{"type": "observe", "target_module": "policy", "proposal": "keep"}]}),
                encoding="utf-8",
            )
            (data_dir / "control" / "control_state.json").write_text(
                json.dumps({"current_version": "v2", "previous_version": "v1", "last_reason": "apply_slow_path_actions", "last_updated_utc": "2026-03-20T00:00:00+00:00"}),
                encoding="utf-8",
            )
            (data_dir / "control" / "control_versions.jsonl").write_text(
                json.dumps({"version_id": "v1", "action_types": ["bootstrap"]}) + "\n"
                + json.dumps({"version_id": "v2", "action_types": ["raise_tool_bias"]}) + "\n",
                encoding="utf-8",
            )
            (data_dir / "control" / "control_version_metrics.json").write_text(
                json.dumps(
                    {
                        "version_metrics": {
                            "v2": {
                                "success_rate": 0.8,
                                "avg_value_score": 0.7,
                                "avg_step_count": 3.5,
                            }
                        },
                        "comparison": {
                            "delta_success_rate": 0.2,
                            "delta_avg_value_score": 0.1,
                            "delta_avg_step_count": -0.5,
                        },
                    }
                ),
                encoding="utf-8",
            )

            from scripts.system_status import main as status_main

            current = Path.cwd()
            try:
                os.chdir(temp_dir)
                from io import StringIO
                import sys

                buffer = StringIO()
                previous_stdout = sys.stdout
                sys.stdout = buffer
                status_main()
                output = buffer.getvalue()
            finally:
                sys.stdout = previous_stdout
                os.chdir(current)

            self.assertIn("episodes: 2", output)
            self.assertIn("reviews: 1", output)
            self.assertIn("current_control_version: v2", output)
            self.assertIn("control_history_count: 2", output)
            self.assertIn("current_version_success_rate: 0.8", output)
            self.assertIn("control_delta_success_rate: 0.2", output)

    def test_evaluate_control_versions_builds_metrics(self) -> None:
        with TemporaryDirectory() as temp_dir:
            experience_path = Path(temp_dir) / "episodes.jsonl"
            control_state_path = Path(temp_dir) / "control_state.json"
            samples_path = Path(temp_dir) / "control_cycle_samples.jsonl"
            output_path = Path(temp_dir) / "control_version_metrics.json"
            experience_path.write_text(
                "\n".join(
                    [
                        json.dumps(
                            {
                                "user_input": "a",
                                "answer": "ok",
                                "summary": "x",
                                "success": True,
                                "value_score": 0.8,
                                "episode_features": {"control_version": "v1", "used_tool": "search"},
                                "outcome_signature": {"control_version": "v1", "uncertainty": 0.2},
                                "steps": [{"step_idx": 0, "action": "CALL_TOOL", "reason": "x", "score": 1.0, "glance_used": True}],
                            }
                        ),
                        json.dumps(
                            {
                                "user_input": "b",
                                "answer": "ok",
                                "summary": "y",
                                "success": False,
                                "value_score": 0.3,
                                "episode_features": {"control_version": "v2", "used_tool": ""},
                                "outcome_signature": {"control_version": "v2", "uncertainty": 0.7},
                                "steps": [{"step_idx": 0, "action": "ANSWER", "reason": "y", "score": 0.5, "glance_used": False}],
                            }
                        ),
                        json.dumps(
                            {
                                "user_input": "c",
                                "answer": "ok",
                                "summary": "z",
                                "success": True,
                                "value_score": 0.9,
                                "episode_features": {"control_version": "v2", "used_tool": "calc"},
                                "outcome_signature": {"control_version": "v2", "uncertainty": 0.1},
                                "steps": [{"step_idx": 0, "action": "CALL_TOOL", "reason": "z", "score": 0.9, "glance_used": True}],
                            }
                        ),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            samples_path.write_text(
                json.dumps({"control_version": "v1", "tag": "numeric", "expected_tool": "calculator", "used_tool": "calculator", "tool_match": True})
                + "\n"
                + json.dumps({"control_version": "v2", "tag": "comparison", "expected_tool": "search", "used_tool": "search", "tool_match": True})
                + "\n",
                encoding="utf-8",
            )
            control_state_path.write_text(
                json.dumps({"current_version": "v2", "previous_version": "v1"}),
                encoding="utf-8",
            )

            from scripts.evaluate_control_versions import main as evaluate_main

            os.environ["CARM_EXPERIENCE_PATH"] = str(experience_path)
            os.environ["CARM_CONTROL_STATE_PATH"] = str(control_state_path)
            os.environ["CARM_CONTROL_SAMPLES_PATH"] = str(samples_path)
            os.environ["CARM_CONTROL_METRICS_PATH"] = str(output_path)
            try:
                evaluate_main()
            finally:
                os.environ.pop("CARM_EXPERIENCE_PATH", None)
                os.environ.pop("CARM_CONTROL_STATE_PATH", None)
                os.environ.pop("CARM_CONTROL_SAMPLES_PATH", None)
                os.environ.pop("CARM_CONTROL_METRICS_PATH", None)

            payload = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["episode_count"], 3)
            self.assertIn("v1", payload["version_metrics"])
            self.assertIn("v2", payload["version_metrics"])
            self.assertEqual(payload["version_metrics"]["v2"]["episode_count"], 2)
            self.assertIn("v2", payload["sample_metrics"])
            self.assertEqual(payload["sample_metrics"]["v2"]["by_tag"]["comparison"]["tool_match_rate"], 1.0)
            self.assertIn("delta_success_rate", payload["comparison"])

    def test_apply_slow_path_actions_updates_controls(self) -> None:
        with TemporaryDirectory() as temp_dir:
            consolidated_path = Path(temp_dir) / "consolidated_recommendations.json"
            controls_path = Path(temp_dir) / "runtime_controls.json"
            audit_path = Path(temp_dir) / "applied_actions.jsonl"
            versions_path = Path(temp_dir) / "control_versions.jsonl"
            state_path = Path(temp_dir) / "control_state.json"
            history_dir = Path(temp_dir) / "history"
            consolidated_path.write_text(
                json.dumps(
                    {
                        "slow_path_actions": [
                            {
                                "type": "raise_tool_bias",
                                "target_module": "policy",
                                "reason": "test",
                                "proposal": "test",
                            }
                        ]
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

            from scripts.apply_slow_path_actions import main as apply_main

            os.environ["CARM_CONSOLIDATED_PATH"] = str(consolidated_path)
            os.environ["CARM_CONTROLS_PATH"] = str(controls_path)
            os.environ["CARM_APPLY_AUDIT_PATH"] = str(audit_path)
            os.environ["CARM_CONTROL_VERSIONS_PATH"] = str(versions_path)
            os.environ["CARM_CONTROL_STATE_PATH"] = str(state_path)
            os.environ["CARM_CONTROL_HISTORY_DIR"] = str(history_dir)
            try:
                apply_main()
            finally:
                os.environ.pop("CARM_CONSOLIDATED_PATH", None)
                os.environ.pop("CARM_CONTROLS_PATH", None)
                os.environ.pop("CARM_APPLY_AUDIT_PATH", None)
                os.environ.pop("CARM_CONTROL_VERSIONS_PATH", None)
                os.environ.pop("CARM_CONTROL_STATE_PATH", None)
                os.environ.pop("CARM_CONTROL_HISTORY_DIR", None)

            controls = json.loads(controls_path.read_text(encoding="utf-8"))
            state = json.loads(state_path.read_text(encoding="utf-8"))
            versions = [json.loads(line) for line in versions_path.read_text(encoding="utf-8").splitlines() if line.strip()]
            self.assertGreater(controls["policy"]["call_tool_bonus"], 0.0)
            self.assertTrue(audit_path.exists())
            self.assertEqual(len(versions), 2)
            self.assertTrue(state["current_version"])
            self.assertEqual(state["previous_version"], versions[0]["version_id"])
            self.assertTrue(history_dir.exists())
            self.assertEqual(state["rollout_status"], "candidate")
            self.assertEqual(state["candidate_version"], state["current_version"])

    def test_rollback_runtime_controls_restores_previous_version(self) -> None:
        with TemporaryDirectory() as temp_dir:
            consolidated_path = Path(temp_dir) / "consolidated_recommendations.json"
            controls_path = Path(temp_dir) / "runtime_controls.json"
            apply_audit_path = Path(temp_dir) / "applied_actions.jsonl"
            rollback_audit_path = Path(temp_dir) / "rollback_actions.jsonl"
            versions_path = Path(temp_dir) / "control_versions.jsonl"
            state_path = Path(temp_dir) / "control_state.json"
            history_dir = Path(temp_dir) / "history"
            consolidated_path.write_text(
                json.dumps(
                    {
                        "slow_path_actions": [
                            {
                                "type": "raise_tool_bias",
                                "target_module": "policy",
                                "reason": "test",
                                "proposal": "test",
                            }
                        ]
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

            from scripts.apply_slow_path_actions import main as apply_main
            from scripts.rollback_runtime_controls import main as rollback_main

            os.environ["CARM_CONSOLIDATED_PATH"] = str(consolidated_path)
            os.environ["CARM_CONTROLS_PATH"] = str(controls_path)
            os.environ["CARM_APPLY_AUDIT_PATH"] = str(apply_audit_path)
            os.environ["CARM_CONTROL_VERSIONS_PATH"] = str(versions_path)
            os.environ["CARM_CONTROL_STATE_PATH"] = str(state_path)
            os.environ["CARM_CONTROL_HISTORY_DIR"] = str(history_dir)
            os.environ["CARM_ROLLBACK_AUDIT_PATH"] = str(rollback_audit_path)
            try:
                apply_main()
                after_apply = json.loads(controls_path.read_text(encoding="utf-8"))
                rollback_main()
            finally:
                os.environ.pop("CARM_CONSOLIDATED_PATH", None)
                os.environ.pop("CARM_CONTROLS_PATH", None)
                os.environ.pop("CARM_APPLY_AUDIT_PATH", None)
                os.environ.pop("CARM_CONTROL_VERSIONS_PATH", None)
                os.environ.pop("CARM_CONTROL_STATE_PATH", None)
                os.environ.pop("CARM_CONTROL_HISTORY_DIR", None)
                os.environ.pop("CARM_ROLLBACK_AUDIT_PATH", None)
                os.environ.pop("CARM_TARGET_VERSION", None)

            rolled_back = json.loads(controls_path.read_text(encoding="utf-8"))
            state = json.loads(state_path.read_text(encoding="utf-8"))
            versions = [json.loads(line) for line in versions_path.read_text(encoding="utf-8").splitlines() if line.strip()]
            self.assertGreater(after_apply["policy"]["call_tool_bonus"], 0.0)
            self.assertEqual(rolled_back["policy"]["call_tool_bonus"], 0.0)
            self.assertEqual(len(versions), 3)
            self.assertEqual(state["last_reason"], "rollback")
            self.assertTrue(rollback_audit_path.exists())

    def test_judge_control_rollout_pending_when_budget_not_met(self) -> None:
        with TemporaryDirectory() as temp_dir:
            state_path = Path(temp_dir) / "control_state.json"
            metrics_path = Path(temp_dir) / "control_version_metrics.json"
            audit_path = Path(temp_dir) / "rollout_judgments.jsonl"
            state_path.write_text(
                json.dumps(
                    {
                        "rollout_status": "candidate",
                        "candidate_version": "v2",
                        "candidate_baseline_version": "v1",
                        "candidate_episode_budget": 3,
                    }
                ),
                encoding="utf-8",
            )
            metrics_path.write_text(
                json.dumps({"version_metrics": {"v2": {"episode_count": 1}, "v1": {"episode_count": 5, "success_rate": 0.8, "avg_value_score": 0.7, "avg_step_count": 4.0}}}),
                encoding="utf-8",
            )

            from scripts.judge_control_rollout import main as judge_main

            os.environ["CARM_CONTROL_STATE_PATH"] = str(state_path)
            os.environ["CARM_CONTROL_METRICS_PATH"] = str(metrics_path)
            os.environ["CARM_ROLLOUT_AUDIT_PATH"] = str(audit_path)
            try:
                judge_main()
            finally:
                os.environ.pop("CARM_CONTROL_STATE_PATH", None)
                os.environ.pop("CARM_CONTROL_METRICS_PATH", None)
                os.environ.pop("CARM_ROLLOUT_AUDIT_PATH", None)

            state = json.loads(state_path.read_text(encoding="utf-8"))
            self.assertEqual(state["rollout_status"], "candidate")
            self.assertTrue(audit_path.exists())

    def test_judge_control_rollout_promotes_candidate(self) -> None:
        with TemporaryDirectory() as temp_dir:
            state_path = Path(temp_dir) / "control_state.json"
            metrics_path = Path(temp_dir) / "control_version_metrics.json"
            audit_path = Path(temp_dir) / "rollout_judgments.jsonl"
            state_path.write_text(
                json.dumps(
                    {
                        "rollout_status": "candidate",
                        "candidate_version": "v2",
                        "candidate_baseline_version": "v1",
                        "candidate_episode_budget": 2,
                    }
                ),
                encoding="utf-8",
            )
            metrics_path.write_text(
                json.dumps(
                    {
                        "version_metrics": {
                            "v1": {"episode_count": 3, "success_rate": 0.8, "avg_value_score": 0.7, "avg_step_count": 4.0},
                            "v2": {"episode_count": 2, "success_rate": 0.8, "avg_value_score": 0.72, "avg_step_count": 4.5},
                        },
                        "sample_metrics": {
                            "v1": {"by_tag": {"comparison": {"expected_tool_count": 1, "tool_match_rate": 1.0}}},
                            "v2": {"by_tag": {"comparison": {"expected_tool_count": 1, "tool_match_rate": 1.0}}},
                        },
                    }
                ),
                encoding="utf-8",
            )

            from scripts.judge_control_rollout import main as judge_main

            os.environ["CARM_CONTROL_STATE_PATH"] = str(state_path)
            os.environ["CARM_CONTROL_METRICS_PATH"] = str(metrics_path)
            os.environ["CARM_ROLLOUT_AUDIT_PATH"] = str(audit_path)
            try:
                judge_main()
            finally:
                os.environ.pop("CARM_CONTROL_STATE_PATH", None)
                os.environ.pop("CARM_CONTROL_METRICS_PATH", None)
                os.environ.pop("CARM_ROLLOUT_AUDIT_PATH", None)

            state = json.loads(state_path.read_text(encoding="utf-8"))
            self.assertEqual(state["rollout_status"], "stable")
            self.assertEqual(state["last_rollout_decision"], "promote")

    def test_judge_control_rollout_blocks_promotion_on_tool_mismatch(self) -> None:
        with TemporaryDirectory() as temp_dir:
            state_path = Path(temp_dir) / "control_state.json"
            metrics_path = Path(temp_dir) / "control_version_metrics.json"
            audit_path = Path(temp_dir) / "rollout_judgments.jsonl"
            config_path = Path(temp_dir) / "control_cycle.json"
            state_path.write_text(
                json.dumps(
                    {
                        "rollout_status": "candidate",
                        "candidate_version": "v2",
                        "candidate_baseline_version": "v1",
                        "candidate_episode_budget": 2,
                    }
                ),
                encoding="utf-8",
            )
            metrics_path.write_text(
                json.dumps(
                    {
                        "version_metrics": {
                            "v1": {"episode_count": 3, "success_rate": 0.8, "avg_value_score": 0.7, "avg_step_count": 4.0},
                            "v2": {"episode_count": 2, "success_rate": 0.8, "avg_value_score": 0.72, "avg_step_count": 4.5},
                        },
                        "sample_metrics": {
                            "v1": {"by_tag": {"comparison": {"expected_tool_count": 1, "tool_match_rate": 1.0}}},
                            "v2": {"by_tag": {"comparison": {"expected_tool_count": 1, "tool_match_rate": 0.0}}},
                        },
                    }
                ),
                encoding="utf-8",
            )
            config_path.write_text(
                json.dumps(
                    {
                        "rollout_tool_gate": {
                            "default": {"min_tool_match_rate": 0.5, "tool_match_drop_limit": -0.25},
                            "by_tag": {
                                "comparison": {"min_tool_match_rate": 0.8, "tool_match_drop_limit": -0.15},
                            },
                        }
                    }
                ),
                encoding="utf-8",
            )

            from scripts.judge_control_rollout import main as judge_main

            os.environ["CARM_CONTROL_STATE_PATH"] = str(state_path)
            os.environ["CARM_CONTROL_METRICS_PATH"] = str(metrics_path)
            os.environ["CARM_ROLLOUT_AUDIT_PATH"] = str(audit_path)
            os.environ["CARM_CONTROL_CYCLE_CONFIG"] = str(config_path)
            try:
                judge_main()
            finally:
                os.environ.pop("CARM_CONTROL_STATE_PATH", None)
                os.environ.pop("CARM_CONTROL_METRICS_PATH", None)
                os.environ.pop("CARM_ROLLOUT_AUDIT_PATH", None)
                os.environ.pop("CARM_CONTROL_CYCLE_CONFIG", None)

            state = json.loads(state_path.read_text(encoding="utf-8"))
            self.assertEqual(state["rollout_status"], "candidate")
            audit = [json.loads(line) for line in audit_path.read_text(encoding="utf-8").splitlines() if line.strip()]
            self.assertEqual(audit[-1]["decision"], "rollback")
            self.assertIn("tool_gate", audit[-1]["reason"])

    def test_judge_control_rollout_uses_tag_specific_thresholds(self) -> None:
        with TemporaryDirectory() as temp_dir:
            state_path = Path(temp_dir) / "control_state.json"
            metrics_path = Path(temp_dir) / "control_version_metrics.json"
            audit_path = Path(temp_dir) / "rollout_judgments.jsonl"
            config_path = Path(temp_dir) / "control_cycle.json"
            state_path.write_text(
                json.dumps(
                    {
                        "rollout_status": "candidate",
                        "candidate_version": "v2",
                        "candidate_baseline_version": "v1",
                        "candidate_episode_budget": 2,
                    }
                ),
                encoding="utf-8",
            )
            metrics_path.write_text(
                json.dumps(
                    {
                        "version_metrics": {
                            "v1": {"episode_count": 3, "success_rate": 0.8, "avg_value_score": 0.7, "avg_step_count": 4.0},
                            "v2": {"episode_count": 2, "success_rate": 0.8, "avg_value_score": 0.72, "avg_step_count": 4.5},
                        },
                        "sample_metrics": {
                            "v1": {"by_tag": {"coding": {"expected_tool_count": 1, "tool_match_rate": 0.6}}},
                            "v2": {"by_tag": {"coding": {"expected_tool_count": 1, "tool_match_rate": 0.6}}},
                        },
                    }
                ),
                encoding="utf-8",
            )
            config_path.write_text(
                json.dumps(
                    {
                        "rollout_tool_gate": {
                            "default": {"min_tool_match_rate": 0.5, "tool_match_drop_limit": -0.25},
                            "by_tag": {
                                "coding": {"min_tool_match_rate": 0.5, "tool_match_drop_limit": -0.25},
                                "comparison": {"min_tool_match_rate": 0.8, "tool_match_drop_limit": -0.15},
                            },
                        }
                    }
                ),
                encoding="utf-8",
            )

            from scripts.judge_control_rollout import main as judge_main

            os.environ["CARM_CONTROL_STATE_PATH"] = str(state_path)
            os.environ["CARM_CONTROL_METRICS_PATH"] = str(metrics_path)
            os.environ["CARM_ROLLOUT_AUDIT_PATH"] = str(audit_path)
            os.environ["CARM_CONTROL_CYCLE_CONFIG"] = str(config_path)
            try:
                judge_main()
            finally:
                os.environ.pop("CARM_CONTROL_STATE_PATH", None)
                os.environ.pop("CARM_CONTROL_METRICS_PATH", None)
                os.environ.pop("CARM_ROLLOUT_AUDIT_PATH", None)
                os.environ.pop("CARM_CONTROL_CYCLE_CONFIG", None)

            state = json.loads(state_path.read_text(encoding="utf-8"))
            self.assertEqual(state["rollout_status"], "stable")
            self.assertEqual(state["last_rollout_decision"], "promote")

    def test_judge_control_rollout_can_auto_rollback(self) -> None:
        with TemporaryDirectory() as temp_dir:
            controls_path = Path(temp_dir) / "runtime_controls.json"
            state_path = Path(temp_dir) / "control_state.json"
            versions_path = Path(temp_dir) / "control_versions.jsonl"
            history_dir = Path(temp_dir) / "history"
            metrics_path = Path(temp_dir) / "control_version_metrics.json"
            audit_path = Path(temp_dir) / "rollout_judgments.jsonl"

            baseline_controls = {"policy": {"call_tool_bonus": 0.0, "verify_bonus": 0.0, "think_penalty": 0.0, "answer_penalty": 0.0}, "glance": {"budget": 1, "high_uncertainty_threshold": 0.78}, "core": {"result_draft_answer_ready_bonus": 0.0, "result_draft_uncertainty_delta": 0.0}}
            candidate_controls = {"policy": {"call_tool_bonus": 0.2, "verify_bonus": 0.0, "think_penalty": 0.0, "answer_penalty": 0.0}, "glance": {"budget": 1, "high_uncertainty_threshold": 0.78}, "core": {"result_draft_answer_ready_bonus": 0.0, "result_draft_uncertainty_delta": 0.0}}

            history_dir.mkdir(parents=True, exist_ok=True)
            baseline_path = history_dir / "v1.json"
            candidate_path = history_dir / "v2.json"
            baseline_path.write_text(json.dumps(baseline_controls), encoding="utf-8")
            candidate_path.write_text(json.dumps(candidate_controls), encoding="utf-8")
            controls_path.write_text(json.dumps(candidate_controls), encoding="utf-8")
            state_path.write_text(
                json.dumps(
                    {
                        "current_version": "v2",
                        "previous_version": "v1",
                        "history_count": 2,
                        "rollout_status": "candidate",
                        "candidate_version": "v2",
                        "candidate_baseline_version": "v1",
                        "candidate_episode_budget": 2,
                    }
                ),
                encoding="utf-8",
            )
            versions_path.write_text(
                json.dumps({"version_id": "v1", "snapshot_path": str(baseline_path), "created_at_utc": "2026-03-20T00:00:00+00:00"}) + "\n"
                + json.dumps({"version_id": "v2", "snapshot_path": str(candidate_path), "created_at_utc": "2026-03-20T00:01:00+00:00"}) + "\n",
                encoding="utf-8",
            )
            metrics_path.write_text(
                json.dumps(
                    {
                        "version_metrics": {
                            "v1": {"episode_count": 3, "success_rate": 0.9, "avg_value_score": 0.8, "avg_step_count": 4.0},
                            "v2": {"episode_count": 2, "success_rate": 0.6, "avg_value_score": 0.6, "avg_step_count": 5.5},
                        }
                    }
                ),
                encoding="utf-8",
            )

            from scripts.judge_control_rollout import main as judge_main

            os.environ["CARM_CONTROLS_PATH"] = str(controls_path)
            os.environ["CARM_CONTROL_STATE_PATH"] = str(state_path)
            os.environ["CARM_CONTROL_VERSIONS_PATH"] = str(versions_path)
            os.environ["CARM_CONTROL_HISTORY_DIR"] = str(history_dir)
            os.environ["CARM_CONTROL_METRICS_PATH"] = str(metrics_path)
            os.environ["CARM_ROLLOUT_AUDIT_PATH"] = str(audit_path)
            os.environ["CARM_CONTROL_AUTO_ROLLBACK"] = "1"
            try:
                judge_main()
            finally:
                os.environ.pop("CARM_CONTROLS_PATH", None)
                os.environ.pop("CARM_CONTROL_STATE_PATH", None)
                os.environ.pop("CARM_CONTROL_VERSIONS_PATH", None)
                os.environ.pop("CARM_CONTROL_HISTORY_DIR", None)
                os.environ.pop("CARM_CONTROL_METRICS_PATH", None)
                os.environ.pop("CARM_ROLLOUT_AUDIT_PATH", None)
                os.environ.pop("CARM_CONTROL_AUTO_ROLLBACK", None)

            controls = json.loads(controls_path.read_text(encoding="utf-8"))
            state = json.loads(state_path.read_text(encoding="utf-8"))
            self.assertEqual(controls["policy"]["call_tool_bonus"], 0.0)
            self.assertEqual(state["rollout_status"], "stable")
            self.assertEqual(state["last_rollout_decision"], "rollback")
            self.assertTrue(audit_path.exists())

    def test_run_control_cycle_executes_end_to_end(self) -> None:
        with TemporaryDirectory() as temp_dir:
            data_dir = Path(temp_dir) / "data"
            config_dir = Path(temp_dir) / "configs"
            (data_dir / "experience").mkdir(parents=True, exist_ok=True)
            (data_dir / "review").mkdir(parents=True, exist_ok=True)
            (data_dir / "control").mkdir(parents=True, exist_ok=True)
            config_dir.mkdir(parents=True, exist_ok=True)
            (config_dir / "control_cycle.json").write_text(
                json.dumps(
                    {
                        "prompt_sets": {
                            "default": [
                                "请计算 12 + 30 / 3",
                                "比较 PostgreSQL 和 MySQL 在中小团队里的适用性",
                            ]
                        }
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            (data_dir / "review" / "reviews.jsonl").write_text(
                json.dumps(
                    {
                        "user_input": "x",
                        "success": False,
                        "value_score": 0.2,
                        "issue_tags": ["tool_underuse", "tool_underuse"],
                        "strengths": [],
                        "weaknesses": [],
                        "recommendations": ["raise tool usage"],
                        "target_modules": ["policy"],
                        "evidence": {},
                    },
                    ensure_ascii=False,
                )
                + "\n"
                + json.dumps(
                    {
                        "user_input": "y",
                        "success": False,
                        "value_score": 0.3,
                        "issue_tags": ["tool_underuse"],
                        "strengths": [],
                        "weaknesses": [],
                        "recommendations": ["raise tool usage"],
                        "target_modules": ["policy"],
                        "evidence": {},
                    },
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )

            from scripts.run_control_cycle import main as cycle_main

            current = Path.cwd()
            try:
                os.chdir(temp_dir)
                os.environ["CARM_CONTROL_ROLLOUT_BUDGET"] = "2"
                cycle_main([])
            finally:
                os.environ.pop("CARM_CONTROL_ROLLOUT_BUDGET", None)
                os.chdir(current)

            metrics_path = data_dir / "control" / "control_version_metrics.json"
            state_path = data_dir / "control" / "control_state.json"
            self.assertTrue(metrics_path.exists())
            self.assertTrue(state_path.exists())
            state = json.loads(state_path.read_text(encoding="utf-8"))
            self.assertEqual(state["rollout_status"], "stable")

    def test_run_control_cycle_loads_prompts_from_config(self) -> None:
        with TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "control_cycle.json"
            config_path.write_text(
                json.dumps(
                    {
                        "prompt_sets": {
                            "default": ["a", "b"],
                            "alt": ["x", "y", "z"],
                        }
                    }
                ),
                encoding="utf-8",
            )

            from scripts.run_control_cycle import load_sampling_prompts

            prompts = load_sampling_prompts(config_path=config_path, prompt_set="alt")
            self.assertEqual([item["prompt"] for item in prompts], ["x", "y", "z"])

    def test_run_control_cycle_records_structured_samples(self) -> None:
        with TemporaryDirectory() as temp_dir:
            data_dir = Path(temp_dir) / "data"
            config_dir = Path(temp_dir) / "configs"
            (data_dir / "experience").mkdir(parents=True, exist_ok=True)
            (data_dir / "review").mkdir(parents=True, exist_ok=True)
            (data_dir / "control").mkdir(parents=True, exist_ok=True)
            config_dir.mkdir(parents=True, exist_ok=True)
            (config_dir / "control_cycle.json").write_text(
                json.dumps(
                    {
                        "prompt_sets": {
                            "default": [
                                {
                                    "id": "calc-basic",
                                    "prompt": "请计算 12 + 30 / 3",
                                    "tag": "numeric",
                                    "expected_tool": "calculator",
                                }
                            ]
                        }
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            (data_dir / "review" / "reviews.jsonl").write_text(
                json.dumps(
                    {
                        "user_input": "x",
                        "success": False,
                        "value_score": 0.2,
                        "issue_tags": ["tool_underuse", "tool_underuse"],
                        "strengths": [],
                        "weaknesses": [],
                        "recommendations": ["raise tool usage"],
                        "target_modules": ["policy"],
                        "evidence": {},
                    },
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )

            from scripts.run_control_cycle import main as cycle_main

            current = Path.cwd()
            try:
                os.chdir(temp_dir)
                os.environ["CARM_CONTROL_ROLLOUT_BUDGET"] = "1"
                cycle_main([])
            finally:
                os.environ.pop("CARM_CONTROL_ROLLOUT_BUDGET", None)
                os.chdir(current)

            samples_path = data_dir / "control" / "control_cycle_samples.jsonl"
            self.assertTrue(samples_path.exists())
            records = [json.loads(line) for line in samples_path.read_text(encoding="utf-8").splitlines() if line.strip()]
            self.assertEqual(records[0]["task_id"], "calc-basic")
            self.assertEqual(records[0]["tag"], "numeric")
            self.assertEqual(records[0]["expected_tool"], "calculator")


if __name__ == "__main__":
    unittest.main()
