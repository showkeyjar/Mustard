import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from scripts.claw_team_control import apply_human_review_session, build_human_review_session, doctor, status
from scripts.team_conductor import bootstrap_workspace


class ClawTeamControlTests(unittest.TestCase):
    def test_build_delivery_summary_reports_lane_and_git_result(self) -> None:
        from scripts.claw_team_control import _build_delivery_summary

        cycle_payload = {
            "delivery_decision": {
                "delivery_lane": "sync_only",
                "reason": "core_changes_detected",
                "should_commit": True,
                "should_push": True,
                "should_open_pr": False,
                "file_groups": {
                    "core": ["scripts/team_conductor.py"],
                    "artifacts": ["team/GITHUB_AUTOMATION.md"],
                    "volatile": ["data/team/role_content_history.jsonl"],
                    "other": [],
                },
            }
        }
        git_delivery = {"committed": True, "pushed": True, "commit_sha": "abc123", "branch": "main"}
        summary = _build_delivery_summary(cycle_payload, git_delivery)
        self.assertEqual(summary["delivery_lane"], "sync_only")
        self.assertEqual(summary["core_count"], 1)
        self.assertEqual(summary["artifact_count"], 1)
        self.assertTrue(summary["git_committed"])
        self.assertEqual(summary["git_branch"], "main")

    def test_auto_sync_git_includes_artifacts_when_requested(self) -> None:
        from scripts.claw_team_control import _paths_for_git_delivery

        decision = {
            "file_groups": {
                "core": ["scripts/team_conductor.py"],
                "artifacts": ["backlog/opportunities/research_latest.md", "team/GITHUB_AUTOMATION.md"],
            }
        }
        paths = _paths_for_git_delivery(decision, include_artifacts=True)
        self.assertIn("scripts/team_conductor.py", paths)
        self.assertIn("backlog/opportunities/research_latest.md", paths)
        self.assertIn("team/GITHUB_AUTOMATION.md", paths)

    @patch("scripts.claw_team_control.push_current_branch", return_value={"pushed": True, "branch": "main"})
    @patch("scripts.claw_team_control.commit_selected_paths", return_value={"committed": True, "commit_sha": "abc123", "paths": ["scripts/team_conductor.py"]})
    def test_run_auto_sync_git_commits_and_pushes_selected_paths(self, commit_mock, push_mock) -> None:
        from scripts.claw_team_control import _auto_sync_git_from_cycle

        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            payload = {
                "digest_path": "memory/daily/2026-03-27.md",
                "team_actions": {},
                "proposal_paths": [],
                "delivery_decision": {
                    "delivery_lane": "sync_only",
                    "should_push": True,
                    "file_groups": {
                        "core": ["scripts/team_conductor.py"],
                        "artifacts": ["backlog/opportunities/research_latest.md"],
                        "volatile": ["data/team/role_content_history.jsonl"],
                    },
                },
            }
            result = _auto_sync_git_from_cycle(root, payload, include_artifacts=False)
            self.assertTrue(result["committed"])
            self.assertTrue(result["pushed"])
            commit_mock.assert_called_once()
            push_mock.assert_called_once()

    def test_deliver_skips_when_delivery_decision_is_not_pr_lane(self) -> None:
        cycle_payload = {
            "team_name": "mustard-claw",
            "delivery_decision": {"delivery_lane": "sync_only", "reason": "core_changes_detected"},
        }
        payload = {
            "cycle": cycle_payload,
            "delivery": {
                "submitted": False,
                "reason": "pr_lane_not_selected",
                "delivery_decision": cycle_payload["delivery_decision"],
            },
        }
        self.assertFalse(payload["delivery"]["submitted"])
        self.assertEqual(payload["delivery"]["reason"], "pr_lane_not_selected")

    def test_doctor_reports_missing_files_before_bootstrap_assets_exist(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            bootstrap_workspace(root)
            payload = doctor(root)
            self.assertFalse(payload["ok"])
            self.assertTrue(payload["missing_files"])

    def test_status_reports_counts_after_bootstrap(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            bootstrap_workspace(root)
            (root / "configs").mkdir(parents=True, exist_ok=True)
            (root / "artifacts").mkdir(parents=True, exist_ok=True)
            (root / "configs" / "team_cycle.json").write_text('{"team_name":"mustard-claw"}', encoding="utf-8")
            (root / "team" / "AGENTS.md").write_text("# team\n", encoding="utf-8")
            (root / "team" / "CONDUCTOR.md").write_text("# conductor\n", encoding="utf-8")
            (root / "team" / "OBSERVER.md").write_text("# observer\n", encoding="utf-8")
            (root / "team" / "GUARDIAN.md").write_text("# guardian\n", encoding="utf-8")
            (root / "memory" / "MEMORY.md").write_text("# memory\n", encoding="utf-8")
            (root / "artifacts" / "current_best.json").write_text(
                json.dumps({"best_run_id": "run-1", "summary": {"real_prompt_match_rate": 0.95}}, ensure_ascii=False),
                encoding="utf-8",
            )

            payload = status(root)
            self.assertEqual(payload["team_name"], "mustard-claw")
            self.assertEqual(payload["proposal_count"], 0)
            self.assertEqual(payload["daily_digest_count"], 0)
            self.assertEqual(payload["current_best"]["best_run_id"], "run-1")
            self.assertEqual(payload["human_review"]["pending_count"], 0)

    def test_build_human_review_session_refreshes_panel_pipeline(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            bootstrap_workspace(root)
            (root / "configs").mkdir(parents=True, exist_ok=True)
            (root / "artifacts").mkdir(parents=True, exist_ok=True)
            (root / "backlog" / "opportunities").mkdir(parents=True, exist_ok=True)
            (root / "data" / "learning").mkdir(parents=True, exist_ok=True)
            (root / "data" / "evolution").mkdir(parents=True, exist_ok=True)
            (root / "data" / "pretrain").mkdir(parents=True, exist_ok=True)
            (root / "configs" / "team_cycle.json").write_text('{"team_name":"mustard-claw"}', encoding="utf-8")
            (root / "configs" / "training.yaml").write_text(
                json.dumps(
                    {
                        "training": {
                            "pretraining": {
                                "dataset_path": "data/pretrain/pretrain_corpus.jsonl",
                                "min_quality_score": 0.72,
                                "max_dataset_samples": 5000,
                            }
                        }
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            (root / "data" / "pretrain" / "pretrain_corpus.jsonl").write_text("", encoding="utf-8")
            (root / "data" / "learning" / "candidate_pretrain_review_pack.jsonl").write_text(
                "\n".join(
                    [
                        json.dumps(
                            {
                                "user_input": "Learning-focus routing stress 学习：样本=stress-learning-focus-evidence-routing-001。 期望工具=search。",
                                "task_type": "fact_check",
                                "logic_skill": "evidence_judgment",
                                "source_type": "learning_intake:learning_focus_stress",
                                "expected_tool": "search",
                                "target_slot": "PLAN",
                                "quality_score": 0.99,
                                "plan_action_items": ["a", "b", "c"],
                                "plan_unknowns": ["u"],
                                "evidence_targets": ["e"],
                                "plan_summary": "p",
                                "draft_summary": "d",
                                "review_status": "pending",
                                "review_note": "",
                                "override_task_type": "",
                                "override_expected_tool": "",
                                "override_target_slot": "",
                                "override_plan_summary": "",
                                "override_action_items": [],
                                "override_unknowns": [],
                                "override_evidence_targets": [],
                                "override_draft_summary": "",
                            },
                            ensure_ascii=False,
                        )
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
                                "id": "stress-learning-focus-evidence-routing-001",
                                "pretrained_match": False,
                                "baseline_match": False,
                                "pretrained_used_tool": "calculator",
                                "baseline_used_tool": "calculator",
                            }
                        ]
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            (root / "artifacts" / "attention_flow_latest.json").write_text(json.dumps({"summary": {}}, ensure_ascii=False), encoding="utf-8")
            (root / "artifacts" / "attention_training_views_latest.json").write_text(json.dumps({"summary": {}}, ensure_ascii=False), encoding="utf-8")

            payload = build_human_review_session(root)

            self.assertEqual(payload["mode"], "human_review_session")
            self.assertEqual(payload["panel_summary"]["total_candidates"], 1)
            self.assertTrue((root / "data" / "learning" / "candidate_pretrain_human_review_sheet.jsonl").exists())

    def test_apply_human_review_session_can_export_import(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "data" / "learning").mkdir(parents=True, exist_ok=True)
            (root / "artifacts").mkdir(parents=True, exist_ok=True)
            (root / "backlog" / "opportunities").mkdir(parents=True, exist_ok=True)
            (root / "data" / "learning" / "candidate_pretrain_review_pack.jsonl").write_text(
                json.dumps(
                    {
                        "user_input": "样本 A",
                        "task_type": "fact_check",
                        "logic_skill": "evidence_judgment",
                        "source_type": "learning_intake:learning_focus_stress",
                        "expected_tool": "search",
                        "target_slot": "PLAN",
                        "quality_score": 0.99,
                        "plan_action_items": ["a", "b", "c"],
                        "plan_unknowns": ["u"],
                        "evidence_targets": ["e"],
                        "plan_summary": "p",
                        "draft_summary": "d",
                        "review_status": "pending",
                        "review_note": "",
                        "override_task_type": "",
                        "override_expected_tool": "",
                        "override_target_slot": "",
                        "override_plan_summary": "",
                        "override_action_items": [],
                        "override_unknowns": [],
                        "override_evidence_targets": [],
                        "override_draft_summary": "",
                    },
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )
            (root / "data" / "learning" / "candidate_pretrain_human_review_sheet.jsonl").write_text(
                json.dumps(
                    {
                        "user_input": "样本 A",
                        "task_type": "fact_check",
                        "logic_skill": "evidence_judgment",
                        "source_type": "learning_intake:learning_focus_stress",
                        "expected_tool": "search",
                        "target_slot": "PLAN",
                        "quality_score": 0.99,
                        "plan_action_items": ["a", "b", "c"],
                        "plan_unknowns": ["u"],
                        "evidence_targets": ["e"],
                        "plan_summary": "p",
                        "draft_summary": "d",
                        "review_status": "pending",
                        "review_note": "",
                        "override_task_type": "",
                        "override_expected_tool": "",
                        "override_target_slot": "",
                        "override_plan_summary": "",
                        "override_action_items": [],
                        "override_unknowns": [],
                        "override_evidence_targets": [],
                        "override_draft_summary": "",
                        "human_review_status": "accept",
                        "human_review_note": "looks good",
                    },
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )

            payload = apply_human_review_session(root, export_import=True)

            self.assertEqual(payload["apply_summary"]["applied_count"], 1)
            self.assertEqual(payload["import_summary"]["approved_count"], 1)
            self.assertTrue((root / "data" / "learning" / "candidate_pretrain_import.jsonl").exists())

    def test_status_reports_preview_human_review_counts(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            bootstrap_workspace(root)
            (root / "configs").mkdir(parents=True, exist_ok=True)
            (root / "artifacts").mkdir(parents=True, exist_ok=True)
            (root / "configs" / "team_cycle.json").write_text('{"team_name":"mustard-claw"}', encoding="utf-8")
            (root / "team" / "AGENTS.md").write_text("# team\n", encoding="utf-8")
            (root / "team" / "CONDUCTOR.md").write_text("# conductor\n", encoding="utf-8")
            (root / "team" / "OBSERVER.md").write_text("# observer\n", encoding="utf-8")
            (root / "team" / "GUARDIAN.md").write_text("# guardian\n", encoding="utf-8")
            (root / "memory" / "MEMORY.md").write_text("# memory\n", encoding="utf-8")
            (root / "data" / "learning").mkdir(parents=True, exist_ok=True)
            (root / "artifacts" / "learning_intake_human_review_panel_latest.json").write_text(
                json.dumps({"summary": {"total_candidates": 4, "recommend_accept": 1, "recommend_edit": 2, "recommend_defer": 1}}, ensure_ascii=False),
                encoding="utf-8",
            )
            (root / "artifacts" / "learning_intake_human_review_preview_latest.json").write_text(
                json.dumps({"summary": {"ready_to_apply_count": 2, "blank_decision_count": 2, "invalid_decision_count": 0}}, ensure_ascii=False),
                encoding="utf-8",
            )

            payload = status(root)

            self.assertEqual(payload["human_review"]["ready_to_apply_count"], 2)
            self.assertEqual(payload["human_review"]["blank_decision_count"], 2)

    def test_status_reports_preview_next_action(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            bootstrap_workspace(root)
            (root / "configs").mkdir(parents=True, exist_ok=True)
            (root / "artifacts").mkdir(parents=True, exist_ok=True)
            (root / "configs" / "team_cycle.json").write_text('{"team_name":"mustard-claw"}', encoding="utf-8")
            (root / "team" / "AGENTS.md").write_text("# team\n", encoding="utf-8")
            (root / "team" / "CONDUCTOR.md").write_text("# conductor\n", encoding="utf-8")
            (root / "team" / "OBSERVER.md").write_text("# observer\n", encoding="utf-8")
            (root / "team" / "GUARDIAN.md").write_text("# guardian\n", encoding="utf-8")
            (root / "memory" / "MEMORY.md").write_text("# memory\n", encoding="utf-8")
            (root / "artifacts" / "learning_intake_human_review_preview_latest.json").write_text(
                json.dumps(
                    {
                        "summary": {"ready_to_apply_count": 3, "blank_decision_count": 0, "invalid_decision_count": 0},
                        "next_action": {
                            "state": "ready_to_apply",
                            "command": "python -m scripts.claw_team_control apply-human-review --export-import",
                        },
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

            payload = status(root)

            self.assertEqual(payload["human_review"]["next_action_state"], "ready_to_apply")
            self.assertIn("apply-human-review", payload["human_review"]["next_action_command"])

    def test_status_reports_human_review_draft_path(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            bootstrap_workspace(root)
            (root / "configs").mkdir(parents=True, exist_ok=True)
            (root / "artifacts").mkdir(parents=True, exist_ok=True)
            (root / "configs" / "team_cycle.json").write_text('{"team_name":"mustard-claw"}', encoding="utf-8")
            (root / "team" / "AGENTS.md").write_text("# team\n", encoding="utf-8")
            (root / "team" / "CONDUCTOR.md").write_text("# conductor\n", encoding="utf-8")
            (root / "team" / "OBSERVER.md").write_text("# observer\n", encoding="utf-8")
            (root / "team" / "GUARDIAN.md").write_text("# guardian\n", encoding="utf-8")
            (root / "memory" / "MEMORY.md").write_text("# memory\n", encoding="utf-8")
            (root / "artifacts" / "learning_intake_human_review_draft_latest.json").write_text(
                json.dumps({"summary": {"draft_sheet_path": "data/learning/candidate_pretrain_human_review_sheet.draft.jsonl"}}, ensure_ascii=False),
                encoding="utf-8",
            )

            payload = status(root)

            self.assertIn("candidate_pretrain_human_review_sheet.draft.jsonl", payload["human_review"]["draft_sheet_path"])

    def test_status_reports_human_review_draft_sync_count(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            bootstrap_workspace(root)
            (root / "configs").mkdir(parents=True, exist_ok=True)
            (root / "artifacts").mkdir(parents=True, exist_ok=True)
            (root / "configs" / "team_cycle.json").write_text('{"team_name":"mustard-claw"}', encoding="utf-8")
            (root / "team" / "AGENTS.md").write_text("# team\n", encoding="utf-8")
            (root / "team" / "CONDUCTOR.md").write_text("# conductor\n", encoding="utf-8")
            (root / "team" / "OBSERVER.md").write_text("# observer\n", encoding="utf-8")
            (root / "team" / "GUARDIAN.md").write_text("# guardian\n", encoding="utf-8")
            (root / "memory" / "MEMORY.md").write_text("# memory\n", encoding="utf-8")
            (root / "artifacts" / "learning_intake_human_review_draft_sync_latest.json").write_text(
                json.dumps({"synced_count": 3}, ensure_ascii=False),
                encoding="utf-8",
            )

            payload = status(root)

            self.assertEqual(payload["human_review"]["draft_sync_count"], 3)

    @patch("scripts.claw_team_control.github_doctor", return_value={"ok": False, "token_present": False})
    @patch("scripts.claw_team_control.run_cycle", return_value={"team_name": "mustard-claw", "proposal_count": 0})
    def test_deliver_mode_can_stop_before_submit_when_github_not_ready(self, run_cycle_mock, doctor_mock) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            payload = {
                "cycle": run_cycle_mock.return_value,
                "delivery": {
                    "submitted": False,
                    "reason": "github_doctor_failed",
                    "github": doctor_mock.return_value,
                },
            }
            self.assertFalse(payload["delivery"]["submitted"])
            self.assertEqual(payload["delivery"]["reason"], "github_doctor_failed")


if __name__ == "__main__":
    unittest.main()
