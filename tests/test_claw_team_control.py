import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from scripts.claw_team_control import doctor, status
from scripts.team_conductor import bootstrap_workspace


class ClawTeamControlTests(unittest.TestCase):
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
            (root / "configs" / "team_cycle.json").write_text('{"team_name":"mustard-claw"}', encoding="utf-8")
            (root / "team" / "AGENTS.md").write_text("# team\n", encoding="utf-8")
            (root / "team" / "CONDUCTOR.md").write_text("# conductor\n", encoding="utf-8")
            (root / "team" / "OBSERVER.md").write_text("# observer\n", encoding="utf-8")
            (root / "team" / "GUARDIAN.md").write_text("# guardian\n", encoding="utf-8")
            (root / "memory" / "MEMORY.md").write_text("# memory\n", encoding="utf-8")

            payload = status(root)
            self.assertEqual(payload["team_name"], "mustard-claw")
            self.assertEqual(payload["proposal_count"], 0)
            self.assertEqual(payload["daily_digest_count"], 0)

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
