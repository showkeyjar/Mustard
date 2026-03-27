import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from scripts.claw_team_github import (
    automerge_pr,
    build_review_body,
    can_automerge,
    choose_review_event,
    commit_selected_paths,
    doctor,
    parse_owner_repo,
    push_current_branch,
)


class ClawTeamGitHubTests(unittest.TestCase):
    @patch("scripts.claw_team_github._git_output", side_effect=["scripts/team_conductor.py\nconfigs/real_prompt_eval.json", "abc123"])
    @patch("scripts.claw_team_github._run")
    def test_commit_selected_paths_stages_only_requested_paths(self, run_mock, git_output_mock) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "scripts").mkdir(parents=True, exist_ok=True)
            (root / "configs").mkdir(parents=True, exist_ok=True)
            (root / "scripts" / "team_conductor.py").write_text("x", encoding="utf-8")
            (root / "configs" / "real_prompt_eval.json").write_text("{}", encoding="utf-8")

            payload = commit_selected_paths(
                root,
                ["scripts/team_conductor.py", "configs/real_prompt_eval.json"],
                "test commit",
            )

            self.assertTrue(payload["committed"])
            rendered_calls = [repr(call) for call in run_mock.call_args_list]
            add_call_found = any("git', 'add', '--'" in item and "scripts/team_conductor.py" in item and "configs/real_prompt_eval.json" in item for item in rendered_calls)
            commit_call_found = any("git', 'commit', '-m', 'test commit'" in item for item in rendered_calls)
            self.assertTrue(add_call_found, rendered_calls)
            self.assertTrue(commit_call_found, rendered_calls)

    @patch("scripts.claw_team_github.push_branch")
    @patch("scripts.claw_team_github.get_current_branch", return_value="main")
    def test_push_current_branch_pushes_detected_branch(self, branch_mock, push_mock) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            payload = push_current_branch(root)
            self.assertTrue(payload["pushed"])
            self.assertEqual(payload["branch"], "main")
            push_mock.assert_called_once_with(root, "main")

    def test_parse_owner_repo_supports_https_remote(self) -> None:
        owner, repo = parse_owner_repo("https://github.com/showkeyjar/Mustard.git")
        self.assertEqual(owner, "showkeyjar")
        self.assertEqual(repo, "Mustard")

    def test_choose_review_event_requests_changes_on_failed_checks(self) -> None:
        pr = {"number": 12, "title": "x", "draft": False, "mergeable": True}
        event, reason = choose_review_event(
            pr,
            [{"command": "python -m unittest", "ok": False, "stderr": "boom", "stdout": "", "returncode": 1}],
            "auto",
        )
        self.assertEqual(event, "REQUEST_CHANGES")
        self.assertIn("failed", reason)

    def test_build_review_body_contains_validation_summary(self) -> None:
        body = build_review_body(
            {"number": 12, "title": "Improve team flow"},
            "APPROVE",
            "all configured checks passed",
            [{"command": "python -m unittest", "ok": True, "stderr": "", "stdout": "", "returncode": 0}],
        )
        self.assertIn("Verdict: APPROVE", body)
        self.assertIn("Validation summary:", body)

    def test_can_automerge_accepts_low_risk_ready_pr(self) -> None:
        ok, reasons = can_automerge(
            {
                "draft": False,
                "mergeable": True,
                "labels": [{"name": "claw-automerge"}],
                "head": {"repo": {"full_name": "showkeyjar/Mustard"}},
                "base": {"repo": {"full_name": "showkeyjar/Mustard"}},
            },
            reviews=[{"state": "APPROVED"}],
            status_payload={"state": "success"},
            auto_merge_config={
                "enabled": True,
                "same_repo_only": True,
                "required_label": "claw-automerge",
                "require_approved_review": True,
                "require_clean_status": True,
            },
        )
        self.assertTrue(ok)
        self.assertEqual(reasons, [])

    def test_can_automerge_blocks_missing_label(self) -> None:
        ok, reasons = can_automerge(
            {
                "draft": False,
                "mergeable": True,
                "labels": [],
                "head": {"repo": {"full_name": "showkeyjar/Mustard"}},
                "base": {"repo": {"full_name": "showkeyjar/Mustard"}},
            },
            reviews=[{"state": "APPROVED"}],
            status_payload={"state": "success"},
            auto_merge_config={
                "enabled": True,
                "same_repo_only": True,
                "required_label": "claw-automerge",
                "require_approved_review": True,
                "require_clean_status": True,
            },
        )
        self.assertFalse(ok)
        self.assertIn("required_label_missing", reasons)

    @patch("scripts.claw_team_github.merge_pull_request", return_value={"merged": True, "sha": "abc123", "message": "merged"})
    @patch("scripts.claw_team_github.get_commit_status", return_value={"state": "success"})
    @patch("scripts.claw_team_github.list_reviews", return_value=[{"state": "APPROVED"}])
    @patch(
        "scripts.claw_team_github.get_pull_request",
        return_value={
            "number": 12,
            "title": "Low risk update",
            "draft": False,
            "mergeable": True,
            "labels": [{"name": "claw-automerge"}],
            "head": {"sha": "abc123", "repo": {"full_name": "showkeyjar/Mustard"}},
            "base": {"repo": {"full_name": "showkeyjar/Mustard"}},
        },
    )
    def test_automerge_pr_merges_when_policy_allows(
        self,
        pr_mock,
        reviews_mock,
        status_mock,
        merge_mock,
    ) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "configs").mkdir(parents=True, exist_ok=True)
            (root / "configs" / "team_github.json").write_text(
                '{"auto_merge":{"enabled":true,"same_repo_only":true,"required_label":"claw-automerge","require_approved_review":true,"require_clean_status":true,"merge_method":"squash"}}',
                encoding="utf-8",
            )
            payload = automerge_pr(root, pr_number=12, merge_method="squash")
            self.assertTrue(payload["merged"])
            self.assertEqual(payload["merge_method"], "squash")

    @patch("scripts.claw_team_github._token", return_value="token")
    @patch("scripts.claw_team_github.get_actions_workflow_permissions", return_value={"default_workflow_permissions": "write", "can_approve_pull_request_reviews": True})
    @patch("scripts.claw_team_github.get_actions_permissions", return_value={"enabled": True})
    @patch("scripts.claw_team_github.list_repository_labels", return_value=[{"name": "claw-automerge"}])
    @patch("scripts.claw_team_github.get_repository", return_value={"default_branch": "main"})
    @patch("scripts.claw_team_github.get_worktree_status", return_value="")
    @patch("scripts.claw_team_github.get_current_branch", return_value="main")
    @patch("scripts.claw_team_github.parse_owner_repo", return_value=("showkeyjar", "Mustard"))
    @patch("scripts.claw_team_github.get_origin_url", return_value="https://github.com/showkeyjar/Mustard.git")
    @patch("scripts.claw_team_github._git_output", return_value="true")
    def test_doctor_reports_ready_repo(
        self,
        git_output_mock,
        origin_mock,
        parse_mock,
        branch_mock,
        status_mock,
        get_repo_mock,
        list_labels_mock,
        actions_permissions_mock,
        workflow_permissions_mock,
        token_mock,
    ) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "configs").mkdir(parents=True, exist_ok=True)
            (root / "configs" / "team_github.json").write_text(
                '{"auto_merge":{"required_label":"claw-automerge"}}',
                encoding="utf-8",
            )
            payload = doctor(root)
            self.assertTrue(payload["ok"])
            self.assertTrue(payload["token_present"])
            self.assertEqual(payload["repo"], "Mustard")
            self.assertTrue(payload["required_label_present"])
            self.assertEqual(payload["default_branch"], "main")


if __name__ == "__main__":
    unittest.main()
