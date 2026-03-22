import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from scripts.claw_team_control import doctor, status
from scripts.team_conductor import bootstrap_workspace


class ClawTeamControlTests(unittest.TestCase):
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
