import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from scripts.team_conductor import bootstrap_workspace, run_cycle


class TeamConductorTests(unittest.TestCase):
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
                            "real_prompt_eval": {"delta_tool_match_rate": -0.15},
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
            self.assertEqual(result["proposal_count"], 2)

            proposal_paths = [Path(item) for item in result["proposal_paths"]]
            self.assertEqual(len(proposal_paths), 2)
            proposal_text = proposal_paths[1].read_text(encoding="utf-8")
            self.assertIn("needs_human_approval: True", proposal_text)


if __name__ == "__main__":
    unittest.main()
