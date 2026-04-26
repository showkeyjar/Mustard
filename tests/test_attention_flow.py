import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from carm.attention_flow import build_attention_report, project_episode_attention, project_eval_row_attention
from carm.experience import ExperienceStore
from carm.schemas import EpisodeRecord, StepRecord
from scripts.evaluate_attention_flow import write_attention_flow_report
from scripts.export_attention_flow import export_attention_flow


def make_conflict_episode(*, verify: bool = False) -> EpisodeRecord:
    steps = [
        StepRecord(
            step_idx=1,
            action="WRITE_MEM",
            reason="goal",
            score=1.0,
            target_slot="GOAL",
            feature_snapshot={"needs_conflict_detection": 1.0},
            memory_signature={"has_result": False, "has_draft": False},
        ),
        StepRecord(
            step_idx=2,
            action="CALL_TOOL",
            reason="search",
            score=1.0,
            selected_tool="search",
            feature_snapshot={"needs_conflict_detection": 1.0},
            memory_signature={"has_result": False, "has_draft": False},
        ),
        StepRecord(
            step_idx=3,
            action="WRITE_MEM",
            reason="draft",
            score=1.0,
            target_slot="DRAFT",
            feature_snapshot={"needs_conflict_detection": 1.0},
            memory_signature={"has_result": True, "has_draft": False},
        ),
    ]
    if verify:
        steps.append(
            StepRecord(
                step_idx=4,
                action="VERIFY",
                reason="verify",
                score=1.0,
                feature_snapshot={"needs_conflict_detection": 1.0},
                memory_signature={"has_result": True, "has_draft": True},
                state_signature={"verified": "1"},
            )
        )
    steps.append(
        StepRecord(
            step_idx=5 if verify else 4,
            action="ANSWER",
            reason="answer",
            score=1.0,
            feature_snapshot={"needs_conflict_detection": 1.0},
            memory_signature={"has_result": True, "has_draft": True},
            state_signature={"verified": "1"} if verify else {},
        )
    )
    return EpisodeRecord(
        user_input="两个来源对同一个数据库参数给出了相反建议，应该怎么处理冲突？",
        answer="answer",
        summary="summary",
        success=True,
        value_score=0.8,
        steps=steps,
    )


class AttentionFlowTests(unittest.TestCase):
    def test_project_episode_attention_marks_conflict_residuals_and_transitions(self) -> None:
        nodes = project_episode_attention(make_conflict_episode())

        self.assertEqual(nodes[0].focus_target, "goal")
        self.assertTrue(all(node.transition for node in nodes))
        self.assertTrue(any(node.focus_target == "evidence" for node in nodes))
        self.assertTrue(any("conflict_unresolved" in node.residual_pressure for node in nodes))
        self.assertIn("released_with_risky_residuals", nodes[-1].release_condition)
        self.assertIn("focus=release", nodes[-1].model_view)

    def test_attention_report_counts_premature_release_and_resolution(self) -> None:
        risky_nodes = project_episode_attention(make_conflict_episode(), episode_id="risk")
        verified_nodes = project_episode_attention(make_conflict_episode(verify=True), episode_id="verified")

        report = build_attention_report(risky_nodes + verified_nodes)

        self.assertEqual(report["summary"]["episode_count"], 2)
        self.assertGreater(report["summary"]["risky_residual_count"], 0)
        self.assertGreater(report["summary"]["premature_release_rate"], 0.0)
        self.assertGreater(report["summary"]["residual_resolution"], 0.0)
        self.assertIn("conflict_unresolved", report["residual_counts"])

    def test_export_and_evaluate_attention_flow_scripts(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            experience_path = root / "episodes.jsonl"
            flow_path = root / "attention.jsonl"
            report_path = root / "attention_report.json"
            eval_path = root / "eval.json"
            prompt_path = root / "prompts.json"
            ExperienceStore(experience_path).append(make_conflict_episode(verify=True))
            eval_path.write_text(
                json.dumps(
                    {
                        "rows": [
                            {
                                "id": "eval-conflict",
                                "logic_skill": "conflict_detection",
                                "expected_tool": "search",
                                "pretrained_used_tool": "search",
                                "pretrained_actions": ["CALL_TOOL", "WRITE_MEM", "ANSWER"],
                                "pretrained_match": True,
                            }
                        ]
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            prompt_path.write_text(
                json.dumps(
                    {"prompts": [{"id": "eval-conflict", "prompt": "两个来源给出相反建议"}]},
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

            nodes = export_attention_flow(experience_path, flow_path, eval_path, prompt_path)
            report = write_attention_flow_report(flow_path, report_path)

            self.assertTrue(flow_path.exists())
            self.assertTrue(report_path.exists())
            self.assertEqual(len(nodes), 8)
            self.assertEqual(report["summary"]["episode_count"], 2)
            payload = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertIn("focus_counts", payload)

    def test_project_eval_row_attention_uses_prompt_and_actions(self) -> None:
        nodes = project_eval_row_attention(
            {
                "id": "stress",
                "logic_skill": "conflict_detection",
                "expected_tool": "search",
                "pretrained_used_tool": "search",
                "pretrained_actions": ["CALL_TOOL", "WRITE_MEM", "ANSWER"],
                "pretrained_match": True,
            },
            "两份资料结论相反，应该先处理冲突吗？",
        )

        self.assertEqual(nodes[0].episode_id, "eval:stress")
        self.assertEqual(nodes[0].focus_target, "evidence")
        self.assertTrue(any("conflict_unresolved" in node.residual_pressure for node in nodes))
        self.assertEqual(nodes[-1].focus_target, "release")

    def test_project_eval_row_attention_treats_verify_as_resolving_conflict_release(self) -> None:
        nodes = project_eval_row_attention(
            {
                "id": "verified",
                "logic_skill": "conflict_detection",
                "expected_tool": "search",
                "pretrained_used_tool": "search",
                "pretrained_actions": ["CALL_TOOL", "WRITE_MEM", "VERIFY", "ANSWER"],
                "pretrained_match": True,
            },
            "两份资料结论相反，应该先处理冲突吗？",
        )

        self.assertEqual(nodes[-1].focus_target, "release")
        self.assertNotIn("draft_not_verified", nodes[-1].residual_pressure)
        self.assertNotIn("conflict_unresolved", nodes[-1].residual_pressure)
        self.assertEqual(nodes[-1].release_condition, "draft_ready_and_residual_pressure_low")


if __name__ == "__main__":
    unittest.main()
