import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from carm.reasoning_codec import build_pattern_report, encode_eval_row, validate_hard_eval
from scripts.analyze_reasoning_patterns import write_reasoning_pattern_report


class ReasoningCodecTests(unittest.TestCase):
    def test_encode_conflict_case_marks_missing_verify_residual(self) -> None:
        row = {
            "id": "conflict",
            "logic_skill": "conflict_detection",
            "expected_tool": "search",
            "pretrained_used_tool": "search",
            "pretrained_actions": ["CALL_TOOL", "WRITE_MEM", "ANSWER"],
            "pretrained_match": True,
        }

        record = encode_eval_row(row, "两个来源给出相反建议")

        self.assertEqual(record.pattern_id, "conflict_first")
        self.assertIn("conflict_unresolved", record.residual_features)
        self.assertIn("missing_verify", record.residual_features)
        self.assertLess(record.fit_score, 0.9)

    def test_build_pattern_report_summarizes_residuals(self) -> None:
        report = build_pattern_report(
            {
                "rows": [
                    {
                        "id": "mixed",
                        "logic_skill": "tool_selection",
                        "expected_tool": "calculator",
                        "pretrained_used_tool": "calculator",
                        "pretrained_actions": ["CALL_TOOL", "WRITE_MEM", "ANSWER"],
                        "pretrained_match": True,
                    },
                    {
                        "id": "summary",
                        "logic_skill": "result_integration",
                        "expected_tool": "bigmodel_proxy",
                        "pretrained_used_tool": "bigmodel_proxy",
                        "pretrained_actions": ["CALL_TOOL", "WRITE_MEM", "ANSWER"],
                        "pretrained_match": True,
                    },
                ]
            },
            {
                "prompts": [
                    {
                        "id": "mixed",
                        "prompt": "既有代码又问严格做数值计算，你会先走哪类工具？",
                    },
                    {
                        "id": "summary",
                        "prompt": "给负责人一份正式结论摘要",
                    },
                ]
            },
        )

        self.assertEqual(report["summary"]["record_count"], 2)
        self.assertEqual(report["pattern_counts"]["tool_boundary"], 1)
        self.assertIn("boundary_ambiguous", report["residual_counts"])

    def test_write_reasoning_pattern_report_creates_output(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            eval_path = root / "eval.json"
            prompt_path = root / "prompts.json"
            output_path = root / "out.json"
            hard_path = root / "hard.json"
            eval_path.write_text(
                json.dumps(
                    {
                        "rows": [
                            {
                                "id": "cmp",
                                "logic_skill": "comparison",
                                "expected_tool": "search",
                                "pretrained_used_tool": "search",
                                "pretrained_actions": ["CALL_TOOL", "WRITE_MEM", "VERIFY", "ANSWER"],
                                "pretrained_match": True,
                            }
                        ]
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            prompt_path.write_text(
                json.dumps({"prompts": [{"id": "cmp", "prompt": "比较 A 和 B，并指出证据"}]}, ensure_ascii=False),
                encoding="utf-8",
            )
            hard_path.write_text(
                json.dumps(
                    {
                        "cases": [
                            {
                                "id": "cmp",
                                "expected_pattern": "compare_with_evidence",
                                "required_residuals": ["needs_evidence"],
                                "unacceptable_failures": ["tool_mismatch"],
                            }
                        ]
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

            report = write_reasoning_pattern_report(eval_path, prompt_path, output_path, hard_path)

            self.assertTrue(output_path.exists())
            self.assertEqual(report["summary"]["record_count"], 1)
            self.assertEqual(report["hard_eval"]["pass_rate"], 1.0)

    def test_validate_hard_eval_fails_on_unacceptable_residual(self) -> None:
        record = encode_eval_row(
            {
                "id": "conflict",
                "logic_skill": "conflict_detection",
                "expected_tool": "search",
                "pretrained_used_tool": "search",
                "pretrained_actions": ["CALL_TOOL", "WRITE_MEM", "ANSWER"],
                "pretrained_match": True,
            },
            "两个来源给出相反建议",
        )

        result = validate_hard_eval(
            [record],
            {
                "cases": [
                    {
                        "id": "conflict",
                        "expected_pattern": "conflict_first",
                        "required_residuals": ["conflict_unresolved"],
                        "unacceptable_failures": ["missing_verify"],
                    }
                ]
            },
        )

        self.assertEqual(result["pass_rate"], 0.0)
        self.assertEqual(result["failed_case_ids"], ["conflict"])


if __name__ == "__main__":
    unittest.main()
