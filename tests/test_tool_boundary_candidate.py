import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from scripts.evaluate_tool_boundary_candidate import evaluate_candidate


class ToolBoundaryCandidateTests(unittest.TestCase):
    def test_candidate_prefers_calculator_for_mixed_numeric_code_case(self) -> None:
        with TemporaryDirectory() as temp_dir:
            report = evaluate_candidate(Path(temp_dir) / "candidate.json")

        self.assertEqual(report["decision"], "candidate_pass")
        self.assertEqual(report["candidate_tool"], "calculator")
        self.assertEqual(report["guard_summary"]["candidate_match_rate"], 1.0)


if __name__ == "__main__":
    unittest.main()
