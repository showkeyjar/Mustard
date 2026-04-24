import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from scripts.evaluate_comparison_search_candidate import evaluate_candidate


class ComparisonSearchCandidateTests(unittest.TestCase):
    def test_candidate_prefers_search_for_comparison_without_hurting_management_summary(self) -> None:
        with TemporaryDirectory() as temp_dir:
            report = evaluate_candidate(Path(temp_dir) / "candidate.json")

        self.assertEqual(report["decision"], "candidate_pass")
        self.assertEqual(report["candidate_tool"], "search")
        self.assertEqual(report["guard_summary"]["candidate_match_rate"], 1.0)


if __name__ == "__main__":
    unittest.main()
