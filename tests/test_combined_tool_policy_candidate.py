import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from scripts.evaluate_combined_tool_policy_candidate import evaluate_candidate


class CombinedToolPolicyCandidateTests(unittest.TestCase):
    def test_combined_candidate_passes_regression_subset_and_hard_eval(self) -> None:
        with TemporaryDirectory() as temp_dir:
            report = evaluate_candidate(Path(temp_dir) / "candidate.json")

        # The candidate gate must keep the stable regression subset (cases the
        # current policy already routes correctly) at 100%, and the canonical
        # hard-logic set at 100%. The full 63-case coverage set is allowed to
        # contain known-hard cases (BFCL multi-entity parallel constructs) that
        # are model upper-bound limits, not routing regressions.
        self.assertEqual(report["decision"], "candidate_pass")
        self.assertEqual(report["regression_summary"]["pretrained_match_rate"], 1.0)
        self.assertEqual(report["hard_eval_summary"]["pass_rate"], 1.0)


if __name__ == "__main__":
    unittest.main()
