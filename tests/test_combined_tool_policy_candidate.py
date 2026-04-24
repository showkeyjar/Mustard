import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from scripts.evaluate_combined_tool_policy_candidate import evaluate_candidate


class CombinedToolPolicyCandidateTests(unittest.TestCase):
    def test_combined_candidate_passes_real_prompt_and_hard_eval(self) -> None:
        with TemporaryDirectory() as temp_dir:
            report = evaluate_candidate(Path(temp_dir) / "candidate.json")

        self.assertEqual(report["decision"], "candidate_pass")
        self.assertEqual(report["real_prompt_summary"]["pretrained_match_rate"], 1.0)
        self.assertEqual(report["hard_eval_summary"]["pass_rate"], 1.0)


if __name__ == "__main__":
    unittest.main()
