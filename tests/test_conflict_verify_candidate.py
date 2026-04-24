import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from scripts.evaluate_conflict_verify_candidate import evaluate_candidate


class ConflictVerifyCandidateTests(unittest.TestCase):
    def test_candidate_report_requires_verify_before_answer(self) -> None:
        with TemporaryDirectory() as temp_dir:
            report = evaluate_candidate(Path(temp_dir) / "candidate.json")

        self.assertEqual(report["decision"], "candidate_pass")
        self.assertTrue(report["candidate_has_verify"])
        self.assertTrue(report["candidate_verify_before_answer"])


if __name__ == "__main__":
    unittest.main()
