import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from carm.experience import ExperienceStore
from carm.schemas import EpisodeRecord, StepRecord


class ExperienceStoreTests(unittest.TestCase):
    def test_recall_returns_matching_episode(self) -> None:
        with TemporaryDirectory() as temp_dir:
            store = ExperienceStore(Path(temp_dir) / "episodes.jsonl")
            store.append(
                EpisodeRecord(
                    user_input="比较 PostgreSQL 和 MySQL",
                    answer="answer",
                    summary="比较维度 成本 性能",
                    success=True,
                    value_score=0.8,
                    steps=[StepRecord(step_idx=1, action="CALL_TOOL", reason="x", score=1.0)],
                )
            )

            recalled = store.recall("比较 PostgreSQL", limit=1)
            self.assertEqual(len(recalled), 1)
            self.assertIn("PostgreSQL", recalled[0].user_input)


if __name__ == "__main__":
    unittest.main()
