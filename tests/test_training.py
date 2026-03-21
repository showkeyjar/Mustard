import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from carm.experience import ExperienceStore
from carm.pretrain_data import generate_task_pool, save_pretrain_samples
from carm.review import ReviewStore
from carm.schemas import EpisodeRecord, ReviewRecord, StepRecord
from carm.training import OfflinePretrainer


class TrainingTests(unittest.TestCase):
    def test_pretrainer_replays_episodes_and_writes_manifest(self) -> None:
        with TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            experience_path = base / "episodes.jsonl"
            review_path = base / "reviews.jsonl"
            signal_path = base / "signals.jsonl"
            dataset_path = base / "pretrain.jsonl"
            artifact_dir = base / "pretrain"

            ExperienceStore(experience_path).append(
                EpisodeRecord(
                    user_input="数据库选型",
                    answer="answer",
                    summary="summary",
                    success=True,
                    value_score=0.9,
                    steps=[
                        StepRecord(
                            step_idx=1,
                            action="CALL_TOOL",
                            reason="use search",
                            score=1.0,
                            feature_snapshot={"bias": 1.0},
                            user_input="数据库选型",
                            selected_tool="search",
                            reward=1.0,
                            high_value=True,
                        )
                    ],
                )
            )
            ReviewStore(review_path).append(
                ReviewRecord(
                    user_input="数据库选型",
                    success=True,
                    value_score=0.9,
                    issue_tags=["stable_path"],
                )
            )
            signal_path.write_text(
                json.dumps(
                    {
                        "source": "test",
                        "query": "数据库选型",
                        "preferred_tool": "search",
                        "reward": 1.0,
                    },
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )
            save_pretrain_samples(dataset_path, generate_task_pool(seed=3, count_per_type=1))

            result = OfflinePretrainer(artifact_dir).run(experience_path, review_path, signal_path, dataset_path=dataset_path)

            self.assertEqual(result.episode_count, 1)
            self.assertEqual(result.synthetic_sample_count, 8)
            self.assertEqual(result.signal_count, 1)
            self.assertTrue((artifact_dir / "manifest.json").exists())
            self.assertTrue((artifact_dir / "policy_state.json").exists())
            self.assertTrue((artifact_dir / "core_state.json").exists())

    def test_pretrainer_resets_old_artifacts_by_default(self) -> None:
        with TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            artifact_dir = base / "pretrain"
            artifact_dir.mkdir(parents=True, exist_ok=True)
            stale = artifact_dir / "signals.jsonl"
            stale.write_text("stale\n", encoding="utf-8")
            result = OfflinePretrainer(artifact_dir).run(
                experience_path=base / "episodes.jsonl",
                review_path=base / "reviews.jsonl",
                dataset_path=base / "missing.jsonl",
            )
            self.assertEqual(result.episode_count, 0)
            self.assertFalse(stale.exists())


if __name__ == "__main__":
    unittest.main()
