import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from carm.experience import ExperienceStore
from carm.normalize import normalize_draft_payload, normalize_episode_payload


class MigrationTests(unittest.TestCase):
    def test_normalize_legacy_draft_payload(self) -> None:
        legacy = {
            "kind": "draft",
            "claim": "基于外部结果形成初步结论",
            "support": ["result text"],
            "status": "grounded",
        }
        normalized = normalize_draft_payload(legacy)

        self.assertEqual(normalized["summary"], "基于外部结果形成初步结论")
        self.assertEqual(normalized["support_items"], ["result text"])
        self.assertEqual(normalized["confidence_band"], "grounded")

    def test_experience_store_normalizes_legacy_episode(self) -> None:
        with TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "episodes.jsonl"
            legacy_episode = {
                "user_input": "test",
                "answer": "任务: test\n结论: done\n",
                "summary": '{"kind":"draft","claim":"x","support":["y"],"status":"grounded"} | {"kind":"draft","claim":"x","support":["y"],"status":"grounded"}',
                "success": True,
                "value_score": 0.8,
                "steps": [{"step_idx": 1, "action": "ANSWER", "reason": "ok", "score": 1.0}],
            }
            path.write_text(json.dumps(legacy_episode, ensure_ascii=False) + "\n", encoding="utf-8")

            store = ExperienceStore(path)
            recalled = store.recall("test", limit=1)

            self.assertEqual(len(recalled), 1)
            self.assertIn('"confidence_band": "grounded"', recalled[0].summary)
            self.assertEqual(recalled[0].steps[0].target_slot, "")
            self.assertIn("keywords", recalled[0].episode_features)
            self.assertIn("final_action", recalled[0].outcome_signature)
            self.assertIn("state_signature", recalled[0].steps[0].__dict__)
            self.assertIn("memory_signature", recalled[0].steps[0].__dict__)
            self.assertIn("reward_reason", recalled[0].steps[0].__dict__)

    def test_normalize_episode_dedupes_summary(self) -> None:
        payload = {
            "user_input": "x",
            "answer": "a\n",
            "summary": "same | same",
            "success": True,
            "value_score": 0.5,
            "steps": [],
        }
        normalized = normalize_episode_payload(payload)
        self.assertEqual(normalized["summary"], "same")
        self.assertIn("episode_features", normalized)
        self.assertIn("outcome_signature", normalized)
        self.assertIn("state_signature", normalized["steps"][0] if normalized["steps"] else {"state_signature": {}})


if __name__ == "__main__":
    unittest.main()
