import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from carm.experience import ExperienceStore
from carm.schemas import EpisodeRecord
from scripts.build_real_prompt_candidates import build_candidates, save_candidates


class RealPromptCandidateTests(unittest.TestCase):
    def test_build_candidates_filters_and_dedupes_successful_prompts(self) -> None:
        with TemporaryDirectory() as temp_dir:
            experience_path = Path(temp_dir) / "episodes.jsonl"
            store = ExperienceStore(experience_path)
            store.append(
                EpisodeRecord(
                    user_input="我们团队 9 个人，某 SaaS 每席位 129 元/月，如果按年预算估算请按 129 * 9 * 12 计算。",
                    answer="ok",
                    summary="summary",
                    success=True,
                    value_score=0.91,
                    episode_features={"used_tool": "calculator", "action_sequence": ["CALL_TOOL", "ANSWER"]},
                )
            )
            store.append(
                EpisodeRecord(
                    user_input="我们团队 9 个人，某 SaaS 每席位 129 元/月，如果按年预算估算请按 129 * 9 * 12 计算。",
                    answer="ok",
                    summary="summary",
                    success=True,
                    value_score=0.7,
                    episode_features={"used_tool": "search", "action_sequence": ["CALL_TOOL", "ANSWER"]},
                )
            )
            store.append(
                EpisodeRecord(
                    user_input="请整理成正式结论",
                    answer="ok",
                    summary="summary",
                    success=False,
                    value_score=0.95,
                    episode_features={"used_tool": "bigmodel_proxy", "action_sequence": ["CALL_BIGMODEL", "ANSWER"]},
                )
            )

            candidates = build_candidates(experience_path, min_value_score=0.65, limit=10)

            self.assertEqual(len(candidates), 1)
            self.assertEqual(candidates[0]["expected_tool"], "calculator")
            self.assertEqual(candidates[0]["logic_skill"], "tool_selection")

    def test_save_candidates_writes_prompt_payload(self) -> None:
        with TemporaryDirectory() as temp_dir:
            target = Path(temp_dir) / "candidates.json"
            save_candidates(target, [{"id": "x", "prompt": "请计算 1 + 2", "expected_tool": "calculator", "logic_skill": "tool_selection"}])

            payload = json.loads(target.read_text(encoding="utf-8"))
            self.assertIn("prompts", payload)
            self.assertEqual(payload["prompts"][0]["expected_tool"], "calculator")


if __name__ == "__main__":
    unittest.main()
