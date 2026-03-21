import unittest
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from carm.pretrain_data import (
    apply_review_feedback,
    dedupe_samples,
    export_review_pack,
    generate_task_pool,
    import_raw_tasks,
    load_pretrain_samples,
    merge_and_filter_samples,
    sample_to_episode,
    save_pretrain_samples,
)
from carm.teacher_distill import distill_prompt_with_teacher


class PretrainDataTests(unittest.TestCase):
    def test_generate_task_pool_produces_structured_samples(self) -> None:
        samples = generate_task_pool(seed=1, count_per_type=3)
        self.assertEqual(len(samples), 24)
        self.assertTrue(all(sample.expected_tool for sample in samples))
        self.assertTrue(all(sample.plan_action_items for sample in samples))
        self.assertTrue(all(sample.evidence_targets for sample in samples))
        self.assertTrue(all(sample.logic_skill for sample in samples))
        self.assertTrue(any(sample.source_type.startswith("logic_hard_") for sample in samples))

    def test_samples_roundtrip_and_convert_to_episode(self) -> None:
        with TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "pretrain.jsonl"
            samples = generate_task_pool(seed=2, count_per_type=1)
            save_pretrain_samples(path, samples)
            loaded = load_pretrain_samples(path)
            self.assertEqual(len(loaded), 8)

            episode = sample_to_episode(loaded[0])
            self.assertTrue(episode.success)
            self.assertTrue(episode.steps)
            self.assertIn("expected_tool", episode.episode_features)

    def test_import_merge_and_review_pack(self) -> None:
        with TemporaryDirectory() as temp_dir:
            raw_path = Path(temp_dir) / "public_tasks.jsonl"
            raw_path.write_text(
                json.dumps({"prompt": "比较 PostgreSQL 和 MySQL 在中小团队里的适用性"}, ensure_ascii=False)
                + "\n"
                + json.dumps({"question": "请计算 12 + 30 / 3"}, ensure_ascii=False)
                + "\n"
                + json.dumps({"instruction": "比较 PostgreSQL 和 MySQL 在中小团队里的适用性"}, ensure_ascii=False)
                + "\n",
                encoding="utf-8",
            )
            imported = import_raw_tasks([raw_path])
            deduped = dedupe_samples(imported)
            merged = merge_and_filter_samples(deduped, min_quality_score=0.7, max_samples=10)
            review_pack = Path(temp_dir) / "review_pack.jsonl"
            export_review_pack(review_pack, merged, limit=10)

            self.assertEqual(len(deduped), 2)
            self.assertTrue(all(sample.quality_score >= 0.7 for sample in merged))
            self.assertTrue(review_pack.exists())

    def test_apply_review_feedback_updates_dataset(self) -> None:
        with TemporaryDirectory() as temp_dir:
            dataset_path = Path(temp_dir) / "pretrain.jsonl"
            review_pack_path = Path(temp_dir) / "review_pack.jsonl"
            samples = generate_task_pool(seed=4, count_per_type=1)
            save_pretrain_samples(dataset_path, samples)
            export_review_pack(review_pack_path, samples, limit=2)

            lines = [json.loads(line) for line in review_pack_path.read_text(encoding="utf-8").splitlines() if line.strip()]
            lines[0]["review_status"] = "edit"
            lines[0]["override_expected_tool"] = "code_executor"
            lines[0]["override_target_slot"] = "PLAN"
            lines[0]["override_action_items"] = ["先定位报错", "再构造复现", "最后验证修复"]
            review_pack_path.write_text("\n".join(json.dumps(item, ensure_ascii=False) for item in lines) + "\n", encoding="utf-8")

            merged = apply_review_feedback(dataset_path, review_pack_path)
            self.assertTrue(any(sample.expected_tool == "code_executor" for sample in merged))

    def test_teacher_distill_generates_structured_sample(self) -> None:
        sample = distill_prompt_with_teacher("我们团队 9 个人，某 SaaS 每席位 129 元/月，如果按年预算估算请按 129 * 9 * 12 计算。")

        self.assertEqual(sample.expected_tool, "calculator")
        self.assertEqual(sample.target_slot, "HYP")
        self.assertTrue(sample.plan_action_items)
        self.assertEqual(sample.source_type, "teacher_distill")


if __name__ == "__main__":
    unittest.main()
