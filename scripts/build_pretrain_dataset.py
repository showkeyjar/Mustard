from __future__ import annotations

import os
from pathlib import Path

from carm.pretrain_data import (
    build_samples_from_experience,
    export_review_pack,
    generate_task_pool,
    import_raw_tasks,
    merge_and_filter_samples,
    save_pretrain_samples,
)
from carm.training import load_training_config
from scripts.build_real_prompt_candidates import build_candidates, save_candidates


def main() -> int:
    config = load_training_config("configs/training.yaml")
    pretraining = config.get("training", {}).get("pretraining", {})
    dataset_path = Path(str(pretraining.get("dataset_path", "data/pretrain/pretrain_corpus.jsonl")))
    review_pack_path = Path(str(pretraining.get("review_pack_path", "data/pretrain/review_pack.jsonl")))
    count_per_type = int(pretraining.get("count_per_task_type", 24))
    seed = int(pretraining.get("seed", 7))
    min_quality_score = float(pretraining.get("min_quality_score", 0.72))
    max_samples = int(pretraining.get("max_dataset_samples", 5000))
    auto_sample_from_experience = bool(pretraining.get("auto_sample_from_experience", True))
    experience_path = Path(str(pretraining.get("experience_path", "data/experience/episodes.jsonl")))
    min_experience_value_score = float(pretraining.get("min_experience_value_score", 0.72))
    max_experience_samples = int(pretraining.get("max_experience_samples", 300))
    real_prompt_candidate_path = Path(str(pretraining.get("real_prompt_candidate_path", "data/eval/real_prompt_candidates.json")))
    auto_export_real_prompt_candidates = bool(pretraining.get("auto_export_real_prompt_candidates", True))
    max_real_prompt_candidates = int(pretraining.get("max_real_prompt_candidates", 50))

    samples = generate_task_pool(seed=seed, count_per_type=count_per_type)
    extra_paths = [
        Path(item.strip())
        for item in os.environ.get("CARM_PRETRAIN_IMPORT_PATHS", "").split(";")
        if item.strip()
    ]
    if extra_paths:
        samples.extend(import_raw_tasks(extra_paths))
    if auto_sample_from_experience and experience_path.exists():
        samples.extend(
            build_samples_from_experience(
                experience_path,
                min_value_score=min_experience_value_score,
                limit=max_experience_samples,
            )
        )
    samples = merge_and_filter_samples(samples, min_quality_score=min_quality_score, max_samples=max_samples)
    save_pretrain_samples(dataset_path, samples)
    export_review_pack(review_pack_path, samples, limit=min(50, len(samples)))
    if auto_export_real_prompt_candidates and experience_path.exists():
        candidates = build_candidates(
            experience_path,
            min_value_score=min_experience_value_score,
            limit=max_real_prompt_candidates,
        )
        save_candidates(real_prompt_candidate_path, candidates)
    print(f"Built pretraining dataset: samples={len(samples)} path={dataset_path}")
    print(f"Review pack written to {review_pack_path}")
    if auto_export_real_prompt_candidates:
        print(f"Real prompt candidates written to {real_prompt_candidate_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
