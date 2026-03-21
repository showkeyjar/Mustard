from __future__ import annotations

from pathlib import Path

from carm.pretrain_data import apply_review_feedback
from carm.training import load_training_config


def main() -> int:
    config = load_training_config("configs/training.yaml")
    pretraining = config.get("training", {}).get("pretraining", {})
    dataset_path = Path(str(pretraining.get("dataset_path", "data/pretrain/pretrain_corpus.jsonl")))
    review_pack_path = Path(str(pretraining.get("review_pack_path", "data/pretrain/review_pack.jsonl")))
    min_quality_score = float(pretraining.get("min_quality_score", 0.72))
    max_samples = int(pretraining.get("max_dataset_samples", 5000))

    merged = apply_review_feedback(
        dataset_path=dataset_path,
        review_pack_path=review_pack_path,
        min_quality_score=min_quality_score,
        max_samples=max_samples,
    )
    print(f"Applied review feedback: samples={len(merged)} dataset={dataset_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
