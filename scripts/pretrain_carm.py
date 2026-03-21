from __future__ import annotations

from pathlib import Path

from carm.training import OfflinePretrainer, load_training_config
from carm.pretrain_data import generate_task_pool, save_pretrain_samples


def main() -> int:
    config = load_training_config("configs/training.yaml")
    training = config.get("training", {})
    pretraining = training.get("pretraining", {})
    online = training.get("online_evolution", {})
    dataset_path = Path(str(pretraining.get("dataset_path", "data/pretrain/pretrain_corpus.jsonl")))

    if not dataset_path.exists():
        samples = generate_task_pool(
            seed=int(pretraining.get("seed", 7)),
            count_per_type=int(pretraining.get("count_per_task_type", 24)),
        )
        save_pretrain_samples(dataset_path, samples)
        print(f"Bootstrapped low-cost dataset: samples={len(samples)} path={dataset_path}")

    artifact_dir = Path(str(pretraining.get("artifact_dir", "data/pretrain")))
    trainer = OfflinePretrainer(artifact_dir)
    result = trainer.run(
        experience_path=Path("data/experience/episodes.jsonl"),
        review_path=Path("data/review/reviews.jsonl"),
        signal_log_path=Path(str(online.get("signal_log_path", "data/evolution/signals.jsonl"))),
        dataset_path=dataset_path,
        max_episodes=int(pretraining.get("max_episodes", 500)),
        max_synthetic_samples=int(pretraining.get("max_synthetic_samples", 2000)),
        replay_success_only=bool(pretraining.get("replay_success_only", True)),
        reset_artifacts=bool(pretraining.get("reset_artifacts", True)),
    )
    print(
        "Pretraining complete: "
        f"episodes={result.episode_count}, synthetic_samples={result.synthetic_sample_count}, "
        f"replayed_steps={result.replayed_step_count}, signals={result.signal_count}"
    )
    print(f"Artifacts written to {result.artifact_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
