from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from carm.pretrain_data import (
    apply_review_feedback,
    build_samples_from_experience,
    export_review_pack,
    generate_task_pool,
    import_raw_tasks,
    merge_and_filter_samples,
    save_pretrain_samples,
)
from carm.teacher_distill import distill_prompts_with_teacher, export_teacher_samples
from carm.training import OfflinePretrainer, load_training_config
from scripts.build_real_prompt_candidates import build_candidates, save_candidates
from scripts.current_best import write_current_best
from scripts.evaluate_pretraining import compare_results, evaluate_runner, build_runner_from_state_dir, load_eval_prompts
from scripts.evaluate_real_prompts import evaluate_isolated_prompts


def _timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def run_auto_train(config_path: str | Path = "configs/training.yaml") -> dict[str, object]:
    config = load_training_config(config_path)
    training = config.get("training", {})
    pretraining = training.get("pretraining", {})
    online = training.get("online_evolution", {})
    automation = training.get("automation", {})

    dataset_path = Path(str(pretraining.get("dataset_path", "data/pretrain/pretrain_corpus.jsonl")))
    review_pack_path = Path(str(pretraining.get("review_pack_path", "data/pretrain/review_pack.jsonl")))
    artifact_dir = Path(str(pretraining.get("artifact_dir", "data/pretrain")))
    experience_path = Path(str(pretraining.get("experience_path", "data/experience/episodes.jsonl")))
    real_prompt_candidate_path = Path(str(pretraining.get("real_prompt_candidate_path", "data/eval/real_prompt_candidates.json")))

    count_per_type = int(pretraining.get("count_per_task_type", 24))
    seed = int(pretraining.get("seed", 7))
    min_quality_score = float(pretraining.get("min_quality_score", 0.72))
    max_dataset_samples = int(pretraining.get("max_dataset_samples", 5000))
    auto_sample_from_experience = bool(pretraining.get("auto_sample_from_experience", True))
    min_experience_value_score = float(pretraining.get("min_experience_value_score", 0.72))
    max_experience_samples = int(pretraining.get("max_experience_samples", 300))
    auto_export_real_prompt_candidates = bool(pretraining.get("auto_export_real_prompt_candidates", True))
    max_real_prompt_candidates = int(pretraining.get("max_real_prompt_candidates", 50))
    teacher_distill_enabled = bool(pretraining.get("teacher_distill_enabled", True))
    teacher_distill_limit = int(pretraining.get("teacher_distill_limit", 120))
    teacher_dataset_path = Path(str(pretraining.get("teacher_dataset_path", "data/pretrain/teacher_distill.jsonl")))

    report_dir = Path(str(automation.get("report_dir", "data/train_runs")))
    run_apply_review_feedback = bool(automation.get("run_apply_review_feedback", True))
    run_pretrain_eval = bool(automation.get("run_pretrain_eval", True))
    run_real_prompt_eval = bool(automation.get("run_real_prompt_eval", True))
    pretrain_eval_path = Path(str(automation.get("pretrain_eval_path", "configs/pretrain_eval.json")))
    real_eval_path = Path(str(automation.get("real_eval_path", "configs/real_prompt_eval.json")))

    samples = generate_task_pool(seed=seed, count_per_type=count_per_type)
    import_env = str(automation.get("import_paths_env", "CARM_PRETRAIN_IMPORT_PATHS"))
    extra_paths = [
        Path(item.strip())
        for item in __import__("os").environ.get(import_env, "").split(";")
        if item.strip()
    ]
    if extra_paths:
        samples.extend(import_raw_tasks(extra_paths))
    if auto_sample_from_experience and experience_path.exists():
        experience_samples = build_samples_from_experience(
            experience_path,
            min_value_score=min_experience_value_score,
            limit=max_experience_samples,
        )
        samples.extend(experience_samples)
    else:
        experience_samples = []

    if teacher_distill_enabled:
        teacher_prompts = [sample.user_input for sample in samples]
        teacher_samples = distill_prompts_with_teacher(teacher_prompts, limit=teacher_distill_limit)
        samples.extend(teacher_samples)
        export_teacher_samples(teacher_dataset_path, teacher_samples)
    else:
        teacher_samples = []

    samples = merge_and_filter_samples(samples, min_quality_score=min_quality_score, max_samples=max_dataset_samples)
    save_pretrain_samples(dataset_path, samples)
    export_review_pack(review_pack_path, samples, limit=min(50, len(samples)))

    candidate_count = 0
    if auto_export_real_prompt_candidates and experience_path.exists():
        candidates = build_candidates(
            experience_path,
            min_value_score=min_experience_value_score,
            limit=max_real_prompt_candidates,
        )
        save_candidates(real_prompt_candidate_path, candidates)
        candidate_count = len(candidates)

    merged_samples = samples
    if run_apply_review_feedback and review_pack_path.exists():
        merged_samples = apply_review_feedback(
            dataset_path=dataset_path,
            review_pack_path=review_pack_path,
            min_quality_score=min_quality_score,
            max_samples=max_dataset_samples,
        )

    trainer = OfflinePretrainer(artifact_dir)
    pretrain_result = trainer.run(
        experience_path=experience_path,
        review_path=Path("data/review/reviews.jsonl"),
        signal_log_path=Path(str(online.get("signal_log_path", "data/evolution/signals.jsonl"))),
        dataset_path=dataset_path,
        max_episodes=int(pretraining.get("max_episodes", 500)),
        max_synthetic_samples=int(pretraining.get("max_synthetic_samples", 2000)),
        replay_success_only=bool(pretraining.get("replay_success_only", True)),
        reset_artifacts=bool(pretraining.get("reset_artifacts", True)),
    )

    pretrain_eval_result: dict[str, object] | None = None
    if run_pretrain_eval:
        prompts = load_eval_prompts(pretrain_eval_path)
        baseline_runner = build_runner_from_state_dir(None, report_dir / "_tmp_pretrain_eval_baseline")
        pretrained_runner = build_runner_from_state_dir(artifact_dir, report_dir / "_tmp_pretrain_eval_pretrained")
        try:
            baseline = evaluate_runner(baseline_runner, prompts)
            pretrained = evaluate_runner(pretrained_runner, prompts)
            pretrain_eval_result = compare_results(baseline, pretrained)
        finally:
            for temp_dir in (report_dir / "_tmp_pretrain_eval_baseline", report_dir / "_tmp_pretrain_eval_pretrained"):
                if temp_dir.exists():
                    import shutil
                    shutil.rmtree(temp_dir)

    real_prompt_eval_result: dict[str, object] | None = None
    if run_real_prompt_eval:
        real_prompts = load_eval_prompts(real_eval_path)
        real_prompt_eval_result = evaluate_isolated_prompts(real_prompts, artifact_dir=artifact_dir)

    run_id = _timestamp()
    report_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "run_id": run_id,
        "dataset": {
            "path": str(dataset_path),
            "sample_count": len(merged_samples),
            "review_pack_path": str(review_pack_path),
            "teacher_dataset_path": str(teacher_dataset_path),
            "teacher_sample_count": len(teacher_samples),
            "real_prompt_candidate_path": str(real_prompt_candidate_path),
            "real_prompt_candidate_count": candidate_count,
        },
        "pretraining": {
            "artifact_dir": str(pretrain_result.artifact_dir),
            "episode_count": pretrain_result.episode_count,
            "synthetic_sample_count": pretrain_result.synthetic_sample_count,
            "replayed_step_count": pretrain_result.replayed_step_count,
            "signal_count": pretrain_result.signal_count,
        },
        "evaluation": {
            "pretrain_eval": pretrain_eval_result,
            "real_prompt_eval": real_prompt_eval_result,
        },
    }
    report_path = report_dir / f"auto_train_{run_id}.json"
    latest_path = report_dir / "auto_train_latest.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    latest_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    current_best = write_current_best(Path("."))
    report["current_best_path"] = "artifacts/current_best.json"
    report["current_best_summary"] = current_best.get("summary", {})
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    latest_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main() -> int:
    report = run_auto_train()
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
