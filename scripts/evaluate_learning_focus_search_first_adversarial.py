from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory

from carm.training import OfflinePretrainer, load_training_config
from scripts.build_learning_focus_search_first_adversarial_eval import (
    build_learning_focus_search_first_adversarial_eval,
)
from scripts.evaluate_pretraining import load_eval_prompts
from scripts.evaluate_real_prompts import evaluate_isolated_prompts


def _write_report(path: Path, payload: dict[str, object]) -> None:
    current_summary = payload.get("current_summary", {})
    shadow_summary = payload.get("shadow_summary", {})
    delta = payload.get("delta", {})
    lines = [
        "# Learning Focus Search-First Adversarial Eval",
        "",
        "- mode: controlled_search_first_adversarial_eval",
        "- default_runtime_changed: false",
        "- default_training_admission_changed: false",
        f"- target_failure_count: {int(payload.get('target_failure_count', 0) or 0)}",
        f"- prompt_count: {int(current_summary.get('prompt_count', 0) or 0)}",
        f"- current_pretrained_match_rate: {float(current_summary.get('pretrained_match_rate', 0.0) or 0.0):.4f}",
        f"- shadow_pretrained_match_rate: {float(shadow_summary.get('pretrained_match_rate', 0.0) or 0.0):.4f}",
        f"- delta_pretrained_match_rate: {float(delta.get('pretrained_match_rate', 0.0) or 0.0):+.4f}",
        "",
        "## Target IDs",
        "",
        f"- {', '.join(str(item) for item in payload.get('target_ids', []))}",
        "",
        "## Notes",
        "",
        "- 这是一组专门围绕 search-first evidence_judgment 误路由构造的对抗评测。",
        "- 对比对象是当前正式 artifact 与 reviewed-import shadow artifact 在同一批题上的表现。",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def evaluate_learning_focus_search_first_adversarial(root: Path = Path(".")) -> dict[str, object]:
    build_payload = build_learning_focus_search_first_adversarial_eval(root)
    config = load_training_config(root / "configs" / "training.yaml")
    training = config.get("training", {})
    pretraining = training.get("pretraining", {})
    online = training.get("online_evolution", {})

    artifact_dir = root / Path(str(pretraining.get("artifact_dir", "data/pretrain")))
    experience_path = root / Path(str(pretraining.get("experience_path", "data/experience/episodes.jsonl")))
    signal_log_path = root / Path(str(online.get("signal_log_path", "data/evolution/signals.jsonl")))
    shadow_dataset_path = root / "data" / "learning" / "reviewed_import_shadow_corpus.jsonl"
    eval_path = root / "data" / "eval" / "learning_focus_search_first_adversarial_eval.json"
    output_path = root / "artifacts" / "learning_focus_search_first_adversarial_latest.json"
    report_path = root / "backlog" / "opportunities" / "learning_focus_search_first_adversarial.md"

    prompts = load_eval_prompts(eval_path)
    current_payload = evaluate_isolated_prompts(prompts, artifact_dir=artifact_dir)

    with TemporaryDirectory(dir="D:/tmp") as temp_dir:
        shadow_artifact_dir = Path(temp_dir) / "shadow_artifacts"
        trainer = OfflinePretrainer(shadow_artifact_dir)
        trainer.run(
            experience_path=experience_path,
            review_path=root / "data" / "review" / "reviews.jsonl",
            signal_log_path=signal_log_path,
            dataset_path=shadow_dataset_path,
            attention_flow_path=root / "artifacts" / "attention_flow_latest.json",
            attention_views_path=root / "artifacts" / "attention_training_views_latest.json",
            max_episodes=int(pretraining.get("max_episodes", 500)),
            max_synthetic_samples=int(pretraining.get("max_synthetic_samples", 2000)),
            replay_success_only=bool(pretraining.get("replay_success_only", True)),
            reset_artifacts=True,
        )
        shadow_payload = evaluate_isolated_prompts(prompts, artifact_dir=shadow_artifact_dir)

    current_summary = current_payload.get("summary", {}) if isinstance(current_payload, dict) else {}
    shadow_summary = shadow_payload.get("summary", {}) if isinstance(shadow_payload, dict) else {}
    payload = {
        "mode": "controlled_search_first_adversarial_eval",
        "target_failure_count": int(build_payload.get("summary", {}).get("target_failure_count", 0) or 0),
        "target_ids": build_payload.get("summary", {}).get("target_ids", []),
        "eval_path": str(eval_path),
        "shadow_dataset_path": str(shadow_dataset_path),
        "current_summary": current_summary,
        "shadow_summary": shadow_summary,
        "delta": {
            "pretrained_match_rate": round(
                float(shadow_summary.get("pretrained_match_rate", 0.0) or 0.0)
                - float(current_summary.get("pretrained_match_rate", 0.0) or 0.0),
                4,
            ),
            "pretrained_avg_steps": round(
                float(shadow_summary.get("pretrained_avg_steps", 0.0) or 0.0)
                - float(current_summary.get("pretrained_avg_steps", 0.0) or 0.0),
                4,
            ),
        },
        "current_rows": current_payload.get("rows", []) if isinstance(current_payload, dict) else [],
        "shadow_rows": shadow_payload.get("rows", []) if isinstance(shadow_payload, dict) else [],
        "report_path": str(report_path),
        "default_training_admission_changed": False,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _write_report(report_path, payload)
    return payload


def main() -> int:
    payload = evaluate_learning_focus_search_first_adversarial()
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
