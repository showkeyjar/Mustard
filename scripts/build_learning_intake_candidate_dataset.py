from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from carm.pretrain_data import export_review_pack, load_pretrain_samples, merge_and_filter_samples, save_pretrain_samples
from carm.training import load_training_config


DEFAULT_INCLUDE_SOURCES = {
    "learning_intake:learning_focus_stress",
    "learning_intake:attention_gap",
}


def _write_report(path: Path, payload: dict[str, object]) -> None:
    lines = [
        "# Learning Intake Candidate Dataset Report",
        "",
        "- mode: preview_only",
        "- default_runtime_changed: false",
        "- default_training_admission_changed: false",
        f"- base_sample_count: {int(payload.get('base_sample_count', 0) or 0)}",
        f"- selected_candidate_count: {int(payload.get('selected_candidate_count', 0) or 0)}",
        f"- merged_sample_count: {int(payload.get('merged_sample_count', 0) or 0)}",
        f"- selected_source_counts: {json.dumps(payload.get('selected_source_counts', {}), ensure_ascii=False)}",
        "",
        "## Paths",
        "",
        f"- candidate_dataset: {payload.get('candidate_dataset_path', '')}",
        f"- candidate_review_pack: {payload.get('candidate_review_pack_path', '')}",
        "",
        "## Recommended Next Step",
        "",
        "- Step 1: 审阅 candidate review pack，确认 learning_focus_stress / attention_gap 样本是否值得进入正式离线构建。",
        "- Step 2: 若通过人工审阅，再将对应 import path 用于 build_pretrain_dataset 或 auto_train。",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def build_learning_intake_candidate_dataset(
    root: Path = Path("."),
    include_sources: set[str] | None = None,
) -> dict[str, object]:
    include_sources = include_sources or set(DEFAULT_INCLUDE_SOURCES)
    config = load_training_config(str(root / "configs" / "training.yaml"))
    pretraining = config.get("training", {}).get("pretraining", {})

    dataset_path = Path(str(pretraining.get("dataset_path", "data/pretrain/pretrain_corpus.jsonl")))
    min_quality_score = float(pretraining.get("min_quality_score", 0.72))
    max_samples = int(pretraining.get("max_dataset_samples", 5000))

    base_samples = load_pretrain_samples(root / dataset_path)
    intake_samples = load_pretrain_samples(root / "data" / "learning" / "learning_intake_samples.jsonl")
    selected_samples = [sample for sample in intake_samples if sample.source_type in include_sources]

    merged = merge_and_filter_samples(
        base_samples + selected_samples,
        min_quality_score=min_quality_score,
        max_samples=max_samples,
    )

    output_dir = root / "data" / "learning"
    output_dir.mkdir(parents=True, exist_ok=True)
    candidate_dataset_path = output_dir / "candidate_pretrain_corpus.jsonl"
    candidate_review_pack_path = output_dir / "candidate_pretrain_review_pack.jsonl"
    artifact_path = root / "artifacts" / "learning_intake_candidate_dataset_latest.json"
    report_path = root / "backlog" / "opportunities" / "learning_intake_candidate_dataset_report.md"

    save_pretrain_samples(candidate_dataset_path, merged)
    export_review_pack(candidate_review_pack_path, selected_samples, limit=min(50, len(selected_samples)))

    payload = {
        "mode": "preview_only",
        "base_sample_count": len(base_samples),
        "selected_candidate_count": len(selected_samples),
        "merged_sample_count": len(merged),
        "selected_source_counts": dict(Counter(sample.source_type for sample in selected_samples)),
        "include_sources": sorted(include_sources),
        "candidate_dataset_path": str(candidate_dataset_path),
        "candidate_review_pack_path": str(candidate_review_pack_path),
        "report_path": str(report_path),
        "default_training_admission_changed": False,
    }
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _write_report(report_path, payload)
    return payload


def main() -> int:
    payload = build_learning_intake_candidate_dataset()
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
