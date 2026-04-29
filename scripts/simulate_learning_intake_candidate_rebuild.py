from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from carm.pretrain_data import (
    export_review_pack,
    load_pretrain_samples,
    load_review_feedback,
    merge_and_filter_samples,
    review_payload_to_sample,
    save_pretrain_samples,
)
from carm.training import load_training_config


def _read_json(path: Path) -> dict[str, object]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _write_report(path: Path, summary: dict[str, object]) -> None:
    lines = [
        "# Learning Intake Suggested Rebuild Report",
        "",
        "- mode: simulated_review_rebuild",
        "- default_runtime_changed: false",
        "- default_training_admission_changed: false",
        f"- source_review_count: {int(summary.get('source_review_count', 0) or 0)}",
        f"- suggested_accept_count: {int(summary.get('suggested_accept_count', 0) or 0)}",
        f"- suggested_edit_count: {int(summary.get('suggested_edit_count', 0) or 0)}",
        f"- suggested_import_count: {int(summary.get('suggested_import_count', 0) or 0)}",
        f"- base_sample_count: {int(summary.get('base_sample_count', 0) or 0)}",
        f"- shadow_sample_count: {int(summary.get('shadow_sample_count', 0) or 0)}",
        "",
        "## Paths",
        "",
        f"- suggested_review_pack: {summary.get('suggested_review_pack_path', '')}",
        f"- suggested_import: {summary.get('suggested_import_path', '')}",
        f"- suggested_shadow_corpus: {summary.get('suggested_shadow_corpus_path', '')}",
        "",
        "## Notes",
        "",
        "- 这是按系统推荐状态生成的影子结果，不会改动原始 review pack。",
        "- 只有 suggested accept/edit 会进入 suggested import 和 shadow corpus。",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def simulate_learning_intake_candidate_rebuild(root: Path = Path(".")) -> dict[str, object]:
    review_pack_path = root / "data" / "learning" / "candidate_pretrain_review_pack.jsonl"
    queue_payload = _read_json(root / "artifacts" / "learning_intake_review_queue_latest.json")
    queue_rows = queue_payload.get("queue", []) if isinstance(queue_payload, dict) else []
    if not isinstance(queue_rows, list):
        queue_rows = []

    recommendation_by_prompt = {
        str(row.get("user_input", "")).strip(): str(row.get("recommended_status", "hold")).strip().lower()
        for row in queue_rows
        if isinstance(row, dict) and str(row.get("user_input", "")).strip()
    }

    config = load_training_config(str(root / "configs" / "training.yaml"))
    pretraining = config.get("training", {}).get("pretraining", {})
    dataset_path = root / Path(str(pretraining.get("dataset_path", "data/pretrain/pretrain_corpus.jsonl")))
    min_quality_score = float(pretraining.get("min_quality_score", 0.72))
    max_samples = int(pretraining.get("max_dataset_samples", 5000))

    base_samples = load_pretrain_samples(dataset_path)
    review_payloads = load_review_feedback(review_pack_path)

    suggested_review_rows: list[dict[str, object]] = []
    approved_import_rows: list[dict[str, object]] = []
    approved_samples = []
    status_counts: Counter[str] = Counter()

    for payload in review_payloads:
        prompt = str(payload.get("user_input", "")).strip()
        recommended_status = recommendation_by_prompt.get(prompt, "hold")
        updated = dict(payload)
        if recommended_status in {"accept", "edit"}:
            updated["review_status"] = recommended_status
            updated["review_note"] = f"auto_suggested:{recommended_status}"
        else:
            updated["review_status"] = "pending"
        suggested_review_rows.append(updated)
        status_counts[str(updated["review_status"])] += 1

        if updated["review_status"] in {"accept", "edit"}:
            sample = review_payload_to_sample(updated)
            approved_samples.append(sample)
            approved_import_rows.append(
                {
                    "prompt": sample.user_input,
                    "source_type": sample.source_type,
                    "logic_skill": sample.logic_skill,
                    "quality_score": sample.quality_score,
                    "metadata": {
                        **sample.metadata,
                        "review_status": updated["review_status"],
                        "candidate_source_type": str(payload.get("source_type", "")),
                        "simulation_mode": "auto_recommended",
                    },
                }
            )

    shadow_corpus = merge_and_filter_samples(
        base_samples + approved_samples,
        min_quality_score=min_quality_score,
        max_samples=max_samples,
    )

    output_dir = root / "data" / "learning"
    suggested_review_pack_path = output_dir / "candidate_pretrain_suggested_review_pack.jsonl"
    suggested_import_path = output_dir / "candidate_pretrain_suggested_import.jsonl"
    suggested_shadow_corpus_path = output_dir / "candidate_pretrain_suggested_corpus.jsonl"
    suggested_shadow_review_pack_path = output_dir / "candidate_pretrain_suggested_shadow_review_pack.jsonl"
    artifact_path = root / "artifacts" / "learning_intake_suggested_rebuild_latest.json"
    report_path = root / "backlog" / "opportunities" / "learning_intake_suggested_rebuild_report.md"

    _write_jsonl(suggested_review_pack_path, suggested_review_rows)
    _write_jsonl(suggested_import_path, approved_import_rows)
    save_pretrain_samples(suggested_shadow_corpus_path, shadow_corpus)
    export_review_pack(suggested_shadow_review_pack_path, approved_samples, limit=min(50, len(approved_samples)))

    summary = {
        "mode": "simulated_review_rebuild",
        "source_review_count": len(review_payloads),
        "suggested_accept_count": int(status_counts.get("accept", 0)),
        "suggested_edit_count": int(status_counts.get("edit", 0)),
        "suggested_import_count": len(approved_import_rows),
        "base_sample_count": len(base_samples),
        "shadow_sample_count": len(shadow_corpus),
        "suggested_review_pack_path": str(suggested_review_pack_path),
        "suggested_import_path": str(suggested_import_path),
        "suggested_shadow_corpus_path": str(suggested_shadow_corpus_path),
        "suggested_shadow_review_pack_path": str(suggested_shadow_review_pack_path),
        "status_counts": dict(status_counts),
        "default_training_admission_changed": False,
        "report_path": str(report_path),
    }
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _write_report(report_path, summary)
    return summary


def main() -> int:
    summary = simulate_learning_intake_candidate_rebuild()
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
