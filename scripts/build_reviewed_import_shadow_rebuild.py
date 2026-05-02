from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from carm.pretrain_data import (
    export_review_pack,
    load_pretrain_samples,
    load_review_feedback,
    merge_and_filter_samples,
    normalize_user_input,
    review_payload_to_sample,
    save_pretrain_samples,
)
from carm.training import load_training_config


def _load_import_rows(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []
    rows: list[dict[str, object]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        if isinstance(payload, dict):
            rows.append(payload)
    return rows


def _write_report(path: Path, summary: dict[str, object]) -> None:
    lines = [
        "# Reviewed Import Shadow Rebuild",
        "",
        "- mode: controlled_shadow_rebuild",
        "- default_runtime_changed: false",
        "- default_training_admission_changed: false",
        f"- base_sample_count: {int(summary.get('base_sample_count', 0) or 0)}",
        f"- approved_review_count: {int(summary.get('approved_review_count', 0) or 0)}",
        f"- import_prompt_count: {int(summary.get('import_prompt_count', 0) or 0)}",
        f"- shadow_sample_count: {int(summary.get('shadow_sample_count', 0) or 0)}",
        f"- added_count: {int(summary.get('added_count', 0) or 0)}",
        f"- surviving_import_count: {int(summary.get('surviving_import_count', 0) or 0)}",
        f"- deduped_import_count: {int(summary.get('deduped_import_count', 0) or 0)}",
        f"- approved_source_counts: {json.dumps(summary.get('approved_source_counts', {}), ensure_ascii=False)}",
        "",
        "## Paths",
        "",
        f"- import_path: {summary.get('import_path', '')}",
        f"- shadow_corpus_path: {summary.get('shadow_corpus_path', '')}",
        f"- shadow_review_pack_path: {summary.get('shadow_review_pack_path', '')}",
        "",
        "## Notes",
        "",
        "- 这是 Human Gate 通过后的受控影子重建，不会覆盖正式 pretrain corpus。",
        "- 只有 review_status=accept/edit 且进入 import 的样本会参与这次重建。",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def build_reviewed_import_shadow_rebuild(root: Path = Path(".")) -> dict[str, object]:
    config = load_training_config(str(root / "configs" / "training.yaml"))
    pretraining = config.get("training", {}).get("pretraining", {})

    dataset_path = root / Path(str(pretraining.get("dataset_path", "data/pretrain/pretrain_corpus.jsonl")))
    min_quality_score = float(pretraining.get("min_quality_score", 0.72))
    max_samples = int(pretraining.get("max_dataset_samples", 5000))

    review_pack_path = root / "data" / "learning" / "candidate_pretrain_review_pack.jsonl"
    import_path = root / "data" / "learning" / "candidate_pretrain_import.jsonl"
    shadow_corpus_path = root / "data" / "learning" / "reviewed_import_shadow_corpus.jsonl"
    shadow_review_pack_path = root / "data" / "learning" / "reviewed_import_shadow_review_pack.jsonl"
    artifact_path = root / "artifacts" / "reviewed_import_shadow_rebuild_latest.json"
    report_path = root / "backlog" / "opportunities" / "reviewed_import_shadow_rebuild.md"

    base_samples = load_pretrain_samples(dataset_path)
    review_rows = load_review_feedback(review_pack_path)
    import_rows = _load_import_rows(import_path)

    approved_rows = [
        row for row in review_rows if str(row.get("review_status", "pending")).strip().lower() in {"accept", "edit"}
    ]
    approved_samples = [review_payload_to_sample(row) for row in approved_rows]
    shadow_samples = merge_and_filter_samples(
        base_samples + approved_samples,
        min_quality_score=min_quality_score,
        max_samples=max_samples,
    )

    save_pretrain_samples(shadow_corpus_path, shadow_samples)
    export_review_pack(shadow_review_pack_path, approved_samples, limit=min(50, len(approved_samples)))

    base_keys = {normalize_user_input(sample.user_input) for sample in base_samples}
    shadow_keys = {normalize_user_input(sample.user_input) for sample in shadow_samples}
    import_keys = {
        normalize_user_input(str(row.get("prompt", "")).strip())
        for row in import_rows
        if str(row.get("prompt", "")).strip()
    }

    added_keys = shadow_keys - base_keys
    surviving_import_count = sum(1 for key in import_keys if key in added_keys)
    deduped_import_count = max(0, len(import_keys) - surviving_import_count)

    summary = {
        "mode": "controlled_shadow_rebuild",
        "base_sample_count": len(base_samples),
        "approved_review_count": len(approved_rows),
        "import_prompt_count": len(import_keys),
        "shadow_sample_count": len(shadow_samples),
        "added_count": len(added_keys),
        "surviving_import_count": surviving_import_count,
        "deduped_import_count": deduped_import_count,
        "approved_source_counts": dict(Counter(str(row.get("source_type", "")).strip() for row in approved_rows)),
        "import_path": str(import_path),
        "shadow_corpus_path": str(shadow_corpus_path),
        "shadow_review_pack_path": str(shadow_review_pack_path),
        "artifact_path": str(artifact_path),
        "report_path": str(report_path),
        "default_training_admission_changed": False,
    }

    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _write_report(report_path, summary)
    return summary


def main() -> int:
    summary = build_reviewed_import_shadow_rebuild()
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
