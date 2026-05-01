from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from carm.pretrain_data import load_pretrain_samples, normalize_user_input
from carm.training import load_training_config


def _load_jsonl(path: Path) -> list[dict[str, object]]:
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


def _write_report(path: Path, payload: dict[str, object]) -> None:
    summary = payload.get("summary", {})
    added_rows = payload.get("added_rows", [])
    deduped_rows = payload.get("deduped_import_rows", [])
    lines = [
        "# Learning Intake Suggested Shadow Delta",
        "",
        f"- base_sample_count: {int(summary.get('base_sample_count', 0) or 0)}",
        f"- shadow_sample_count: {int(summary.get('shadow_sample_count', 0) or 0)}",
        f"- added_count: {int(summary.get('added_count', 0) or 0)}",
        f"- removed_count: {int(summary.get('removed_count', 0) or 0)}",
        f"- suggested_import_count: {int(summary.get('suggested_import_count', 0) or 0)}",
        f"- surviving_import_count: {int(summary.get('surviving_import_count', 0) or 0)}",
        f"- deduped_import_count: {int(summary.get('deduped_import_count', 0) or 0)}",
        f"- added_source_counts: {json.dumps(summary.get('added_source_counts', {}), ensure_ascii=False)}",
        "",
        "## Surviving Additions",
        "",
    ]
    for row in added_rows:
        lines.extend(
            [
                f"### {row.get('logic_skill', '')}",
                f"- source_type: {row.get('source_type', '')}",
                f"- candidate_source_type: {row.get('candidate_source_type', '')}",
                f"- review_status: {row.get('review_status', '')}",
                f"- prompt: {row.get('user_input', '')}",
                "",
            ]
        )
    if deduped_rows:
        lines.extend(["## Deduped Imports", ""])
        for row in deduped_rows:
            lines.extend(
                [
                    f"### {row.get('logic_skill', '')}",
                    f"- candidate_source_type: {row.get('candidate_source_type', '')}",
                    f"- review_status: {row.get('review_status', '')}",
                    f"- prompt: {row.get('prompt', '')}",
                    "",
                ]
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def compare_learning_intake_suggested_shadow(root: Path = Path(".")) -> dict[str, object]:
    config = load_training_config(str(root / "configs" / "training.yaml"))
    pretraining = config.get("training", {}).get("pretraining", {})
    dataset_path = root / Path(str(pretraining.get("dataset_path", "data/pretrain/pretrain_corpus.jsonl")))

    base_samples = load_pretrain_samples(dataset_path)
    shadow_samples = load_pretrain_samples(root / "data" / "learning" / "candidate_pretrain_suggested_corpus.jsonl")
    suggested_import_rows = _load_jsonl(root / "data" / "learning" / "candidate_pretrain_suggested_import.jsonl")

    base_by_key = {normalize_user_input(sample.user_input): sample for sample in base_samples}
    shadow_by_key = {normalize_user_input(sample.user_input): sample for sample in shadow_samples}
    import_by_key = {normalize_user_input(str(row.get("prompt", ""))): row for row in suggested_import_rows if str(row.get("prompt", "")).strip()}

    added_keys = [key for key in shadow_by_key if key not in base_by_key]
    removed_keys = [key for key in base_by_key if key not in shadow_by_key]
    surviving_import_keys = [key for key in import_by_key if key in shadow_by_key and key not in base_by_key]
    deduped_import_keys = [key for key in import_by_key if key not in surviving_import_keys]

    added_rows = []
    for key in added_keys:
        sample = shadow_by_key[key]
        import_row = import_by_key.get(key, {})
        metadata = dict(sample.metadata or {})
        import_meta = import_row.get("metadata", {}) if isinstance(import_row, dict) else {}
        import_meta = import_meta if isinstance(import_meta, dict) else {}
        added_rows.append(
            {
                "user_input": sample.user_input,
                "source_type": sample.source_type,
                "logic_skill": sample.logic_skill,
                "quality_score": sample.quality_score,
                "candidate_source_type": str(import_meta.get("candidate_source_type", metadata.get("candidate_source_type", ""))),
                "review_status": str(import_meta.get("review_status", metadata.get("review_status", ""))),
            }
        )

    deduped_rows = []
    for key in deduped_import_keys:
        row = import_by_key[key]
        meta = row.get("metadata", {}) if isinstance(row, dict) else {}
        meta = meta if isinstance(meta, dict) else {}
        deduped_rows.append(
            {
                "prompt": str(row.get("prompt", "")),
                "logic_skill": str(row.get("logic_skill", "")),
                "candidate_source_type": str(meta.get("candidate_source_type", "")),
                "review_status": str(meta.get("review_status", "")),
            }
        )

    payload = {
        "summary": {
            "base_sample_count": len(base_samples),
            "shadow_sample_count": len(shadow_samples),
            "added_count": len(added_keys),
            "removed_count": len(removed_keys),
            "suggested_import_count": len(suggested_import_rows),
            "surviving_import_count": len(surviving_import_keys),
            "deduped_import_count": len(deduped_import_keys),
            "added_source_counts": dict(Counter(row["candidate_source_type"] or row["source_type"] for row in added_rows)),
        },
        "added_rows": added_rows,
        "deduped_import_rows": deduped_rows,
        "sources": {
            "base_dataset": str(dataset_path),
            "shadow_dataset": "data/learning/candidate_pretrain_suggested_corpus.jsonl",
            "suggested_import": "data/learning/candidate_pretrain_suggested_import.jsonl",
        },
    }

    artifact_path = root / "artifacts" / "learning_intake_suggested_shadow_delta_latest.json"
    report_path = root / "backlog" / "opportunities" / "learning_intake_suggested_shadow_delta.md"
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _write_report(report_path, payload)
    return payload


def main() -> int:
    payload = compare_learning_intake_suggested_shadow()
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
