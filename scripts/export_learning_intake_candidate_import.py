from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from carm.pretrain_data import load_review_feedback, review_payload_to_sample


def _write_import(path: Path, payloads: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for item in payloads:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")


def _write_report(path: Path, summary: dict[str, object]) -> None:
    lines = [
        "# Learning Intake Candidate Import Report",
        "",
        "- mode: reviewed_export_only",
        "- default_runtime_changed: false",
        "- default_training_admission_changed: false",
        f"- review_total: {int(summary.get('review_total', 0) or 0)}",
        f"- approved_count: {int(summary.get('approved_count', 0) or 0)}",
        f"- pending_count: {int(summary.get('pending_count', 0) or 0)}",
        f"- rejected_count: {int(summary.get('rejected_count', 0) or 0)}",
        f"- status_counts: {json.dumps(summary.get('status_counts', {}), ensure_ascii=False)}",
        "",
        "## Paths",
        "",
        f"- import_path: {summary.get('import_path', '')}",
        "",
        "## Gate",
        "",
        "- 只有 review_status=accept/edit 的样本会进入 import 候选。",
        "- pending 样本保持在 review pack，不进入离线构建。",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def export_learning_intake_candidate_import(root: Path = Path(".")) -> dict[str, object]:
    review_pack_path = root / "data" / "learning" / "candidate_pretrain_review_pack.jsonl"
    import_path = root / "data" / "learning" / "candidate_pretrain_import.jsonl"
    artifact_path = root / "artifacts" / "learning_intake_candidate_import_latest.json"
    report_path = root / "backlog" / "opportunities" / "learning_intake_candidate_import_report.md"

    review_payloads = load_review_feedback(review_pack_path)
    status_counts = Counter(str(item.get("review_status", "pending")).strip().lower() or "pending" for item in review_payloads)

    approved_payloads: list[dict[str, object]] = []
    for payload in review_payloads:
        status = str(payload.get("review_status", "pending")).strip().lower()
        if status not in {"accept", "edit"}:
            continue
        sample = review_payload_to_sample(payload)
        approved_payloads.append(
            {
                "prompt": sample.user_input,
                "source_type": sample.source_type,
                "logic_skill": sample.logic_skill,
                "quality_score": sample.quality_score,
                "metadata": {
                    **sample.metadata,
                    "review_status": status,
                    "candidate_source_type": str(payload.get("source_type", "")),
                },
            }
        )

    _write_import(import_path, approved_payloads)

    summary = {
        "mode": "reviewed_export_only",
        "review_total": len(review_payloads),
        "approved_count": len(approved_payloads),
        "pending_count": int(status_counts.get("pending", 0)),
        "rejected_count": int(status_counts.get("reject", 0)),
        "status_counts": dict(status_counts),
        "import_path": str(import_path),
        "report_path": str(report_path),
        "default_training_admission_changed": False,
    }
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _write_report(report_path, summary)
    return summary


def main() -> int:
    summary = export_learning_intake_candidate_import()
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
