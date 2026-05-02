from __future__ import annotations

import json
import shutil
from collections import Counter
from pathlib import Path

from carm.pretrain_data import load_review_feedback
from scripts.preview_learning_intake_human_review_sheet import preview_learning_intake_human_review_sheet


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _write_report(path: Path, summary: dict[str, object]) -> None:
    lines = [
        "# Learning Intake Human Review Draft Sync",
        "",
        "- mode: sync_human_review_draft_to_sheet",
        "- default_runtime_changed: false",
        "- default_training_admission_changed: false",
        f"- source_count: {int(summary.get('source_count', 0) or 0)}",
        f"- synced_count: {int(summary.get('synced_count', 0) or 0)}",
        f"- nonempty_human_status_count: {int(summary.get('nonempty_human_status_count', 0) or 0)}",
        f"- status_counts: {json.dumps(summary.get('status_counts', {}), ensure_ascii=False)}",
        f"- preview_ready_to_apply_count: {int(summary.get('preview_ready_to_apply_count', 0) or 0)}",
        f"- preview_blank_decision_count: {int(summary.get('preview_blank_decision_count', 0) or 0)}",
        "",
        "## Paths",
        "",
        f"- draft_sheet_path: {summary.get('draft_sheet_path', '')}",
        f"- review_sheet_path: {summary.get('review_sheet_path', '')}",
        f"- backup_path: {summary.get('backup_path', '')}",
        f"- preview_report_path: {summary.get('preview_report_path', '')}",
        "",
        "## Notes",
        "",
        "- 这一步只同步草稿表到正式 human review sheet，不会改 review pack。",
        "- 同步完成后会自动刷新 preview，方便马上看到还剩几条待人工决定。",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def sync_learning_intake_human_review_draft(root: Path = Path(".")) -> dict[str, object]:
    review_sheet_path = root / "data" / "learning" / "candidate_pretrain_human_review_sheet.jsonl"
    draft_sheet_path = root / "data" / "learning" / "candidate_pretrain_human_review_sheet.draft.jsonl"
    backup_path = root / "data" / "learning" / "candidate_pretrain_human_review_sheet.backup.jsonl"
    artifact_path = root / "artifacts" / "learning_intake_human_review_draft_sync_latest.json"
    report_path = root / "backlog" / "opportunities" / "learning_intake_human_review_draft_sync.md"

    if not draft_sheet_path.exists():
        raise FileNotFoundError(f"Draft sheet not found: {draft_sheet_path}")

    draft_rows = load_review_feedback(draft_sheet_path)
    status_counts = Counter(
        str(row.get("human_review_status", "")).strip().lower() or "blank"
        for row in draft_rows
        if isinstance(row, dict)
    )

    if review_sheet_path.exists():
        shutil.copyfile(review_sheet_path, backup_path)

    _write_jsonl(review_sheet_path, draft_rows)
    preview_payload = preview_learning_intake_human_review_sheet(root)
    preview_summary = preview_payload.get("summary", {}) if isinstance(preview_payload, dict) else {}

    summary = {
        "mode": "sync_human_review_draft_to_sheet",
        "source_count": len(draft_rows),
        "synced_count": len(draft_rows),
        "nonempty_human_status_count": sum(1 for row in draft_rows if str(row.get("human_review_status", "")).strip()),
        "status_counts": dict(status_counts),
        "draft_sheet_path": str(draft_sheet_path),
        "review_sheet_path": str(review_sheet_path),
        "backup_path": str(backup_path),
        "preview_ready_to_apply_count": int(preview_summary.get("ready_to_apply_count", 0) or 0),
        "preview_blank_decision_count": int(preview_summary.get("blank_decision_count", 0) or 0),
        "preview_report_path": str(preview_summary.get("report_path", "")),
        "artifact_path": str(artifact_path),
        "report_path": str(report_path),
        "default_training_admission_changed": False,
    }

    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _write_report(report_path, summary)
    return summary


def main() -> int:
    summary = sync_learning_intake_human_review_draft()
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
