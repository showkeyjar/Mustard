from __future__ import annotations

import json
import shutil
from collections import Counter
from pathlib import Path

from carm.pretrain_data import load_review_feedback


VALID_REVIEW_STATUSES = {"pending", "accept", "edit", "reject"}
HELPER_STATUS_FIELDS = {
    "suggested_decision",
    "suggested_review_status",
    "suggested_review_note",
    "suggested_why",
    "human_review_status",
    "human_review_note",
    "sample_id",
}


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _sanitize_row(row: dict[str, object]) -> dict[str, object]:
    payload = {key: value for key, value in row.items() if key not in HELPER_STATUS_FIELDS}
    return payload


def _write_report(path: Path, summary: dict[str, object]) -> None:
    lines = [
        "# Learning Intake Human Review Apply Report",
        "",
        "- mode: apply_human_review_sheet",
        "- default_runtime_changed: false",
        "- default_training_admission_changed: false",
        f"- review_total: {int(summary.get('review_total', 0) or 0)}",
        f"- applied_count: {int(summary.get('applied_count', 0) or 0)}",
        f"- accepted_count: {int(summary.get('accepted_count', 0) or 0)}",
        f"- edited_count: {int(summary.get('edited_count', 0) or 0)}",
        f"- rejected_count: {int(summary.get('rejected_count', 0) or 0)}",
        f"- pending_count: {int(summary.get('pending_count', 0) or 0)}",
        "",
        "## Paths",
        "",
        f"- review_pack_path: {summary.get('review_pack_path', '')}",
        f"- backup_path: {summary.get('backup_path', '')}",
        "",
        "## Notes",
        "",
        "- 只有 `human_review_status` 非空的行会覆盖原 review pack。",
        "- helper 字段不会写回正式 review pack。",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def apply_learning_intake_human_review_sheet(root: Path = Path(".")) -> dict[str, object]:
    review_pack_path = root / "data" / "learning" / "candidate_pretrain_review_pack.jsonl"
    review_sheet_path = root / "data" / "learning" / "candidate_pretrain_human_review_sheet.jsonl"
    backup_path = root / "data" / "learning" / "candidate_pretrain_review_pack.backup.jsonl"
    artifact_path = root / "artifacts" / "learning_intake_human_review_apply_latest.json"
    report_path = root / "backlog" / "opportunities" / "learning_intake_human_review_apply.md"

    review_pack = load_review_feedback(review_pack_path)
    review_sheet = load_review_feedback(review_sheet_path)

    row_by_prompt = {
        str(row.get("user_input", "")).strip(): row
        for row in review_sheet
        if isinstance(row, dict) and str(row.get("user_input", "")).strip()
    }

    updated_rows: list[dict[str, object]] = []
    applied_count = 0

    for payload in review_pack:
        prompt = str(payload.get("user_input", "")).strip()
        sheet_row = row_by_prompt.get(prompt)
        if not sheet_row:
            updated_rows.append(dict(payload))
            continue
        human_status = str(sheet_row.get("human_review_status", "")).strip().lower()
        if not human_status:
            updated_rows.append(dict(payload))
            continue
        if human_status not in VALID_REVIEW_STATUSES:
            raise ValueError(f"Unsupported human_review_status: {human_status}")

        updated = _sanitize_row(dict(sheet_row))
        updated["review_status"] = human_status
        updated["review_note"] = str(sheet_row.get("human_review_note", "")).strip()
        updated_rows.append(updated)
        applied_count += 1

    shutil.copyfile(review_pack_path, backup_path)
    _write_jsonl(review_pack_path, updated_rows)

    status_counts = Counter(str(row.get("review_status", "pending")).strip().lower() or "pending" for row in updated_rows)
    summary = {
        "mode": "apply_human_review_sheet",
        "review_total": len(updated_rows),
        "applied_count": applied_count,
        "accepted_count": int(status_counts.get("accept", 0)),
        "edited_count": int(status_counts.get("edit", 0)),
        "rejected_count": int(status_counts.get("reject", 0)),
        "pending_count": int(status_counts.get("pending", 0)),
        "status_counts": dict(status_counts),
        "review_pack_path": str(review_pack_path),
        "backup_path": str(backup_path),
        "artifact_path": str(artifact_path),
        "report_path": str(report_path),
        "default_training_admission_changed": False,
    }

    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _write_report(report_path, summary)
    return summary


def main() -> int:
    summary = apply_learning_intake_human_review_sheet()
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
