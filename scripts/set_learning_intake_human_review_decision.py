from __future__ import annotations

import json
import shutil
from pathlib import Path

from carm.pretrain_data import load_review_feedback


STATUS_ALIASES = {
    "accept": "accept",
    "edit": "edit",
    "reject": "reject",
    "pending": "pending",
    "defer": "pending",
}


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _write_report(path: Path, summary: dict[str, object]) -> None:
    lines = [
        "# Learning Intake Human Review Decision",
        "",
        "- mode: set_single_human_review_decision",
        "- default_runtime_changed: false",
        "- default_training_admission_changed: false",
        f"- sample_id: {summary.get('sample_id', '')}",
        f"- requested_status: {summary.get('requested_status', '')}",
        f"- applied_status: {summary.get('applied_status', '')}",
        f"- updated: {str(bool(summary.get('updated', False))).lower()}",
        "",
        "## Paths",
        "",
        f"- review_sheet_path: {summary.get('review_sheet_path', '')}",
        f"- backup_path: {summary.get('backup_path', '')}",
        "",
        "## Notes",
        "",
        "- `defer` 会被落成 `pending`，用于保留候选但暂不放行。",
        "- 修改的是 human review sheet，不会直接改正式 review pack。",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def set_learning_intake_human_review_decision(
    sample_id: str,
    status: str,
    *,
    note: str = "",
    root: Path = Path("."),
) -> dict[str, object]:
    normalized_sample_id = sample_id.strip()
    requested_status = status.strip().lower()
    applied_status = STATUS_ALIASES.get(requested_status, "")
    if not normalized_sample_id:
        raise ValueError("sample_id is required")
    if not applied_status:
        raise ValueError(f"Unsupported status: {status}")

    review_sheet_path = root / "data" / "learning" / "candidate_pretrain_human_review_sheet.jsonl"
    backup_path = root / "data" / "learning" / "candidate_pretrain_human_review_sheet.backup.jsonl"
    artifact_path = root / "artifacts" / "learning_intake_human_review_decision_latest.json"
    report_path = root / "backlog" / "opportunities" / "learning_intake_human_review_decision.md"

    rows = load_review_feedback(review_sheet_path)
    updated_rows: list[dict[str, object]] = []
    updated = False
    resolved_note = note.strip() or f"manual:{requested_status}"

    for row in rows:
        payload = dict(row)
        current_sample_id = str(payload.get("sample_id", "")).strip()
        if current_sample_id == normalized_sample_id:
            payload["human_review_status"] = applied_status
            payload["human_review_note"] = resolved_note
            updated = True
        updated_rows.append(payload)

    if not updated:
        raise ValueError(f"Sample not found: {normalized_sample_id}")

    shutil.copyfile(review_sheet_path, backup_path)
    _write_jsonl(review_sheet_path, updated_rows)

    summary = {
        "mode": "set_single_human_review_decision",
        "sample_id": normalized_sample_id,
        "requested_status": requested_status,
        "applied_status": applied_status,
        "updated": updated,
        "review_sheet_path": str(review_sheet_path),
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
    raise SystemExit("Use via scripts.claw_team_control decide-human-review")


if __name__ == "__main__":
    main()
