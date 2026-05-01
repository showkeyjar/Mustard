from __future__ import annotations

import json
from pathlib import Path

from carm.pretrain_data import load_review_feedback


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _write_report(path: Path, payload: dict[str, object]) -> None:
    summary = payload.get("summary", {})
    rows = payload.get("rows", [])
    lines = [
        "# Learning Intake Human Review Draft",
        "",
        "- mode: suggested_human_review_draft",
        "- default_runtime_changed: false",
        "- default_training_admission_changed: false",
        f"- total_candidates: {int(summary.get('total_candidates', 0) or 0)}",
        f"- prefilled_accept_count: {int(summary.get('prefilled_accept_count', 0) or 0)}",
        f"- prefilled_edit_count: {int(summary.get('prefilled_edit_count', 0) or 0)}",
        f"- prefilled_pending_count: {int(summary.get('prefilled_pending_count', 0) or 0)}",
        "",
        "## How To Use",
        "",
        f"1. 打开 `{summary.get('draft_sheet_path', '')}`。",
        "2. 在这个草稿副本里继续改 `human_review_status` / `human_review_note`。",
        "3. 确认后，把需要保留的内容同步回正式 `candidate_pretrain_human_review_sheet.jsonl`，再运行 preview/apply。",
        "",
        "## Prefilled Rows",
        "",
    ]
    for row in rows:
        lines.extend(
            [
                f"### {row.get('sample_id', '')}",
                f"- suggested_review_status: {row.get('suggested_review_status', '')}",
                f"- draft_human_review_status: {row.get('human_review_status', '')}",
                f"- prompt: {row.get('user_input', '')}",
                "",
            ]
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def build_learning_intake_human_review_draft(root: Path = Path(".")) -> dict[str, object]:
    source_path = root / "data" / "learning" / "candidate_pretrain_human_review_sheet.jsonl"
    draft_path = root / "data" / "learning" / "candidate_pretrain_human_review_sheet.draft.jsonl"
    artifact_path = root / "artifacts" / "learning_intake_human_review_draft_latest.json"
    report_path = root / "backlog" / "opportunities" / "learning_intake_human_review_draft.md"

    source_rows = load_review_feedback(source_path)
    draft_rows: list[dict[str, object]] = []
    prefilled_accept_count = 0
    prefilled_edit_count = 0
    prefilled_pending_count = 0
    report_rows: list[dict[str, object]] = []

    for row in source_rows:
        updated = dict(row)
        suggested_status = str(row.get("suggested_review_status", "pending")).strip().lower() or "pending"
        suggested_decision = str(row.get("suggested_decision", "")).strip().lower()
        if suggested_status in {"accept", "edit"}:
            updated["human_review_status"] = suggested_status
            updated["human_review_note"] = f"suggested:{suggested_decision or suggested_status}"
            if suggested_status == "accept":
                prefilled_accept_count += 1
            else:
                prefilled_edit_count += 1
        else:
            updated["human_review_status"] = ""
            updated["human_review_note"] = ""
            prefilled_pending_count += 1
        draft_rows.append(updated)
        report_rows.append(
            {
                "sample_id": str(updated.get("sample_id", "")).strip(),
                "user_input": str(updated.get("user_input", "")).strip(),
                "suggested_review_status": suggested_status,
                "human_review_status": str(updated.get("human_review_status", "")).strip(),
            }
        )

    summary = {
        "mode": "suggested_human_review_draft",
        "total_candidates": len(draft_rows),
        "prefilled_accept_count": prefilled_accept_count,
        "prefilled_edit_count": prefilled_edit_count,
        "prefilled_pending_count": prefilled_pending_count,
        "draft_sheet_path": str(draft_path),
        "artifact_path": str(artifact_path),
        "report_path": str(report_path),
        "default_training_admission_changed": False,
    }
    payload = {"summary": summary, "rows": report_rows}

    _write_jsonl(draft_path, draft_rows)
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _write_report(report_path, payload)
    return payload


def main() -> int:
    payload = build_learning_intake_human_review_draft()
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
