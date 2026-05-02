from __future__ import annotations

import json
from pathlib import Path

from carm.pretrain_data import load_review_feedback
from scripts.set_learning_intake_human_review_decision import set_learning_intake_human_review_decision


def _read_json(path: Path) -> dict[str, object]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def _write_report(path: Path, payload: dict[str, object]) -> None:
    lines = [
        "# Decide Top Human Review",
        "",
        "- mode: decide_top_human_review",
        "- default_runtime_changed: false",
        "- default_training_admission_changed: false",
        f"- selected_sample_id: {payload.get('selected_sample_id', '')}",
        f"- selected_status: {payload.get('selected_status', '')}",
        f"- selected_source_type: {payload.get('selected_source_type', '')}",
        f"- priority_score: {payload.get('priority_score', 0)}",
        "",
        "## Notes",
        "",
        "- 该命令会读取当前 human review sheet 中还未填写的候选，优先选择最高 priority_score 的一条。",
        "- 若 suggested_review_status=pending，则会按 defer 处理，实际写回 pending。",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def decide_top_learning_intake_human_review(root: Path = Path(".")) -> dict[str, object]:
    review_sheet_path = root / "data" / "learning" / "candidate_pretrain_human_review_sheet.jsonl"
    sheet_rows = load_review_feedback(review_sheet_path)
    pending_rows = []
    for row in sheet_rows:
        if not isinstance(row, dict):
            continue
        if str(row.get("human_review_status", "")).strip():
            continue
        sample_id = str(row.get("sample_id", "")).strip()
        suggested = str(row.get("suggested_review_status", "")).strip().lower() or "pending"
        if not sample_id:
            continue
        pending_rows.append(
            {
                "sample_id": sample_id,
                "source_type": str(row.get("source_type", "")).strip(),
                "priority_score": float(row.get("priority_score", 0.0) or 0.0),
                "status": "defer" if suggested == "pending" else suggested,
            }
        )

    if not pending_rows:
        raise ValueError("No pending human review candidates found")

    pending_rows.sort(key=lambda item: item["priority_score"], reverse=True)
    selected = pending_rows[0]
    decision_summary = set_learning_intake_human_review_decision(
        str(selected["sample_id"]),
        str(selected["status"]),
        note="auto:top_priority",
        root=root,
    )

    artifact_path = root / "artifacts" / "decide_top_human_review_latest.json"
    report_path = root / "backlog" / "opportunities" / "decide_top_human_review.md"
    payload = {
        "mode": "decide_top_human_review",
        "selected_sample_id": str(selected["sample_id"]),
        "selected_status": str(selected["status"]),
        "selected_source_type": str(selected["source_type"]),
        "priority_score": float(selected["priority_score"]),
        "decision_summary": decision_summary,
        "artifact_path": str(artifact_path),
        "report_path": str(report_path),
        "default_training_admission_changed": False,
    }
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _write_report(report_path, payload)
    return payload


def main() -> int:
    payload = decide_top_learning_intake_human_review()
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
