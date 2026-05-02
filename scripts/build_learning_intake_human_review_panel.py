from __future__ import annotations

import json
import re
from pathlib import Path

from carm.pretrain_data import load_review_feedback


SAMPLE_ID_PATTERN = re.compile(r"样本=([^。]+)")


def _read_json(path: Path) -> dict[str, object]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _extract_sample_id(prompt: str) -> str:
    match = SAMPLE_ID_PATTERN.search(prompt)
    return match.group(1).strip() if match else ""


def _fallback_sample_id(payload: dict[str, object], prompt: str) -> str:
    source_type = str(payload.get("source_type", "")).strip()
    if source_type == "learning_intake:attention_gap":
        return "attention-gap-001"
    if source_type:
        return source_type.replace(":", "-")
    return prompt[:48]


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _write_report(path: Path, panel: dict[str, object]) -> None:
    summary = panel.get("summary", {})
    rows = panel.get("rows", [])
    lines = [
        "# Learning Intake Human Review Panel",
        "",
        "- mode: human_friendly_review_panel",
        "- default_runtime_changed: false",
        "- default_training_admission_changed: false",
        f"- total_candidates: {int(summary.get('total_candidates', 0) or 0)}",
        f"- recommend_accept: {int(summary.get('recommend_accept', 0) or 0)}",
        f"- recommend_edit: {int(summary.get('recommend_edit', 0) or 0)}",
        f"- recommend_defer: {int(summary.get('recommend_defer', 0) or 0)}",
        "",
        "## Quick Start",
        "",
        f"1. 打开 `{summary.get('review_sheet_path', '')}`。",
        "2. 只改 `human_review_status` / `human_review_note`；如果是 `edit`，再顺手补对应 `override_*` 字段。",
        "3. 运行 `python -m scripts.apply_learning_intake_human_review_sheet`。",
        "4. 如需继续验证，再运行 `python -m scripts.export_learning_intake_candidate_import`。",
        "",
        "## Candidates",
        "",
    ]
    for index, row in enumerate(rows, start=1):
        if not isinstance(row, dict):
            continue
        lines.extend(
            [
                f"### {index}. {row.get('sample_id') or 'manual-review'}",
                f"- suggested_decision: {row.get('suggested_decision', '')}",
                f"- suggested_review_status: {row.get('suggested_review_status', '')}",
                f"- priority_score: {row.get('priority_score', 0)}",
                f"- current_review_status: {row.get('current_review_status', '')}",
                f"- why: {'; '.join(row.get('suggested_why', []))}",
                f"- prompt: {row.get('user_input', '')}",
                "",
            ]
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def build_learning_intake_human_review_panel(root: Path = Path(".")) -> dict[str, object]:
    review_pack_path = root / "data" / "learning" / "candidate_pretrain_review_pack.jsonl"
    packet_path = root / "artifacts" / "learning_intake_human_gate_packet_latest.json"
    sheet_path = root / "data" / "learning" / "candidate_pretrain_human_review_sheet.jsonl"
    artifact_path = root / "artifacts" / "learning_intake_human_review_panel_latest.json"
    report_path = root / "backlog" / "opportunities" / "learning_intake_human_review_panel.md"

    review_pack = load_review_feedback(review_pack_path)
    packet = _read_json(packet_path)
    decisions = packet.get("decisions", []) if isinstance(packet, dict) else []
    if not isinstance(decisions, list):
        decisions = []

    decisions_by_prompt = {
        str(item.get("prompt", "")).strip(): item
        for item in decisions
        if isinstance(item, dict) and str(item.get("prompt", "")).strip()
    }

    sheet_rows: list[dict[str, object]] = []
    panel_rows: list[dict[str, object]] = []

    for payload in review_pack:
        prompt = str(payload.get("user_input", "")).strip()
        decision = decisions_by_prompt.get(prompt, {})
        suggested_review_status = str(decision.get("proposed_review_status", "pending")).strip() or "pending"
        suggested_decision = str(decision.get("decision", "review")).strip() or "review"
        suggested_why = decision.get("why", [])
        if not isinstance(suggested_why, list):
            suggested_why = []
        sample_id = str(decision.get("sample_id", "")).strip() or _extract_sample_id(prompt) or _fallback_sample_id(payload, prompt)
        current_review_status = str(payload.get("review_status", "pending")).strip() or "pending"

        sheet_row = dict(payload)
        sheet_row["sample_id"] = sample_id
        sheet_row["suggested_decision"] = suggested_decision
        sheet_row["suggested_review_status"] = suggested_review_status
        sheet_row["suggested_review_note"] = " | ".join(str(item).strip() for item in suggested_why if str(item).strip())
        sheet_row["suggested_why"] = suggested_why
        sheet_row["priority_score"] = float(decision.get("priority_score", 0.0) or 0.0)
        sheet_row["human_review_status"] = ""
        sheet_row["human_review_note"] = ""
        sheet_rows.append(sheet_row)

        panel_rows.append(
            {
                "sample_id": sample_id,
                "user_input": prompt,
                "source_type": str(payload.get("source_type", "")).strip(),
                "current_review_status": current_review_status,
                "suggested_decision": suggested_decision,
                "suggested_review_status": suggested_review_status,
                "priority_score": float(decision.get("priority_score", 0.0) or 0.0),
                "suggested_why": suggested_why,
            }
        )

    top_row = panel_rows[0] if panel_rows else {}
    summary = {
        "mode": "human_friendly_review_panel",
        "total_candidates": len(panel_rows),
        "recommend_accept": sum(1 for row in panel_rows if row["suggested_review_status"] == "accept"),
        "recommend_edit": sum(1 for row in panel_rows if row["suggested_review_status"] == "edit"),
        "recommend_defer": sum(1 for row in panel_rows if row["suggested_decision"] == "defer"),
        "top_priority_sample_id": str(top_row.get("sample_id", "")),
        "top_priority_source_type": str(top_row.get("source_type", "")),
        "top_priority_review_status": str(top_row.get("suggested_review_status", "")),
        "review_sheet_path": str(sheet_path),
        "artifact_path": str(artifact_path),
        "report_path": str(report_path),
        "default_training_admission_changed": False,
    }
    panel = {"summary": summary, "rows": panel_rows}

    _write_jsonl(sheet_path, sheet_rows)
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_text(json.dumps(panel, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _write_report(report_path, panel)
    return panel


def main() -> int:
    panel = build_learning_intake_human_review_panel()
    print(json.dumps(panel, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
