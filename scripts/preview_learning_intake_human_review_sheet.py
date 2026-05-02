from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from carm.pretrain_data import load_review_feedback


VALID_REVIEW_STATUSES = {"pending", "accept", "edit", "reject"}


def _fallback_sample_id(row: dict[str, object], prompt: str) -> str:
    source_type = str(row.get("source_type", "")).strip()
    if source_type == "learning_intake:attention_gap":
        return "attention-gap-001"
    if source_type:
        return source_type.replace(":", "-")
    return prompt[:48]


def _build_next_action(summary: dict[str, object]) -> dict[str, object]:
    missing_candidates = summary.get("missing_candidates", [])
    if not isinstance(missing_candidates, list):
        missing_candidates = []
    actionable_decision_count = int(summary.get("actionable_decision_count", 0) or 0)
    blank_decision_count = int(summary.get("blank_decision_count", 0) or 0)
    invalid_decision_count = int(summary.get("invalid_decision_count", 0) or 0)

    if invalid_decision_count > 0:
        return {
            "state": "blocked_invalid_status",
            "message": "先修正非法的 human_review_status，再继续应用。",
            "command": "python -m scripts.claw_team_control preview-human-review",
        }
    if blank_decision_count > 0:
        if len(missing_candidates) == 1:
            candidate = missing_candidates[0] if isinstance(missing_candidates[0], dict) else {}
            sample_id = str(candidate.get("sample_id", "")).strip()
            suggested_decision = str(candidate.get("suggested_decision", "")).strip().lower()
            suggested_review_status = str(candidate.get("suggested_review_status", "")).strip().lower()
            suggested_status = "defer" if suggested_decision == "defer" or suggested_review_status == "pending" else suggested_review_status
            if sample_id and suggested_status:
                return {
                    "state": "needs_one_more_review",
                    "message": "还差最后 1 条，已经给出快捷决策命令。",
                    "command": f"python -m scripts.claw_team_control decide-human-review --sample-id {sample_id} --status {suggested_status}",
                }
        top_candidate = {}
        if missing_candidates:
            top_candidate = max(
                [item for item in missing_candidates if isinstance(item, dict)],
                key=lambda item: float(item.get("priority_score", 0.0) or 0.0),
                default={},
            )
        top_sample_id = str(top_candidate.get("sample_id", "")).strip()
        if top_sample_id:
            return {
                "state": "needs_more_review",
                "message": "还有候选没填，先处理最高优先级那条。",
                "command": "python -m scripts.claw_team_control decide-top-human-review",
            }
        return {
            "state": "needs_more_review",
            "message": "还有候选没填，先补完 human_review_status。",
            "command": "python -m scripts.claw_team_control human-review",
        }
    if actionable_decision_count > 0:
        return {
            "state": "ready_to_apply",
            "message": "当前表单已经可应用，可以导出正式 import 候选。",
            "command": "python -m scripts.claw_team_control apply-human-review --export-import",
        }
    if int(summary.get("ready_to_apply_count", 0) or 0) > 0:
        return {
            "state": "already_applied",
            "message": "当前 human review sheet 已和 review pack 对齐，可以继续查看 import 候选。",
            "command": "python -m scripts.claw_team_control status",
        }
    return {
        "state": "idle",
        "message": "当前还没有新的人工决定可应用。",
        "command": "python -m scripts.claw_team_control human-review",
    }


def _write_report(path: Path, payload: dict[str, object]) -> None:
    summary = payload.get("summary", {})
    decisions = payload.get("decisions", [])
    next_action = payload.get("next_action", {})
    missing_samples = payload.get("missing_samples", [])
    invalid_samples = payload.get("invalid_samples", [])
    lines = [
        "# Learning Intake Human Review Preview",
        "",
        "- mode: preview_human_review_sheet",
        "- default_runtime_changed: false",
        "- default_training_admission_changed: false",
        f"- total_candidates: {int(summary.get('total_candidates', 0) or 0)}",
        f"- ready_to_apply_count: {int(summary.get('ready_to_apply_count', 0) or 0)}",
        f"- actionable_decision_count: {int(summary.get('actionable_decision_count', 0) or 0)}",
        f"- blank_decision_count: {int(summary.get('blank_decision_count', 0) or 0)}",
        f"- invalid_decision_count: {int(summary.get('invalid_decision_count', 0) or 0)}",
        f"- would_accept_count: {int(summary.get('would_accept_count', 0) or 0)}",
        f"- would_edit_count: {int(summary.get('would_edit_count', 0) or 0)}",
        f"- would_reject_count: {int(summary.get('would_reject_count', 0) or 0)}",
        "",
        "## Next Step",
        "",
        f"- state: {next_action.get('state', '')}",
        f"- message: {next_action.get('message', '')}",
        f"- command: `{next_action.get('command', '')}`",
        "",
    ]
    if missing_samples:
        lines.extend(["## Missing Decisions", ""])
        for item in missing_samples:
            lines.append(f"- {item}")
        lines.append("")
    if invalid_samples:
        lines.extend(["## Invalid Decisions", ""])
        for item in invalid_samples:
            lines.append(f"- {item}")
        lines.append("")
    lines.extend(
        [
        "## Decisions",
        "",
        ]
    )
    for item in decisions:
        lines.extend(
            [
                f"### {item.get('sample_id') or 'manual-review'}",
                f"- requested_status: {item.get('requested_status', '')}",
                f"- current_review_status: {item.get('current_review_status', '')}",
                f"- suggested_review_status: {item.get('suggested_review_status', '')}",
                f"- note: {item.get('human_review_note', '')}",
                f"- prompt: {item.get('user_input', '')}",
                "",
            ]
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def preview_learning_intake_human_review_sheet(root: Path = Path(".")) -> dict[str, object]:
    review_sheet_path = root / "data" / "learning" / "candidate_pretrain_human_review_sheet.jsonl"
    review_pack_path = root / "data" / "learning" / "candidate_pretrain_review_pack.jsonl"
    artifact_path = root / "artifacts" / "learning_intake_human_review_preview_latest.json"
    report_path = root / "backlog" / "opportunities" / "learning_intake_human_review_preview.md"

    review_sheet = load_review_feedback(review_sheet_path)
    review_pack = load_review_feedback(review_pack_path)
    current_by_prompt = {
        str(row.get("user_input", "")).strip(): str(row.get("review_status", "pending")).strip().lower() or "pending"
        for row in review_pack
        if isinstance(row, dict) and str(row.get("user_input", "")).strip()
    }

    decisions: list[dict[str, object]] = []
    missing_samples: list[str] = []
    missing_candidates: list[dict[str, object]] = []
    invalid_samples: list[str] = []
    blank_decision_count = 0
    invalid_decision_count = 0
    actionable_decision_count = 0
    requested_counter: Counter[str] = Counter()

    for row in review_sheet:
        prompt = str(row.get("user_input", "")).strip()
        if not prompt:
            continue
        sample_id = str(row.get("sample_id", "")).strip() or _fallback_sample_id(row, prompt)
        requested_status = str(row.get("human_review_status", "")).strip().lower()
        if not requested_status:
            blank_decision_count += 1
            missing_samples.append(sample_id)
            missing_candidates.append(
                {
                    "sample_id": sample_id,
                    "suggested_decision": str(row.get("suggested_decision", "")).strip(),
                    "suggested_review_status": str(row.get("suggested_review_status", "")).strip(),
                    "priority_score": float(row.get("priority_score", 0.0) or 0.0),
                }
            )
            continue
        if requested_status not in VALID_REVIEW_STATUSES:
            invalid_decision_count += 1
            invalid_samples.append(f"{sample_id}: {requested_status}")
        requested_counter[requested_status] += 1
        if requested_status != current_by_prompt.get(prompt, "pending"):
            actionable_decision_count += 1
        decisions.append(
            {
                "sample_id": sample_id,
                "user_input": prompt,
                "requested_status": requested_status,
                "current_review_status": current_by_prompt.get(prompt, "pending"),
                "suggested_review_status": str(row.get("suggested_review_status", "")).strip(),
                "human_review_note": str(row.get("human_review_note", "")).strip(),
            }
        )

    summary = {
        "mode": "preview_human_review_sheet",
        "total_candidates": len(review_sheet),
        "ready_to_apply_count": len(decisions),
        "actionable_decision_count": actionable_decision_count,
        "blank_decision_count": blank_decision_count,
        "invalid_decision_count": invalid_decision_count,
        "would_accept_count": int(requested_counter.get("accept", 0)),
        "would_edit_count": int(requested_counter.get("edit", 0)),
        "would_reject_count": int(requested_counter.get("reject", 0)),
        "would_pending_count": int(requested_counter.get("pending", 0)),
        "missing_candidates": missing_candidates,
        "artifact_path": str(artifact_path),
        "report_path": str(report_path),
        "default_training_admission_changed": False,
    }
    payload = {
        "summary": summary,
        "next_action": _build_next_action(summary),
        "missing_samples": missing_samples,
        "invalid_samples": invalid_samples,
        "decisions": decisions,
    }

    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _write_report(report_path, payload)
    return payload


def main() -> int:
    payload = preview_learning_intake_human_review_sheet()
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
