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


def _extract_sample_id(text: str) -> str:
    match = SAMPLE_ID_PATTERN.search(text)
    return match.group(1).strip() if match else ""


def _build_learning_focus_index(root: Path) -> dict[str, dict[str, object]]:
    payload = _read_json(root / "data" / "evolution" / "learning_focus_evidence_routing_eval_result.json")
    rows = payload.get("rows", []) if isinstance(payload, dict) else []
    if not isinstance(rows, list):
        return {}
    result: dict[str, dict[str, object]] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        row_id = str(row.get("id", "")).strip()
        if row_id:
            result[row_id] = row
    return result


def _recommend_row(
    item: dict[str, object],
    learning_focus_index: dict[str, dict[str, object]],
    attention_summary: dict[str, object],
) -> dict[str, object]:
    source_type = str(item.get("source_type", "")).strip()
    user_input = str(item.get("user_input", "")).strip()
    quality_score = float(item.get("quality_score", 0.0) or 0.0)
    priority = round(quality_score * 100, 2)
    recommended_status = "hold"
    reasons: list[str] = []
    sample_id = ""

    if source_type == "learning_intake:learning_focus_stress":
        sample_id = _extract_sample_id(user_input)
        row = learning_focus_index.get(sample_id, {})
        pretrained_match = bool(row.get("pretrained_match", False))
        baseline_match = bool(row.get("baseline_match", False))
        pretrained_tool = str(row.get("pretrained_used_tool", "")).strip()
        baseline_tool = str(row.get("baseline_used_tool", "")).strip()
        if not pretrained_match:
            recommended_status = "accept"
            priority += 25
            reasons.append("pretrained 仍失败，说明这是当前真实缺口的直接监督候选。")
        elif not baseline_match:
            recommended_status = "edit"
            priority += 12
            reasons.append("baseline 失败但 pretrained 已修复，适合保留并压缩成更尖锐样本。")
        else:
            recommended_status = "hold"
            reasons.append("baseline/pretrained 都已匹配，优先级较低。")
        if pretrained_tool:
            reasons.append(f"pretrained_used_tool={pretrained_tool}")
        if baseline_tool:
            reasons.append(f"baseline_used_tool={baseline_tool}")

    elif source_type == "learning_intake:attention_gap":
        premature_release_count = int(attention_summary.get("premature_release_count", 0) or 0)
        conflict_to_verification_rate = float(attention_summary.get("conflict_to_verification_rate", 1.0) or 0.0)
        if premature_release_count > 0 or conflict_to_verification_rate < 0.5:
            recommended_status = "accept"
            priority += 18
            reasons.append("attention handoff 仍是当前 top gap，适合进入候选训练审阅。")
            reasons.append(
                f"premature_release_count={premature_release_count}, conflict_to_verification_rate={conflict_to_verification_rate:.4f}"
            )
        else:
            recommended_status = "hold"
            reasons.append("attention 指标已接近目标，暂不优先。")
    else:
        reasons.append("当前脚本只对 learning_focus_stress / attention_gap 给出显式优先级建议。")

    return {
        "user_input": user_input,
        "source_type": source_type,
        "logic_skill": str(item.get("logic_skill", "")).strip(),
        "current_review_status": str(item.get("review_status", "pending")).strip() or "pending",
        "recommended_status": recommended_status,
        "priority_score": round(priority, 2),
        "sample_id": sample_id,
        "reasons": reasons,
    }


def _write_report(path: Path, queue: list[dict[str, object]], summary: dict[str, object]) -> None:
    lines = [
        "# Learning Intake Review Queue",
        "",
        f"- queue_count: {int(summary.get('queue_count', 0) or 0)}",
        f"- recommend_accept: {int(summary.get('recommend_accept', 0) or 0)}",
        f"- recommend_edit: {int(summary.get('recommend_edit', 0) or 0)}",
        f"- recommend_hold: {int(summary.get('recommend_hold', 0) or 0)}",
        "- default_training_admission_changed: false",
        "",
    ]
    for item in queue:
        lines.extend(
            [
                f"## {item.get('source_type', '')}",
                f"- recommended_status: {item.get('recommended_status', '')}",
                f"- priority_score: {item.get('priority_score', 0)}",
                f"- sample_id: {item.get('sample_id', '')}",
                f"- reasons: {'; '.join(item.get('reasons', []))}",
                f"- prompt: {item.get('user_input', '')}",
                "",
            ]
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def build_learning_intake_review_queue(root: Path = Path(".")) -> dict[str, object]:
    review_pack = load_review_feedback(root / "data" / "learning" / "candidate_pretrain_review_pack.jsonl")
    learning_focus_index = _build_learning_focus_index(root)
    attention_flow = _read_json(root / "artifacts" / "attention_flow_latest.json")
    attention_views = _read_json(root / "artifacts" / "attention_training_views_latest.json")
    attention_flow_summary = attention_flow.get("summary", {}) if isinstance(attention_flow, dict) else {}
    attention_view_summary = attention_views.get("summary", {}) if isinstance(attention_views, dict) else {}
    attention_summary = {}
    if isinstance(attention_flow_summary, dict):
        attention_summary.update(attention_flow_summary)
    if isinstance(attention_view_summary, dict):
        attention_summary.update(attention_view_summary)

    queue = [
        _recommend_row(item, learning_focus_index, attention_summary)
        for item in review_pack
        if isinstance(item, dict)
    ]
    queue.sort(key=lambda item: (item["priority_score"], item["recommended_status"] == "accept"), reverse=True)

    summary = {
        "queue_count": len(queue),
        "recommend_accept": sum(1 for item in queue if item["recommended_status"] == "accept"),
        "recommend_edit": sum(1 for item in queue if item["recommended_status"] == "edit"),
        "recommend_hold": sum(1 for item in queue if item["recommended_status"] == "hold"),
        "artifact_path": str(root / "artifacts" / "learning_intake_review_queue_latest.json"),
        "report_path": str(root / "backlog" / "opportunities" / "learning_intake_review_queue.md"),
        "default_training_admission_changed": False,
    }

    artifact_payload = {"summary": summary, "queue": queue}
    artifact_path = Path(summary["artifact_path"])
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_text(json.dumps(artifact_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _write_report(Path(summary["report_path"]), queue, summary)
    return summary


def main() -> int:
    summary = build_learning_intake_review_queue()
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
