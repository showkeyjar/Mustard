from __future__ import annotations

import json
from pathlib import Path


def _read_json(path: Path) -> dict[str, object]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _write_report(path: Path, packet: dict[str, object]) -> None:
    summary = packet.get("summary", {})
    decisions = packet.get("decisions", [])
    lines = [
        "# Learning Intake Human Gate Packet",
        "",
        f"- recommend_approve_count: {int(summary.get('recommend_approve_count', 0) or 0)}",
        f"- recommend_edit_count: {int(summary.get('recommend_edit_count', 0) or 0)}",
        f"- recommend_defer_count: {int(summary.get('recommend_defer_count', 0) or 0)}",
        f"- expected_shadow_added_count: {int(summary.get('expected_shadow_added_count', 0) or 0)}",
        f"- deduped_candidate_count: {int(summary.get('deduped_candidate_count', 0) or 0)}",
        "- default_training_admission_changed: false",
        "",
        "## Recommended Actions",
        "",
    ]
    for item in decisions:
        lines.extend(
            [
                f"## {item.get('decision', '').upper()}",
                f"- candidate_source_type: {item.get('candidate_source_type', '')}",
                f"- sample_id: {item.get('sample_id', '')}",
                f"- priority_score: {item.get('priority_score', 0)}",
                f"- proposed_review_status: {item.get('proposed_review_status', '')}",
                f"- why: {'; '.join(item.get('why', []))}",
                f"- prompt: {item.get('prompt', '')}",
                "",
            ]
        )
    lines.extend(
        [
            "## Operator Note",
            "",
            "- 这份 packet 只给出建议，不会自动改 candidate review pack。",
            "- 若要正式放行，应由人类将对应 review_status 写回 `data/learning/candidate_pretrain_review_pack.jsonl`。",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def build_learning_intake_human_gate_packet(root: Path = Path(".")) -> dict[str, object]:
    queue_payload = _read_json(root / "artifacts" / "learning_intake_review_queue_latest.json")
    delta_payload = _read_json(root / "artifacts" / "learning_intake_suggested_shadow_delta_latest.json")

    queue = queue_payload.get("queue", []) if isinstance(queue_payload, dict) else []
    if not isinstance(queue, list):
        queue = []
    delta_summary = delta_payload.get("summary", {}) if isinstance(delta_payload, dict) else {}
    added_rows = delta_payload.get("added_rows", []) if isinstance(delta_payload, dict) else []
    if not isinstance(added_rows, list):
        added_rows = []
    deduped_rows = delta_payload.get("deduped_import_rows", []) if isinstance(delta_payload, dict) else []
    if not isinstance(deduped_rows, list):
        deduped_rows = []

    decisions: list[dict[str, object]] = []
    surviving_prompts = {str(item.get("user_input", "")).strip() for item in added_rows if isinstance(item, dict)}
    deduped_prompts = {str(item.get("prompt", "")).strip() for item in deduped_rows if isinstance(item, dict)}

    for item in queue:
        if not isinstance(item, dict):
            continue
        prompt = str(item.get("user_input", "")).strip()
        recommended_status = str(item.get("recommended_status", "")).strip().lower()
        candidate_source_type = str(item.get("source_type", "")).strip()
        decision = "defer"
        proposed_review_status = "pending"
        why = list(item.get("reasons", [])) if isinstance(item.get("reasons", []), list) else []

        if prompt in surviving_prompts and recommended_status == "accept":
            decision = "approve"
            proposed_review_status = "accept"
            why.append("影子重建确认该样本会形成真实新增。")
        elif prompt in surviving_prompts and recommended_status == "edit":
            decision = "approve_with_edit"
            proposed_review_status = "edit"
            why.append("影子重建确认该样本会形成真实新增，但更适合保留为 edit。")
        elif prompt in deduped_prompts:
            decision = "defer"
            proposed_review_status = "pending"
            why.append("影子重建显示该候选当前会被去重吞掉，暂不优先放行。")
        elif recommended_status == "accept":
            decision = "approve"
            proposed_review_status = "accept"
        elif recommended_status == "edit":
            decision = "approve_with_edit"
            proposed_review_status = "edit"

        decisions.append(
            {
                "decision": decision,
                "candidate_source_type": candidate_source_type,
                "sample_id": str(item.get("sample_id", "")),
                "priority_score": float(item.get("priority_score", 0.0) or 0.0),
                "proposed_review_status": proposed_review_status,
                "prompt": prompt,
                "why": why,
            }
        )

    top_decision = decisions[0] if decisions else {}
    summary = {
        "recommend_approve_count": sum(1 for item in decisions if item["decision"] == "approve"),
        "recommend_edit_count": sum(1 for item in decisions if item["decision"] == "approve_with_edit"),
        "recommend_defer_count": sum(1 for item in decisions if item["decision"] == "defer"),
        "expected_shadow_added_count": int(delta_summary.get("added_count", 0) or 0),
        "deduped_candidate_count": int(delta_summary.get("deduped_import_count", 0) or 0),
        "top_priority_sample_id": str(top_decision.get("sample_id", "")),
        "top_priority_source_type": str(top_decision.get("candidate_source_type", "")),
        "top_priority_decision": str(top_decision.get("decision", "")),
        "packet_path": str(root / "artifacts" / "learning_intake_human_gate_packet_latest.json"),
        "report_path": str(root / "backlog" / "opportunities" / "learning_intake_human_gate_packet.md"),
        "default_training_admission_changed": False,
    }
    packet = {"summary": summary, "decisions": decisions}
    packet_path = Path(summary["packet_path"])
    packet_path.parent.mkdir(parents=True, exist_ok=True)
    packet_path.write_text(json.dumps(packet, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _write_report(Path(summary["report_path"]), packet)
    return packet


def main() -> int:
    packet = build_learning_intake_human_gate_packet()
    print(json.dumps(packet, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
