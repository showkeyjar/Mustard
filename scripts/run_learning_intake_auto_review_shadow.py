from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from carm.pretrain_data import (
    export_review_pack,
    load_pretrain_samples,
    load_review_feedback,
    merge_and_filter_samples,
    review_payload_to_sample,
    save_pretrain_samples,
)
from carm.training import load_training_config


def _read_json(path: Path) -> dict[str, object]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _write_report(path: Path, payload: dict[str, object]) -> None:
    lines = [
        "# Learning Intake Auto Review Shadow",
        "",
        "- mode: auto_review_shadow",
        "- default_runtime_changed: false",
        "- default_training_admission_changed: false",
        f"- review_total: {int(payload.get('review_total', 0) or 0)}",
        f"- auto_accept_count: {int(payload.get('auto_accept_count', 0) or 0)}",
        f"- auto_edit_count: {int(payload.get('auto_edit_count', 0) or 0)}",
        f"- auto_pending_count: {int(payload.get('auto_pending_count', 0) or 0)}",
        f"- auto_import_count: {int(payload.get('auto_import_count', 0) or 0)}",
        f"- shadow_sample_count: {int(payload.get('shadow_sample_count', 0) or 0)}",
        f"- top_priority_sample_id: {payload.get('top_priority_sample_id', '')}",
        "",
        "## Paths",
        "",
        f"- auto_review_pack_path: {payload.get('auto_review_pack_path', '')}",
        f"- auto_import_path: {payload.get('auto_import_path', '')}",
        f"- auto_shadow_corpus_path: {payload.get('auto_shadow_corpus_path', '')}",
        "",
        "## Notes",
        "",
        "- 这是自动审阅的影子通道，不会改正式 candidate_pretrain_review_pack.jsonl。",
        "- 只有 packet 里建议为 accept/edit 的候选会进入 auto import 和 auto shadow corpus。",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def run_learning_intake_auto_review_shadow(root: Path = Path(".")) -> dict[str, object]:
    review_pack_path = root / "data" / "learning" / "candidate_pretrain_review_pack.jsonl"
    packet_path = root / "artifacts" / "learning_intake_human_gate_packet_latest.json"
    config = load_training_config(str(root / "configs" / "training.yaml"))
    pretraining = config.get("training", {}).get("pretraining", {})

    base_dataset_path = root / Path(str(pretraining.get("dataset_path", "data/pretrain/pretrain_corpus.jsonl")))
    min_quality_score = float(pretraining.get("min_quality_score", 0.72))
    max_samples = int(pretraining.get("max_dataset_samples", 5000))

    review_rows = load_review_feedback(review_pack_path)
    packet = _read_json(packet_path)
    decisions = packet.get("decisions", []) if isinstance(packet, dict) else []
    if not isinstance(decisions, list):
        decisions = []
    decision_by_prompt = {
        str(item.get("prompt", "")).strip(): item
        for item in decisions
        if isinstance(item, dict) and str(item.get("prompt", "")).strip()
    }

    auto_review_rows: list[dict[str, object]] = []
    auto_import_rows: list[dict[str, object]] = []
    approved_samples = []
    status_counts: Counter[str] = Counter()

    for row in review_rows:
        payload = dict(row)
        prompt = str(payload.get("user_input", "")).strip()
        decision = decision_by_prompt.get(prompt, {})
        proposed_review_status = str(decision.get("proposed_review_status", "pending")).strip().lower() or "pending"
        payload["auto_review_status"] = proposed_review_status
        payload["auto_review_note"] = f"auto_shadow:{str(decision.get('decision', 'defer')).strip() or 'defer'}"
        payload["auto_priority_score"] = float(decision.get("priority_score", 0.0) or 0.0)
        auto_review_rows.append(payload)
        status_counts[proposed_review_status] += 1

        if proposed_review_status not in {"accept", "edit"}:
            continue
        approved_payload = dict(payload)
        approved_payload["review_status"] = proposed_review_status
        approved_payload["review_note"] = payload["auto_review_note"]
        sample = review_payload_to_sample(approved_payload)
        approved_samples.append(sample)
        auto_import_rows.append(
            {
                "prompt": sample.user_input,
                "source_type": sample.source_type,
                "logic_skill": sample.logic_skill,
                "quality_score": sample.quality_score,
                "metadata": {
                    **sample.metadata,
                    "review_status": proposed_review_status,
                    "candidate_source_type": str(row.get("source_type", "")),
                    "auto_review_shadow": True,
                },
            }
        )

    base_samples = load_pretrain_samples(base_dataset_path)
    shadow_samples = merge_and_filter_samples(
        base_samples + approved_samples,
        min_quality_score=min_quality_score,
        max_samples=max_samples,
    )

    output_dir = root / "data" / "learning"
    auto_review_pack_path = output_dir / "candidate_pretrain_auto_review_shadow.jsonl"
    auto_import_path = output_dir / "candidate_pretrain_auto_import_shadow.jsonl"
    auto_shadow_corpus_path = output_dir / "candidate_pretrain_auto_shadow_corpus.jsonl"
    auto_shadow_review_pack_path = output_dir / "candidate_pretrain_auto_shadow_review_pack.jsonl"
    artifact_path = root / "artifacts" / "learning_intake_auto_review_shadow_latest.json"
    report_path = root / "backlog" / "opportunities" / "learning_intake_auto_review_shadow.md"

    _write_jsonl(auto_review_pack_path, auto_review_rows)
    _write_jsonl(auto_import_path, auto_import_rows)
    save_pretrain_samples(auto_shadow_corpus_path, shadow_samples)
    export_review_pack(auto_shadow_review_pack_path, approved_samples, limit=min(50, len(approved_samples)))

    top_decision = decisions[0] if decisions else {}
    payload = {
        "mode": "auto_review_shadow",
        "review_total": len(review_rows),
        "auto_accept_count": int(status_counts.get("accept", 0)),
        "auto_edit_count": int(status_counts.get("edit", 0)),
        "auto_pending_count": int(status_counts.get("pending", 0)),
        "auto_reject_count": int(status_counts.get("reject", 0)),
        "auto_import_count": len(auto_import_rows),
        "base_sample_count": len(base_samples),
        "shadow_sample_count": len(shadow_samples),
        "top_priority_sample_id": str(top_decision.get("sample_id", "")),
        "top_priority_decision": str(top_decision.get("decision", "")),
        "auto_review_pack_path": str(auto_review_pack_path),
        "auto_import_path": str(auto_import_path),
        "auto_shadow_corpus_path": str(auto_shadow_corpus_path),
        "auto_shadow_review_pack_path": str(auto_shadow_review_pack_path),
        "artifact_path": str(artifact_path),
        "report_path": str(report_path),
        "default_training_admission_changed": False,
    }
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _write_report(report_path, payload)
    return payload


def main() -> int:
    payload = run_learning_intake_auto_review_shadow()
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
