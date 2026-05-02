from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from carm.pretrain_data import (
    PretrainSample,
    annotate_sample,
    build_samples_from_experience,
    export_review_pack,
    merge_and_filter_samples,
    save_pretrain_samples,
)


def _read_json(path: Path) -> dict[str, object]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _load_jsonl(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []
    items: list[dict[str, object]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            items.append(payload)
    return items


def _collect_experience_samples(root: Path, limit: int) -> list[PretrainSample]:
    experience_path = root / "data" / "experience" / "episodes.jsonl"
    if not experience_path.exists():
        return []
    samples = build_samples_from_experience(experience_path, min_value_score=0.72, limit=limit)
    for sample in samples:
        sample.source_type = "learning_intake:experience"
        sample.metadata = dict(sample.metadata)
        sample.metadata["learning_source"] = "experience"
    return samples


def _collect_bridge_feedback_samples(root: Path, limit: int) -> list[PretrainSample]:
    events = _load_jsonl(root / "data" / "desktop" / "bridge_events.jsonl")
    feedback = _load_jsonl(root / "data" / "desktop" / "bridge_feedback.jsonl")
    event_by_id = {str(item.get("event_id", "")): item for item in events if str(item.get("event_id", "")).strip()}
    samples: list[PretrainSample] = []
    for item in feedback[-limit:]:
        feedback_type = str(item.get("feedback_type", "")).strip()
        if feedback_type not in {"useful", "misread"}:
            continue
        event_id = str(item.get("event_id", "")).strip()
        event = event_by_id.get(event_id, {})
        summary = str(event.get("summary", "") or "")
        prompt = str(event.get("prompt", "") or "")
        note = str(item.get("note", "") or "").strip()
        if not (summary or prompt or note):
            continue
        user_input = (
            f"桌面纠偏学习：场景={summary or prompt}。"
            f" 原始交互={prompt or summary}。"
            f" 用户反馈={feedback_type}。"
            f" 纠偏说明={note or '无'}。"
            " 下次应优先关注哪些证据、工具边界或追问条件？"
        )
        quality = 0.9 if feedback_type == "useful" else 0.86
        sample = annotate_sample(
            PretrainSample(
                user_input=user_input,
                task_type="fact_check",
                source_type="learning_intake:bridge_feedback",
                expected_tool="",
                target_slot="",
                logic_skill="evidence_judgment" if feedback_type == "misread" else "step_planning",
                quality_score=quality,
                metadata={
                    "learning_source": "bridge_feedback",
                    "feedback_type": feedback_type,
                    "event_id": event_id,
                    "forced_expected_tool": "search",
                    "forced_target_slot": "PLAN" if feedback_type == "useful" else "HYP",
                },
            )
        )
        samples.append(sample)
    return samples


def _collect_frontier_samples(root: Path, limit: int) -> list[PretrainSample]:
    rows = _load_jsonl(root / "data" / "research" / "frontier_observations.jsonl")
    samples: list[PretrainSample] = []
    for item in rows[-limit:]:
        topic = str(item.get("topic", "")).strip()
        if not topic:
            continue
        label = str(item.get("label", "pending_label")).strip()
        reason = str(item.get("reason", "")).strip()
        user_input = (
            f"公开 agent 设计学习：主题={topic}。"
            f" 当前标签={label}。"
            f" 触发原因={reason or '研究跟踪'}。"
            " 请基于公开资料总结可借鉴/不建议/待观察要点，并给出一个能在 Mustard 里低风险验证的实验。"
        )
        quality = 0.82 if label != "pending_label" else 0.76
        sample = annotate_sample(
            PretrainSample(
                user_input=user_input,
                task_type="fact_check",
                source_type="learning_intake:frontier",
                expected_tool="",
                target_slot="",
                logic_skill="evidence_judgment",
                quality_score=quality,
                metadata={
                    "learning_source": "frontier_observation",
                    "topic": topic,
                    "label": label,
                    "reason": reason,
                    "forced_expected_tool": "search",
                    "forced_target_slot": "PLAN",
                },
            )
        )
        samples.append(sample)
    return samples


def _collect_public_idea_samples(root: Path, limit: int) -> list[PretrainSample]:
    rows = _load_jsonl(root / "data" / "research" / "public_agent_ideas.jsonl")
    samples: list[PretrainSample] = []
    for item in rows[:limit]:
        topic = str(item.get("topic", "")).strip()
        insight = str(item.get("insight", "")).strip()
        source_name = str(item.get("source_name", "")).strip()
        source_url = str(item.get("source_url", "")).strip()
        if not (topic and insight):
            continue
        user_input = (
            f"公开 agent 设计思想内化：主题={topic}。"
            f" 外部观点={insight}。"
            f" 来源={source_name or source_url or 'public'}。"
            " 请把它转写成一条适合 Mustard 离线评测或监督学习的任务，并说明验证通过阈值。"
        )
        sample = annotate_sample(
            PretrainSample(
                user_input=user_input,
                task_type="planning",
                source_type="learning_intake:public_idea",
                expected_tool="",
                target_slot="",
                logic_skill="result_integration",
                quality_score=0.84,
                metadata={
                    "learning_source": "public_agent_idea",
                    "topic": topic,
                    "source_name": source_name,
                    "source_url": source_url,
                    "forced_expected_tool": "search",
                    "forced_target_slot": "DRAFT",
                },
            )
        )
        samples.append(sample)
    return samples


def _collect_attention_gap_samples(root: Path) -> list[PretrainSample]:
    attention_flow_payload = _read_json(root / "artifacts" / "attention_flow_latest.json")
    attention_views_payload = _read_json(root / "artifacts" / "attention_training_views_latest.json")
    flow = attention_flow_payload.get("summary", {}) if isinstance(attention_flow_payload, dict) else {}
    views = attention_views_payload.get("summary", {}) if isinstance(attention_views_payload, dict) else {}
    if not isinstance(flow, dict) or not isinstance(views, dict):
        return []

    premature_release_count = int(flow.get("premature_release_count", 0) or 0)
    conflict_to_verification_rate = float(views.get("conflict_to_verification_rate", 1.0) or 0.0)
    tool_boundary_block_rate = float(views.get("tool_boundary_block_rate", 1.0) or 0.0)
    if premature_release_count <= 0 and conflict_to_verification_rate >= 0.5:
        return []

    prompts = [
        (
            "attention_verification_handoff",
            "当前发现 premature release 或 conflict->verification handoff 偏弱。请设计 2 条离线监督任务，要求残差必须先流向 VERIFY，之后才能 release。",
            "conflict_detection",
        ),
        (
            "attention_tool_boundary_handoff",
            "当前工具边界相关任务存在 release 风险。请设计 2 条 mixed numeric/code 的离线监督任务，要求先消解 tool boundary 再回答。",
            "tool_selection",
        ),
    ]
    samples: list[PretrainSample] = []
    for tag, body, logic_skill in prompts:
        if tag == "attention_tool_boundary_handoff" and tool_boundary_block_rate >= 1.0:
            continue
        user_input = (
            f"AttentionFlow 学习任务：premature_release_count={premature_release_count}，"
            f"conflict_to_verification_rate={conflict_to_verification_rate:.4f}。"
            f" {body}"
        )
        samples.append(
            annotate_sample(
                PretrainSample(
                    user_input=user_input,
                    task_type="planning",
                    source_type="learning_intake:attention_gap",
                    expected_tool="",
                    target_slot="",
                    logic_skill=logic_skill,
                    quality_score=0.9,
                    metadata={
                        "learning_source": "attention_gap",
                        "tag": tag,
                        "forced_expected_tool": "search",
                        "forced_target_slot": "PLAN",
                    },
                )
            )
        )
    return samples


def _collect_learning_focus_stress_samples(root: Path, limit: int) -> list[PretrainSample]:
    payload = _read_json(root / "data" / "evolution" / "learning_focus_evidence_routing_eval_result.json")
    rows = payload.get("rows", []) if isinstance(payload, dict) else []
    if not isinstance(rows, list):
        return []

    samples: list[PretrainSample] = []
    for item in rows[:limit]:
        if not isinstance(item, dict):
            continue
        expected_tool = str(item.get("expected_tool", "")).strip()
        pretrained_tool = str(item.get("pretrained_used_tool", "")).strip()
        baseline_tool = str(item.get("baseline_used_tool", "")).strip()
        logic_skill = str(item.get("logic_skill", "")).strip() or "evidence_judgment"
        row_id = str(item.get("id", "")).strip()
        if expected_tool != "search" or logic_skill != "evidence_judgment":
            continue
        if pretrained_tool == "search" and baseline_tool == "search":
            continue

        user_input = (
            f"Learning-focus routing stress 学习：样本={row_id or 'unknown'}。"
            f" 期望工具={expected_tool}，baseline={baseline_tool or 'unknown'}，pretrained={pretrained_tool or 'unknown'}。"
            " 请把这个 evidence_judgment 误路由场景转成一条更稳健的离线监督任务，要求先检索公开证据，再判断是否能回答。"
        )
        quality = 0.92 if pretrained_tool != "search" else 0.86
        samples.append(
            annotate_sample(
                PretrainSample(
                    user_input=user_input,
                    task_type="fact_check",
                    source_type="learning_intake:learning_focus_stress",
                    expected_tool="",
                    target_slot="",
                    logic_skill="evidence_judgment",
                    quality_score=quality,
                    metadata={
                        "learning_source": "learning_focus_stress",
                        "row_id": row_id,
                        "baseline_used_tool": baseline_tool,
                        "pretrained_used_tool": pretrained_tool,
                        "forced_expected_tool": "search",
                        "forced_target_slot": "PLAN",
                    },
                )
            )
        )
    return samples


def _collect_search_first_adversarial_failure_samples(root: Path, limit: int) -> list[PretrainSample]:
    payload = _read_json(root / "artifacts" / "learning_focus_search_first_adversarial_latest.json")
    current_rows = payload.get("current_rows", []) if isinstance(payload, dict) else []
    shadow_rows = payload.get("shadow_rows", []) if isinstance(payload, dict) else []
    if not isinstance(current_rows, list) or not isinstance(shadow_rows, list):
        return []

    shadow_by_id = {
        str(item.get("id", "")).strip(): item
        for item in shadow_rows
        if isinstance(item, dict) and str(item.get("id", "")).strip()
    }
    eval_payload = _read_json(root / "data" / "eval" / "learning_focus_search_first_adversarial_eval.json")
    prompt_rows = eval_payload.get("prompts", []) if isinstance(eval_payload, dict) else []
    prompt_by_id = {
        str(item.get("id", "")).strip(): item
        for item in prompt_rows
        if isinstance(item, dict) and str(item.get("id", "")).strip()
    }

    samples: list[PretrainSample] = []
    for item in current_rows[:limit]:
        if not isinstance(item, dict):
            continue
        row_id = str(item.get("id", "")).strip()
        if not row_id:
            continue
        shadow_item = shadow_by_id.get(row_id, {})
        expected_tool = str(item.get("expected_tool", "")).strip()
        logic_skill = str(item.get("logic_skill", "")).strip()
        pretrained_tool = str(item.get("pretrained_used_tool", "")).strip()
        shadow_pretrained_tool = str(shadow_item.get("pretrained_used_tool", "")).strip()
        if expected_tool != "search" or logic_skill != "evidence_judgment":
            continue
        if bool(item.get("pretrained_match", True)) or bool(shadow_item.get("pretrained_match", True)):
            continue

        prompt_row = prompt_by_id.get(row_id, {})
        prompt = str(prompt_row.get("prompt", "")).strip()
        mutation = str(prompt_row.get("mutation", "")).strip()
        source_learning_focus_id = str(prompt_row.get("source_learning_focus_id", "")).strip()
        source_prompt = str(prompt_row.get("source_prompt", "")).strip()

        user_input = (
            f"Search-first adversarial failure 学习：样本={row_id}。"
            f" 当前与 shadow 都把 expected_tool=search 误路由成 {pretrained_tool or 'unknown'} / {shadow_pretrained_tool or 'unknown'}。"
            f" 原题变体={mutation or 'unknown'}。"
            f" 原始主题题目={source_prompt or source_learning_focus_id or 'unknown'}。"
            f" 请把这个失败改写成一条更稳健的 evidence_judgment 离线监督任务，要求先检索公开证据，再区分事实、引用和待验证假设。"
        )
        if prompt:
            user_input += f" 参考失败 prompt={prompt}"

        samples.append(
            annotate_sample(
                PretrainSample(
                    user_input=user_input,
                    task_type="fact_check",
                    source_type="learning_intake:search_first_adversarial_failure",
                    expected_tool="",
                    target_slot="",
                    logic_skill="evidence_judgment",
                    quality_score=0.96,
                    metadata={
                        "learning_source": "search_first_adversarial_failure",
                        "row_id": row_id,
                        "mutation": mutation,
                        "source_learning_focus_id": source_learning_focus_id,
                        "pretrained_used_tool": pretrained_tool,
                        "shadow_pretrained_used_tool": shadow_pretrained_tool,
                        "forced_expected_tool": "search",
                        "forced_target_slot": "PLAN",
                    },
                )
            )
        )
    return samples


def _write_import_tasks(path: Path, samples: list[PretrainSample]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for sample in samples:
            payload = {
                "prompt": sample.user_input,
                "source_type": sample.source_type,
                "logic_skill": sample.logic_skill,
                "quality_score": sample.quality_score,
                "metadata": sample.metadata,
            }
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _write_report(path: Path, samples: list[PretrainSample], source_counts: Counter[str]) -> None:
    lines = [
        "# Learning Intake Report",
        "",
        f"- candidate_count: {len(samples)}",
        f"- source_breakdown: {json.dumps(dict(source_counts), ensure_ascii=False)}",
        "- default_runtime_changed: false",
        "- default_training_admission_changed: false",
        "",
        "## Recommended Use",
        "",
        "- Step 1: 审阅 data/learning/learning_intake_review_pack.jsonl",
        "- Step 2: 如需将提示并入离线构建，使用环境变量 CARM_PRETRAIN_IMPORT_PATHS=data/learning/learning_intake_import.jsonl",
        "- Step 3: 运行 python -m scripts.build_pretrain_dataset 或 python -m scripts.auto_train",
        "",
        "## Top Candidates",
        "",
    ]
    for sample in samples[:10]:
        lines.extend(
            [
                f"### {sample.source_type}",
                f"- logic_skill: {sample.logic_skill}",
                f"- expected_tool: {sample.expected_tool}",
                f"- quality_score: {sample.quality_score:.4f}",
                f"- prompt: {sample.user_input}",
                "",
            ]
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def build_learning_intake(root: Path = Path("."), max_per_source: int = 12) -> dict[str, object]:
    output_dir = root / "data" / "learning"
    output_dir.mkdir(parents=True, exist_ok=True)

    collected = {
        "experience": _collect_experience_samples(root, max_per_source),
        "bridge_feedback": _collect_bridge_feedback_samples(root, max_per_source),
        "frontier": _collect_frontier_samples(root, max_per_source),
        "public_ideas": _collect_public_idea_samples(root, max_per_source),
        "attention_gap": _collect_attention_gap_samples(root),
        "learning_focus_stress": _collect_learning_focus_stress_samples(root, max_per_source),
        "search_first_adversarial_failure": _collect_search_first_adversarial_failure_samples(root, max_per_source),
    }
    all_samples = [sample for group in collected.values() for sample in group]
    merged = merge_and_filter_samples(all_samples, min_quality_score=0.72, max_samples=max_per_source * 8)

    samples_path = output_dir / "learning_intake_samples.jsonl"
    import_path = output_dir / "learning_intake_import.jsonl"
    review_pack_path = output_dir / "learning_intake_review_pack.jsonl"
    manifest_path = output_dir / "learning_intake_manifest.json"
    report_path = root / "backlog" / "opportunities" / "learning_intake_report.md"

    save_pretrain_samples(samples_path, merged)
    _write_import_tasks(import_path, merged)
    export_review_pack(review_pack_path, merged, limit=min(50, len(merged)))

    source_counts = Counter(sample.source_type for sample in merged)
    manifest = {
        "candidate_count": len(merged),
        "source_counts": dict(source_counts),
        "paths": {
            "samples": str(samples_path),
            "import_path": str(import_path),
            "review_pack": str(review_pack_path),
            "report": str(report_path),
        },
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_report(report_path, merged, source_counts)
    return manifest


def main() -> int:
    manifest = build_learning_intake()
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
