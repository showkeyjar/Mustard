from __future__ import annotations

import json
from copy import deepcopy


def normalize_episode_payload(payload: dict) -> dict:
    normalized = deepcopy(payload)
    normalized["summary"] = normalize_summary(normalized.get("summary", ""))
    normalized["answer"] = normalize_answer_text(normalized.get("answer", ""))
    normalized["steps"] = [normalize_step_payload(step) for step in normalized.get("steps", [])]
    normalized["episode_features"] = normalize_episode_features(
        normalized.get("episode_features", {}),
        normalized.get("user_input", ""),
        normalized.get("summary", ""),
        normalized.get("steps", []),
    )
    normalized["outcome_signature"] = normalize_outcome_signature(
        normalized.get("outcome_signature", {}),
        normalized.get("success", False),
        normalized.get("value_score", 0.0),
        normalized.get("steps", []),
        normalized.get("answer", ""),
        normalized.get("summary", ""),
    )
    return normalized


def normalize_step_payload(step: dict) -> dict:
    normalized = dict(step)
    normalized.setdefault("user_input", "")
    normalized.setdefault("selected_tool", "")
    normalized.setdefault("target_slot", "")
    normalized.setdefault("feature_snapshot", {})
    normalized.setdefault("state_signature", infer_state_signature(normalized))
    normalized.setdefault("memory_signature", infer_memory_signature(normalized))
    normalized.setdefault("reward", 0.0)
    normalized.setdefault("reward_reason", infer_reward_reason(normalized))
    normalized.setdefault("high_value", False)
    normalized.setdefault("glance_used", bool(normalized.get("state_signature", {}).get("glance_trigger")))
    normalized.setdefault("glance_helped", bool(normalized.get("glance_used")) and float(normalized.get("reward", 0.0) or 0.0) > 0.0)
    return normalized


def normalize_summary(summary: str) -> str:
    if not isinstance(summary, str):
        return ""
    parts = [part.strip() for part in summary.split(" | ") if part.strip()]
    normalized_parts: list[str] = []
    seen: set[str] = set()
    for part in parts:
        normalized = normalize_json_fragment(part)
        if normalized not in seen:
            seen.add(normalized)
            normalized_parts.append(normalized)
    return " | ".join(normalized_parts)


def normalize_episode_features(features: dict, user_input: str, summary: str, steps: list[dict]) -> dict:
    normalized = dict(features) if isinstance(features, dict) else {}
    normalized.setdefault("control_version", "")
    normalized.setdefault("keywords", infer_keywords(user_input))
    normalized.setdefault("action_sequence", [step.get("action", "") for step in steps if step.get("action")])
    normalized.setdefault("plan_summary", infer_from_summary(summary, "plan"))
    normalized.setdefault("plan_action_items", [])
    normalized.setdefault("plan_unknowns", infer_unknowns(summary))
    normalized.setdefault("hyp_summary", infer_from_summary(summary, "hypothesis"))
    normalized.setdefault("evidence_targets", infer_evidence_targets(summary))
    normalized.setdefault("draft_summary", infer_from_summary(summary, "draft"))
    normalized.setdefault("used_tool", next((step.get("selected_tool", "") for step in steps if step.get("selected_tool")), ""))
    normalized.setdefault("has_result", "结果" in summary or "result" in summary.lower())
    normalized.setdefault("result_brief", infer_result_brief(summary))
    return normalized


def normalize_outcome_signature(
    outcome: dict,
    success: bool,
    value_score: float,
    steps: list[dict],
    answer: str,
    summary: str,
) -> dict:
    normalized = dict(outcome) if isinstance(outcome, dict) else {}
    normalized.setdefault("control_version", "")
    normalized.setdefault("success", bool(success))
    normalized.setdefault("value_score", float(value_score))
    normalized.setdefault("final_action", steps[-1].get("action", "") if steps else "")
    normalized.setdefault("step_count", len(steps))
    normalized.setdefault("uncertainty", infer_uncertainty(answer))
    normalized.setdefault("confidence_band", infer_confidence(summary, answer))
    normalized.setdefault("has_conflict", "风险:" in answer)
    normalized.setdefault("used_external_result", "外部结果:" in answer)
    return normalized


def normalize_answer_text(answer: str) -> str:
    if not isinstance(answer, str):
        return ""
    lines = [line.rstrip() for line in answer.splitlines()]
    return "\n".join(lines).strip()


def normalize_json_fragment(fragment: str) -> str:
    fragment = fragment.strip()
    if not fragment.startswith("{"):
        return fragment
    try:
        payload = json.loads(fragment)
    except json.JSONDecodeError:
        return fragment
    if not isinstance(payload, dict):
        return fragment
    normalized = normalize_slot_payload(payload)
    return json.dumps(normalized, ensure_ascii=False, sort_keys=True)


def normalize_slot_payload(payload: dict) -> dict:
    kind = payload.get("kind", "")
    if kind == "plan":
        return normalize_plan_payload(payload)
    if kind == "hypothesis":
        return normalize_hyp_payload(payload)
    if kind == "draft":
        return normalize_draft_payload(payload)
    return payload


def normalize_plan_payload(payload: dict) -> dict:
    return {
        "kind": "plan",
        "summary": str(payload.get("summary", "")),
        "action_items": dedupe_str_list(payload.get("action_items") or payload.get("steps") or []),
        "unknowns": dedupe_str_list(payload.get("unknowns") or []),
        "evidence_targets": dedupe_str_list(payload.get("evidence_targets") or payload.get("needs") or []),
        "keywords": dedupe_str_list(payload.get("keywords") or []),
        "confidence_band": str(payload.get("confidence_band", "medium")),
    }


def normalize_hyp_payload(payload: dict) -> dict:
    return {
        "kind": "hypothesis",
        "summary": str(payload.get("summary") or payload.get("question") or ""),
        "assumptions": dedupe_str_list(payload.get("assumptions") or []),
        "evidence_targets": dedupe_str_list(payload.get("evidence_targets") or payload.get("evidence_needed") or []),
        "confidence_band": str(payload.get("confidence_band", "medium")),
    }


def normalize_draft_payload(payload: dict) -> dict:
    summary = payload.get("summary") or payload.get("claim") or ""
    support = payload.get("support_items") or payload.get("support") or []
    confidence = payload.get("confidence_band") or payload.get("status") or "medium"
    return {
        "kind": "draft",
        "summary": str(summary),
        "support_items": dedupe_str_list(support),
        "open_risks": dedupe_str_list(payload.get("open_risks") or []),
        "confidence_band": str(confidence),
    }


def dedupe_str_list(values: object) -> list[str]:
    if not isinstance(values, list):
        return []
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        text = str(value).strip()
        if text and text not in seen:
            seen.add(text)
            result.append(text)
    return result


def infer_keywords(user_input: str) -> list[str]:
    if not isinstance(user_input, str):
        return []
    words = [token.strip() for token in user_input.replace(",", " ").split() if token.strip()]
    if words:
        return words[:6]
    return [user_input] if user_input else []


def infer_from_summary(summary: str, kind: str) -> str:
    if not isinstance(summary, str):
        return ""
    for part in summary.split(" | "):
        part = part.strip()
        if not part:
            continue
        if kind == "draft" and ("草稿" in part or '"kind": "draft"' in part):
            return part
        if kind == "plan" and ("维度" in part or '"kind": "plan"' in part):
            return part
        if kind == "hypothesis" and ("假设" in part or '"kind": "hypothesis"' in part):
            return part
    return ""


def infer_unknowns(summary: str) -> list[str]:
    if "比较" in summary:
        return ["比较维度待确认"]
    if "计算" in summary:
        return ["需要精确数值"]
    return []


def infer_evidence_targets(summary: str) -> list[str]:
    targets: list[str] = []
    if "检索" in summary:
        targets.append("外部事实")
    if "计算结果" in summary:
        targets.append("精确数值")
    return targets


def infer_result_brief(summary: str) -> str:
    for part in summary.split(" | "):
        if "检索" in part or "计算结果" in part:
            return part.strip()
    return ""


def infer_uncertainty(answer: str) -> float:
    if not isinstance(answer, str):
        return 1.0
    marker = "不确定度:"
    if marker not in answer:
        return 1.0
    try:
        return float(answer.split(marker, 1)[1].strip().splitlines()[0])
    except (ValueError, IndexError):
        return 1.0


def infer_confidence(summary: str, answer: str) -> str:
    if "置信=high" in answer or '"confidence_band": "high"' in summary:
        return "high"
    if "置信=medium" in answer or '"confidence_band": "medium"' in summary:
        return "medium"
    if "置信=low" in answer or '"confidence_band": "low"' in summary:
        return "low"
    return "medium"


def infer_state_signature(step: dict) -> dict:
    snapshot = step.get("feature_snapshot", {})
    if not isinstance(snapshot, dict):
        snapshot = {}
    return {
        "phase": "LEGACY",
        "last_action": "",
        "step_idx": int(step.get("step_idx", 0)),
        "uncertainty": float(snapshot.get("uncertainty", 1.0)),
        "answer_ready": float(snapshot.get("answer_ready", 0.0)),
        "candidate_slot": step.get("target_slot", ""),
        "latent_summary": "",
    }


def infer_memory_signature(step: dict) -> dict:
    snapshot = step.get("feature_snapshot", {})
    if not isinstance(snapshot, dict):
        snapshot = {}
    return {
        "has_goal": bool(snapshot.get("has_goal", 0.0)),
        "has_plan": bool(snapshot.get("has_plan", 0.0)),
        "has_hyp": False,
        "has_result": bool(snapshot.get("has_result", 0.0)),
        "has_draft": bool(snapshot.get("has_draft", 0.0)),
        "has_conflict": bool(snapshot.get("has_conflict", 0.0)),
        "focus_slot": "",
        "plan_brief": "",
        "result_brief": "",
    }


def infer_reward_reason(step: dict) -> str:
    action = step.get("action", "")
    reward = float(step.get("reward", 0.0) or 0.0)
    if action == "CALL_TOOL" and reward > 0:
        return "tool_result_obtained"
    if action == "WRITE_MEM" and reward > 0:
        return "state_written"
    if action == "ANSWER" and reward > 0:
        return "stable_answer"
    if action == "THINK" and reward < 0:
        return "idle_reasoning_cost"
    if action == "ANSWER" and reward < 0:
        return "premature_answer"
    return "neutral"
