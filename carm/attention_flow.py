from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any

from carm.schemas import EpisodeRecord, StepRecord


RISKY_RESIDUALS = {
    "conflict_unresolved",
    "missing_evidence",
    "tool_boundary_ambiguous",
    "draft_not_verified",
}


@dataclass(frozen=True)
class AttentionNode:
    episode_id: str
    step_idx: int
    action: str
    focus_target: str
    focus_reason: str
    evidence_need: list[str] = field(default_factory=list)
    residual_pressure: list[str] = field(default_factory=list)
    transition: str = ""
    release_condition: str = ""
    model_view: str = ""
    source_step: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class AttentionTrainingView:
    episode_id: str
    step_idx: int
    current_focus: str
    next_focus: str
    residual_pressure: list[str]
    evidence_need: list[str]
    recommended_transition: str
    recommended_action: str
    release_allowed: bool
    release_condition: str
    supervision_note: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def project_episode_attention(episode: EpisodeRecord, episode_id: str = "") -> list[AttentionNode]:
    nodes: list[AttentionNode] = []
    previous_focus = "start"
    stable_episode_id = episode_id or _stable_episode_id(episode)

    for step in episode.steps:
        focus_target = _focus_target(step)
        evidence_need = _evidence_need(episode, step)
        residual_pressure = _residual_pressure(episode, step)
        transition = _transition(previous_focus, focus_target, step)
        release_condition = _release_condition(step, residual_pressure)
        focus_reason = _focus_reason(step, focus_target, evidence_need, residual_pressure)
        node = AttentionNode(
            episode_id=stable_episode_id,
            step_idx=step.step_idx,
            action=step.action,
            focus_target=focus_target,
            focus_reason=focus_reason,
            evidence_need=evidence_need,
            residual_pressure=residual_pressure,
            transition=transition,
            release_condition=release_condition,
            model_view=_model_view(
                focus_target=focus_target,
                evidence_need=evidence_need,
                residual_pressure=residual_pressure,
                transition=transition,
                action=step.action,
                release_condition=release_condition,
            ),
            source_step={
                "target_slot": step.target_slot,
                "selected_tool": step.selected_tool,
                "reward": step.reward,
                "reward_reason": step.reward_reason,
                "state_signature": dict(step.state_signature),
                "memory_signature": dict(step.memory_signature),
            },
        )
        nodes.append(node)
        previous_focus = focus_target

    return nodes


def project_eval_row_attention(row: dict[str, Any], prompt: str = "") -> list[AttentionNode]:
    case_id = str(row.get("id", "")) or "eval-row"
    logic_skill = str(row.get("logic_skill", ""))
    expected_tool = str(row.get("expected_tool", ""))
    used_tool = str(row.get("pretrained_used_tool", row.get("used_tool", "")))
    actions = row.get("pretrained_actions", row.get("actions", []))
    if not isinstance(actions, list):
        actions = []

    steps: list[StepRecord] = []
    has_result = False
    has_draft = False
    verified = False
    write_count = 0
    for index, raw_action in enumerate(actions, start=1):
        action = str(raw_action)
        selected_tool = ""
        target_slot = ""
        if action in {"CALL_TOOL", "CALL_BIGMODEL"}:
            selected_tool = used_tool or expected_tool
            has_result = True
        if action == "WRITE_MEM":
            write_count += 1
            target_slot = _eval_write_slot(logic_skill, write_count, has_result)
            if target_slot == "DRAFT":
                has_draft = True
        if action == "VERIFY":
            verified = True
        feature_snapshot = _eval_feature_snapshot(logic_skill, prompt, expected_tool, used_tool)
        steps.append(
            StepRecord(
                step_idx=index,
                action=action,
                reason=f"eval row action {action}",
                score=1.0,
                feature_snapshot=feature_snapshot,
                memory_signature={
                    "has_result": has_result,
                    "has_draft": has_draft,
                    "has_conflict": logic_skill == "conflict_detection" and not has_result,
                },
                state_signature={"verified": "1"} if verified else {},
                user_input=prompt,
                selected_tool=selected_tool,
                target_slot=target_slot,
                reward=0.0 if bool(row.get("pretrained_match", row.get("tool_match", True))) else -0.4,
                reward_reason="eval_projection",
            )
        )

    episode = EpisodeRecord(
        user_input=prompt or case_id,
        answer="",
        summary=f"eval_row={case_id}",
        success=bool(row.get("pretrained_match", row.get("tool_match", True))),
        value_score=1.0 if bool(row.get("pretrained_match", row.get("tool_match", True))) else 0.4,
        episode_features={
            "logic_skill": logic_skill,
            "expected_tool": expected_tool,
            "used_tool": used_tool,
        },
        steps=steps,
    )
    return project_episode_attention(episode, episode_id=f"eval:{case_id}")


def build_attention_report(nodes: list[AttentionNode]) -> dict[str, Any]:
    total = len(nodes)
    by_episode: dict[str, list[AttentionNode]] = {}
    for node in nodes:
        by_episode.setdefault(node.episode_id, []).append(node)

    risky_nodes = [
        node
        for node in nodes
        if any(residual in RISKY_RESIDUALS for residual in node.residual_pressure)
    ]
    answer_nodes = [node for node in nodes if node.action == "ANSWER"]
    premature_answers = [
        node
        for node in answer_nodes
        if any(residual in RISKY_RESIDUALS for residual in node.residual_pressure)
    ]
    grounded_tool_nodes = _grounded_tool_nodes(by_episode)
    tool_nodes = [
        node
        for node in nodes
        if node.action in {"CALL_TOOL", "CALL_BIGMODEL"} or bool(node.source_step.get("selected_tool"))
    ]
    resolved_risky_nodes = _resolved_risky_nodes(by_episode)
    raw_step_count = sum(len(items) for items in by_episode.values())

    return {
        "summary": {
            "episode_count": len(by_episode),
            "node_count": total,
            "focus_continuity": _rate(
                sum(1 for node in nodes if node.transition and not node.transition.startswith("broken")),
                total,
            ),
            "evidence_grounding": _rate(len(grounded_tool_nodes), len(tool_nodes)),
            "residual_resolution": _rate(len(resolved_risky_nodes), len(risky_nodes)),
            "premature_release_rate": _rate(len(premature_answers), len(answer_nodes)),
            "attention_compression_ratio": _rate(total, raw_step_count),
            "risky_residual_count": len(risky_nodes),
            "premature_release_count": len(premature_answers),
        },
        "focus_counts": _counts(node.focus_target for node in nodes),
        "residual_counts": _counts(residual for node in nodes for residual in node.residual_pressure),
        "transition_counts": _counts(node.transition for node in nodes),
        "premature_release_nodes": [node.to_dict() for node in premature_answers[:10]],
    }


def build_training_views(nodes: list[AttentionNode]) -> list[AttentionTrainingView]:
    by_episode: dict[str, list[AttentionNode]] = {}
    for node in nodes:
        by_episode.setdefault(node.episode_id, []).append(node)

    views: list[AttentionTrainingView] = []
    for episode_id, episode_nodes in by_episode.items():
        ordered = sorted(episode_nodes, key=lambda item: item.step_idx)
        for index, node in enumerate(ordered):
            next_node = ordered[index + 1] if index + 1 < len(ordered) else None
            next_focus = next_node.focus_target if next_node is not None else "end"
            release_allowed = (
                node.focus_target == "release"
                and node.release_condition == "draft_ready_and_residual_pressure_low"
            )
            if node.focus_target != "release":
                release_allowed = not any(residual in RISKY_RESIDUALS for residual in node.residual_pressure) and next_focus == "release"
            views.append(
                AttentionTrainingView(
                    episode_id=episode_id,
                    step_idx=node.step_idx,
                    current_focus=node.focus_target,
                    next_focus=next_focus,
                    residual_pressure=list(node.residual_pressure),
                    evidence_need=list(node.evidence_need),
                    recommended_transition=node.transition,
                    recommended_action=node.action,
                    release_allowed=release_allowed,
                    release_condition=node.release_condition,
                    supervision_note=_supervision_note(node, next_focus, release_allowed),
                )
            )
    return views


def training_views_to_jsonl(views: list[AttentionTrainingView]) -> str:
    return "\n".join(json.dumps(view.to_dict(), ensure_ascii=False) for view in views)


def nodes_to_jsonl(nodes: list[AttentionNode]) -> str:
    return "\n".join(json.dumps(node.to_dict(), ensure_ascii=False) for node in nodes)


def nodes_from_payloads(payloads: list[dict[str, Any]]) -> list[AttentionNode]:
    nodes: list[AttentionNode] = []
    for payload in payloads:
        nodes.append(
            AttentionNode(
                episode_id=str(payload.get("episode_id", "")),
                step_idx=int(payload.get("step_idx", 0) or 0),
                action=str(payload.get("action", "")),
                focus_target=str(payload.get("focus_target", "")),
                focus_reason=str(payload.get("focus_reason", "")),
                evidence_need=_string_list(payload.get("evidence_need", [])),
                residual_pressure=_string_list(payload.get("residual_pressure", [])),
                transition=str(payload.get("transition", "")),
                release_condition=str(payload.get("release_condition", "")),
                model_view=str(payload.get("model_view", "")),
                source_step=dict(payload.get("source_step", {})) if isinstance(payload.get("source_step", {}), dict) else {},
            )
        )
    return nodes


def _focus_target(step: StepRecord) -> str:
    if step.action == "ANSWER":
        return "release"
    if step.action == "VERIFY":
        return "verification"
    if step.selected_tool:
        if step.selected_tool == "bigmodel_proxy":
            return "external_synthesis"
        return "evidence"
    if step.target_slot == "GOAL":
        return "goal"
    if step.target_slot == "PLAN":
        return "plan"
    if step.target_slot == "HYP":
        if _feature(step, "needs_conflict_detection") > 0.0 or _feature(step, "conflict_signal") > 0.0:
            return "conflict"
        return "hypothesis"
    if step.target_slot == "DRAFT":
        return "draft"
    if step.action == "ROLLBACK":
        return "rollback"
    if step.action == "READ_MEM":
        return "memory"
    return "internal_state"


def _evidence_need(episode: EpisodeRecord, step: StepRecord) -> list[str]:
    needs: list[str] = []
    if _is_conflict_episode(episode, step):
        needs.extend(["conflict_points", "source_authority", "time_validity"])
    if _feature(step, "needs_calc") > 0.0 or _feature(step, "calc_signal") > 0.0:
        needs.append("precise_numeric_result")
    if _feature(step, "needs_code") > 0.0 or _feature(step, "code_signal") > 0.0:
        needs.append("executable_validation")
    if _feature(step, "needs_compare") > 0.0 or _feature(step, "compare_signal") > 0.0:
        needs.extend(["comparison_dimensions", "external_facts"])
    memory_signature = step.memory_signature or {}
    if not bool(memory_signature.get("has_result", False)) and step.action not in {"ANSWER", "VERIFY"}:
        needs.append("external_support")
    return _dedupe(needs)


def _residual_pressure(episode: EpisodeRecord, step: StepRecord) -> list[str]:
    residuals: list[str] = []
    memory_signature = step.memory_signature or {}
    state_signature = step.state_signature or {}
    verified = state_signature.get("verified", "") == "1"
    if _is_conflict_episode(episode, step) and not verified:
        residuals.append("conflict_unresolved")
    if not bool(memory_signature.get("has_result", False)) and step.action in {"WRITE_MEM", "ANSWER"}:
        residuals.append("missing_evidence")
    if (
        (_feature(step, "needs_calc") > 0.0 or _feature(step, "calc_signal") > 0.0)
        and (_feature(step, "needs_code") > 0.0 or _feature(step, "code_signal") > 0.0)
    ):
        residuals.append("tool_boundary_ambiguous")
    if (
        step.action == "ANSWER"
        and bool(memory_signature.get("has_draft", False))
        and not verified
        and (_is_conflict_episode(episode, step) or bool(memory_signature.get("has_conflict", False)))
    ):
        residuals.append("draft_not_verified")
    if step.reward < -0.01:
        residuals.append("negative_reward_pressure")
    return _dedupe(residuals)


def _transition(previous_focus: str, focus_target: str, step: StepRecord) -> str:
    if previous_focus == "start":
        return f"start_to_{focus_target}"
    if previous_focus == focus_target:
        return f"hold_{focus_target}"
    if step.action in {"CALL_TOOL", "CALL_BIGMODEL"}:
        return f"{previous_focus}_to_evidence"
    if step.action == "VERIFY":
        return f"{previous_focus}_to_verification"
    if step.action == "ANSWER":
        return f"{previous_focus}_to_release"
    return f"{previous_focus}_to_{focus_target}"


def _release_condition(step: StepRecord, residual_pressure: list[str]) -> str:
    if step.action == "ANSWER":
        if any(residual in RISKY_RESIDUALS for residual in residual_pressure):
            return "released_with_risky_residuals"
        return "draft_ready_and_residual_pressure_low"
    if step.action == "VERIFY":
        return "verification_required_before_release"
    if step.action in {"CALL_TOOL", "CALL_BIGMODEL"}:
        return "release_after_evidence_is_integrated"
    if "conflict_unresolved" in residual_pressure:
        return "release_blocked_until_conflict_resolved"
    if "missing_evidence" in residual_pressure:
        return "release_blocked_until_evidence_arrives"
    return "continue_attention_flow"


def _focus_reason(
    step: StepRecord,
    focus_target: str,
    evidence_need: list[str],
    residual_pressure: list[str],
) -> str:
    if residual_pressure:
        return f"Focus {focus_target} because residual pressure remains: {', '.join(residual_pressure)}."
    if evidence_need:
        return f"Focus {focus_target} to satisfy evidence needs: {', '.join(evidence_need)}."
    return f"Focus {focus_target} based on action {step.action}."


def _model_view(
    *,
    focus_target: str,
    evidence_need: list[str],
    residual_pressure: list[str],
    transition: str,
    action: str,
    release_condition: str,
) -> str:
    return (
        f"focus={focus_target}; "
        f"transition={transition}; "
        f"evidence_need={','.join(evidence_need) or 'none'}; "
        f"residual_pressure={','.join(residual_pressure) or 'none'}; "
        f"recommended_action={action}; "
        f"release_condition={release_condition}"
    )


def _grounded_tool_nodes(by_episode: dict[str, list[AttentionNode]]) -> list[AttentionNode]:
    grounded: list[AttentionNode] = []
    for nodes in by_episode.values():
        for index, node in enumerate(nodes):
            if node.action not in {"CALL_TOOL", "CALL_BIGMODEL"} and not bool(node.source_step.get("selected_tool")):
                continue
            later = nodes[index + 1 :]
            if any(item.focus_target in {"draft", "verification", "release"} for item in later):
                grounded.append(node)
    return grounded


def _resolved_risky_nodes(by_episode: dict[str, list[AttentionNode]]) -> list[AttentionNode]:
    resolved: list[AttentionNode] = []
    for nodes in by_episode.values():
        for index, node in enumerate(nodes):
            if not any(residual in RISKY_RESIDUALS for residual in node.residual_pressure):
                continue
            later = nodes[index + 1 :]
            if any(item.focus_target == "verification" for item in later) or (
                later and not any(residual in RISKY_RESIDUALS for residual in later[-1].residual_pressure)
            ):
                resolved.append(node)
    return resolved


def _is_conflict_episode(episode: EpisodeRecord, step: StepRecord) -> bool:
    text = episode.user_input
    return (
        _feature(step, "needs_conflict_detection") > 0.0
        or _feature(step, "conflict_signal") > 0.0
        or any(token in text for token in ("冲突", "相反", "矛盾", "不一致", "分歧"))
    )


def _stable_episode_id(episode: EpisodeRecord) -> str:
    normalized = "".join(ch for ch in episode.user_input.lower() if ch.isalnum() or "\u4e00" <= ch <= "\u9fff")
    return normalized[:48] or "episode"


def _feature(step: StepRecord, name: str) -> float:
    try:
        return float(step.feature_snapshot.get(name, 0.0) or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _rate(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return round(numerator / denominator, 4)


def _counts(values: Any) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        key = str(value)
        if not key:
            continue
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items(), key=lambda item: (-item[1], item[0])))


def _dedupe(values: list[str]) -> list[str]:
    return list(dict.fromkeys(value for value in values if value))


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value]


def _supervision_note(node: AttentionNode, next_focus: str, release_allowed: bool) -> str:
    if release_allowed:
        return "release is allowed because risky residual pressure is low or resolved"
    if any(residual in RISKY_RESIDUALS for residual in node.residual_pressure):
        return f"continue toward {next_focus}; risky residuals remain: {', '.join(node.residual_pressure)}"
    if node.evidence_need:
        return f"continue toward {next_focus}; evidence needs remain: {', '.join(node.evidence_need)}"
    return f"continue toward {next_focus}"


def _eval_write_slot(logic_skill: str, write_count: int, has_result: bool) -> str:
    if write_count == 1 and not has_result:
        return "GOAL"
    if has_result:
        return "DRAFT"
    if logic_skill == "conflict_detection":
        return "HYP"
    if logic_skill in {"comparison", "step_planning", "constraint_planning"}:
        return "PLAN"
    return "HYP"


def _eval_feature_snapshot(logic_skill: str, prompt: str, expected_tool: str, used_tool: str) -> dict[str, float]:
    text = prompt.lower()
    return {
        "needs_conflict_detection": 1.0 if logic_skill == "conflict_detection" else 0.0,
        "needs_compare": 1.0 if logic_skill == "comparison" else 0.0,
        "needs_calc": 1.0 if expected_tool == "calculator" or "计算" in prompt else 0.0,
        "needs_code": 1.0 if expected_tool == "code_executor" or "python" in text or "代码" in prompt else 0.0,
        "needs_formal_synthesis": 1.0 if used_tool == "bigmodel_proxy" else 0.0,
    }
