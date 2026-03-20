from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

from carm.runtime_controls import (
    append_control_version,
    find_control_version,
    load_control_state,
    load_control_versions,
    load_controls_from_version,
    make_control_version_entry,
    save_control_state,
    save_controls,
)


def _clear_candidate(state: dict[str, object], decision: str) -> dict[str, object]:
    next_state = dict(state)
    next_state["rollout_status"] = "stable"
    next_state["candidate_version"] = ""
    next_state["candidate_baseline_version"] = ""
    next_state["candidate_episode_budget"] = 0
    next_state["candidate_started_utc"] = ""
    next_state["last_rollout_decision"] = decision
    return next_state


def _load_tool_gate_config(path: Path) -> dict[str, object]:
    if not path.exists():
        return {"default": {}, "by_tag": {}}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        return {"default": {}, "by_tag": {}}
    gate = payload.get("rollout_tool_gate", {})
    if not isinstance(gate, dict):
        return {"default": {}, "by_tag": {}}
    default_cfg = gate.get("default", {})
    by_tag = gate.get("by_tag", {})
    return {
        "default": default_cfg if isinstance(default_cfg, dict) else {},
        "by_tag": by_tag if isinstance(by_tag, dict) else {},
    }


def _tool_gate_issues(
    candidate_samples: dict[str, object],
    baseline_samples: dict[str, object],
    *,
    min_tool_match_rate: float,
    tool_match_drop_limit: float,
    tag_gate_config: dict[str, object] | None = None,
) -> list[str]:
    issues: list[str] = []
    candidate_by_tag = candidate_samples.get("by_tag", {}) if isinstance(candidate_samples, dict) else {}
    baseline_by_tag = baseline_samples.get("by_tag", {}) if isinstance(baseline_samples, dict) else {}
    tag_gate_config = tag_gate_config or {"default": {}, "by_tag": {}}
    default_cfg = tag_gate_config.get("default", {}) if isinstance(tag_gate_config, dict) else {}
    by_tag_cfg = tag_gate_config.get("by_tag", {}) if isinstance(tag_gate_config, dict) else {}
    if not isinstance(candidate_by_tag, dict):
        return issues

    for tag, metrics in candidate_by_tag.items():
        if not isinstance(metrics, dict):
            continue
        expected_count = int(metrics.get("expected_tool_count", 0) or 0)
        if expected_count <= 0:
            continue
        tag_cfg = by_tag_cfg.get(tag, {}) if isinstance(by_tag_cfg, dict) else {}
        tag_min_rate = float(tag_cfg.get("min_tool_match_rate", default_cfg.get("min_tool_match_rate", min_tool_match_rate)) or min_tool_match_rate)
        tag_drop_limit = float(tag_cfg.get("tool_match_drop_limit", default_cfg.get("tool_match_drop_limit", tool_match_drop_limit)) or tool_match_drop_limit)
        candidate_rate = float(metrics.get("tool_match_rate", 0.0) or 0.0)
        baseline_metrics = baseline_by_tag.get(tag, {}) if isinstance(baseline_by_tag, dict) else {}
        baseline_rate = float(baseline_metrics.get("tool_match_rate", candidate_rate) or candidate_rate) if isinstance(baseline_metrics, dict) else candidate_rate
        if candidate_rate < tag_min_rate:
            issues.append(f"{tag}: tool_match_rate={candidate_rate:.4f} below floor {tag_min_rate:.4f}")
        elif candidate_rate - baseline_rate < tag_drop_limit:
            issues.append(
                f"{tag}: tool_match_delta={candidate_rate - baseline_rate:.4f} below limit {tag_drop_limit:.4f}"
            )
    return issues


def main() -> int:
    controls_path = Path(os.environ.get("CARM_CONTROLS_PATH", "data/control/runtime_controls.json"))
    versions_path = Path(os.environ.get("CARM_CONTROL_VERSIONS_PATH", "data/control/control_versions.jsonl"))
    state_path = Path(os.environ.get("CARM_CONTROL_STATE_PATH", "data/control/control_state.json"))
    history_dir = Path(os.environ.get("CARM_CONTROL_HISTORY_DIR", "data/control/history"))
    metrics_path = Path(os.environ.get("CARM_CONTROL_METRICS_PATH", "data/control/control_version_metrics.json"))
    audit_path = Path(os.environ.get("CARM_ROLLOUT_AUDIT_PATH", "data/control/rollout_judgments.jsonl"))
    gate_config_path = Path(os.environ.get("CARM_CONTROL_CYCLE_CONFIG", "configs/control_cycle.json"))
    auto_rollback = os.environ.get("CARM_CONTROL_AUTO_ROLLBACK", "0") == "1"
    success_drop_limit = float(os.environ.get("CARM_CONTROL_SUCCESS_DROP_LIMIT", "-0.05"))
    value_drop_limit = float(os.environ.get("CARM_CONTROL_VALUE_DROP_LIMIT", "-0.05"))
    step_increase_limit = float(os.environ.get("CARM_CONTROL_STEP_INCREASE_LIMIT", "1.0"))
    min_tool_match_rate = float(os.environ.get("CARM_CONTROL_MIN_TOOL_MATCH_RATE", "0.5"))
    tool_match_drop_limit = float(os.environ.get("CARM_CONTROL_TOOL_MATCH_DROP_LIMIT", "-0.25"))
    tag_gate_config = _load_tool_gate_config(gate_config_path)

    control_state = load_control_state(state_path)
    if str(control_state.get("rollout_status", "stable")) != "candidate":
        print("No candidate control rollout to judge.")
        return 0

    candidate_version = str(control_state.get("candidate_version", ""))
    baseline_version = str(control_state.get("candidate_baseline_version", ""))
    episode_budget = int(control_state.get("candidate_episode_budget", 0) or 0)

    metrics = json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.exists() else {}
    version_metrics = metrics.get("version_metrics", {})
    sample_metrics = metrics.get("sample_metrics", {})
    candidate_metrics = version_metrics.get(candidate_version, {})
    baseline_metrics = version_metrics.get(baseline_version, {})
    candidate_samples = sample_metrics.get(candidate_version, {})
    baseline_samples = sample_metrics.get(baseline_version, {})
    candidate_episode_count = int(candidate_metrics.get("episode_count", 0) or 0)

    decision = "observe"
    reason = "candidate rollout still within observation window"
    applied = False

    if candidate_episode_count < episode_budget:
        decision = "pending"
        reason = f"candidate has {candidate_episode_count} episode(s), below budget {episode_budget}"
    elif not baseline_metrics:
        decision = "promote"
        reason = "no baseline metrics available; keeping candidate as current stable version"
        control_state = _clear_candidate(control_state, decision)
        save_control_state(state_path, control_state)
    else:
        delta_success = float(candidate_metrics.get("success_rate", 0.0)) - float(baseline_metrics.get("success_rate", 0.0))
        delta_value = float(candidate_metrics.get("avg_value_score", 0.0)) - float(baseline_metrics.get("avg_value_score", 0.0))
        delta_steps = float(candidate_metrics.get("avg_step_count", 0.0)) - float(baseline_metrics.get("avg_step_count", 0.0))
        tool_issues = _tool_gate_issues(
            candidate_samples,
            baseline_samples,
            min_tool_match_rate=min_tool_match_rate,
            tool_match_drop_limit=tool_match_drop_limit,
            tag_gate_config=tag_gate_config,
        )

        if delta_success < success_drop_limit or delta_value < value_drop_limit or delta_steps > step_increase_limit or tool_issues:
            decision = "rollback"
            reason = (
                f"candidate underperformed baseline: success_delta={delta_success:.4f}, "
                f"value_delta={delta_value:.4f}, step_delta={delta_steps:.4f}"
            )
            if tool_issues:
                reason += f"; tool_gate={' | '.join(tool_issues)}"
            if auto_rollback:
                control_versions = load_control_versions(versions_path)
                target_entry = find_control_version(control_versions, baseline_version)
                if target_entry is not None:
                    restored_controls = load_controls_from_version(target_entry)
                    save_controls(controls_path, restored_controls)
                    rollback_entry = make_control_version_entry(
                        restored_controls,
                        history_dir,
                        "rollback",
                        parent_version=candidate_version,
                        restored_from=baseline_version,
                        action_types=["rollback"],
                        applied_count=0,
                    )
                    append_control_version(versions_path, rollback_entry)
                    control_state["current_version"] = rollback_entry["version_id"]
                    control_state["previous_version"] = candidate_version
                    control_state["history_count"] = int(control_state.get("history_count", 0)) + 1
                    control_state["last_updated_utc"] = rollback_entry["created_at_utc"]
                    control_state["last_reason"] = "rollback"
                    control_state["last_applied_count"] = 0
                    control_state["last_target_version"] = baseline_version
                    control_state = _clear_candidate(control_state, decision)
                    save_control_state(state_path, control_state)
                    applied = True
        else:
            decision = "promote"
            reason = (
                f"candidate acceptable: success_delta={delta_success:.4f}, "
                f"value_delta={delta_value:.4f}, step_delta={delta_steps:.4f}"
            )
            control_state = _clear_candidate(control_state, decision)
            save_control_state(state_path, control_state)

    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_record = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "candidate_version": candidate_version,
        "baseline_version": baseline_version,
        "episode_budget": episode_budget,
        "candidate_episode_count": candidate_episode_count,
        "decision": decision,
        "reason": reason,
        "applied": applied,
    }
    with audit_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(audit_record, ensure_ascii=False) + "\n")

    print(f"Rollout decision: {decision} ({reason})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
