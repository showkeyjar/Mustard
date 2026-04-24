from __future__ import annotations

import json
import os
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path

from carm.runtime_controls import (
    append_control_version,
    load_control_state,
    load_controls,
    make_control_version_entry,
    save_control_state,
    save_controls,
    update_control_state_for_entry,
)


def apply_action(controls: dict[str, dict[str, float | int]], action: dict[str, object]) -> dict[str, object]:
    updated = deepcopy(controls)
    action_type = str(action.get("type", ""))
    applied = False
    changes: list[dict[str, object]] = []

    if action_type == "tighten_constraint":
        before = float(updated["policy"]["think_penalty"])
        after = min(before + 0.05, 0.3)
        updated["policy"]["think_penalty"] = after
        applied = after != before
        changes.append({"path": "policy.think_penalty", "before": before, "after": after})

    elif action_type == "raise_tool_bias":
        before = float(updated["policy"]["call_tool_bonus"])
        after = min(before + 0.08, 0.5)
        updated["policy"]["call_tool_bonus"] = after
        applied = after != before
        changes.append({"path": "policy.call_tool_bonus", "before": before, "after": after})

    elif action_type == "promote_draft_path":
        before_ready = float(updated["core"]["result_draft_answer_ready_bonus"])
        after_ready = min(before_ready + 0.05, 0.3)
        updated["core"]["result_draft_answer_ready_bonus"] = after_ready

        before_unc = float(updated["core"]["result_draft_uncertainty_delta"])
        after_unc = min(before_unc + 0.03, 0.2)
        updated["core"]["result_draft_uncertainty_delta"] = after_unc
        applied = after_ready != before_ready or after_unc != before_unc
        changes.extend(
            [
                {"path": "core.result_draft_answer_ready_bonus", "before": before_ready, "after": after_ready},
                {"path": "core.result_draft_uncertainty_delta", "before": before_unc, "after": after_unc},
            ]
        )

    elif action_type == "reduce_glance_trigger":
        before = float(updated["glance"]["high_uncertainty_threshold"])
        after = min(before + 0.03, 0.95)
        updated["glance"]["high_uncertainty_threshold"] = after
        applied = after != before
        changes.append({"path": "glance.high_uncertainty_threshold", "before": before, "after": after})

    elif action_type == "enable_combined_tool_policy_candidate":
        policy = updated["policy"]
        before_calc = int(policy.get("prefer_calculator_for_mixed_numeric_code", 0) or 0)
        before_search = int(policy.get("prefer_search_for_comparison_evidence", 0) or 0)
        policy["prefer_calculator_for_mixed_numeric_code"] = 1
        policy["prefer_search_for_comparison_evidence"] = 1
        applied = before_calc != 1 or before_search != 1
        changes.extend(
            [
                {
                    "path": "policy.prefer_calculator_for_mixed_numeric_code",
                    "before": before_calc,
                    "after": 1,
                },
                {
                    "path": "policy.prefer_search_for_comparison_evidence",
                    "before": before_search,
                    "after": 1,
                },
            ]
        )

    elif action_type in {"retain_glance_policy", "observe"}:
        applied = False

    return {
        "action": action,
        "applied": applied,
        "changes": changes,
        "controls": updated,
    }


def main() -> int:
    consolidated_path = Path(os.environ.get("CARM_CONSOLIDATED_PATH", "data/review/consolidated_recommendations.json"))
    controls_path = Path(os.environ.get("CARM_CONTROLS_PATH", "data/control/runtime_controls.json"))
    audit_path = Path(os.environ.get("CARM_APPLY_AUDIT_PATH", "data/control/applied_actions.jsonl"))
    versions_path = Path(os.environ.get("CARM_CONTROL_VERSIONS_PATH", "data/control/control_versions.jsonl"))
    state_path = Path(os.environ.get("CARM_CONTROL_STATE_PATH", "data/control/control_state.json"))
    history_dir = Path(os.environ.get("CARM_CONTROL_HISTORY_DIR", "data/control/history"))
    rollout_budget = int(os.environ.get("CARM_CONTROL_ROLLOUT_BUDGET", "3"))

    consolidated = json.loads(consolidated_path.read_text(encoding="utf-8")) if consolidated_path.exists() else {}
    actions = consolidated.get("slow_path_actions", [])
    controls = load_controls(controls_path)
    control_state = load_control_state(state_path)

    current_version = str(control_state.get("current_version", ""))
    if not current_version:
        bootstrap_entry = make_control_version_entry(
            controls,
            history_dir,
            "bootstrap",
            action_types=["bootstrap"],
            applied_count=0,
        )
        append_control_version(versions_path, bootstrap_entry)
        control_state = update_control_state_for_entry(control_state, bootstrap_entry, previous_version="")
        save_control_state(state_path, control_state)
        current_version = str(bootstrap_entry["version_id"])

    applied_results: list[dict[str, object]] = []
    current = deepcopy(controls)
    for action in actions:
        result = apply_action(current, action)
        current = result["controls"]
        applied_results.append({k: v for k, v in result.items() if k != "controls"})

    if current != controls:
        save_controls(controls_path, current)
        applied_count = sum(1 for result in applied_results if result.get("applied"))
        action_types = [str(action.get("type", "")) for action in actions if isinstance(action, dict)]
        version_entry = make_control_version_entry(
            current,
            history_dir,
            "apply_slow_path_actions",
            parent_version=current_version,
            action_types=action_types,
            applied_count=applied_count,
            source_actions=[action for action in actions if isinstance(action, dict)],
        )
        append_control_version(versions_path, version_entry)
        control_state = update_control_state_for_entry(control_state, version_entry, previous_version=current_version)
        control_state["candidate_episode_budget"] = rollout_budget
        save_control_state(state_path, control_state)

    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_record = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "base_version": current_version,
        "result_version": str(control_state.get("current_version", current_version)),
        "input_actions": actions,
        "results": applied_results,
    }
    with audit_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(audit_record, ensure_ascii=False) + "\n")

    applied_count = sum(1 for result in applied_results if result.get("applied"))
    print(f"Applied {applied_count} slow-path action(s) to {controls_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
