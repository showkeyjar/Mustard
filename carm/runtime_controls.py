from __future__ import annotations

import json
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path


DEFAULT_CONTROLS: dict[str, dict[str, float | int]] = {
    "policy": {
        "call_tool_bonus": 0.0,
        "verify_bonus": 0.0,
        "think_penalty": 0.0,
        "answer_penalty": 0.0,
        "require_conflict_verify_before_answer": 0,
        "prefer_calculator_for_mixed_numeric_code": 0,
        "prefer_search_for_comparison_evidence": 0,
    },
    "glance": {
        "budget": 1,
        "high_uncertainty_threshold": 0.78,
    },
    "core": {
        "result_draft_answer_ready_bonus": 0.0,
        "result_draft_uncertainty_delta": 0.0,
    },
}


DEFAULT_CONTROL_STATE: dict[str, object] = {
    "current_version": "",
    "previous_version": "",
    "history_count": 0,
    "last_updated_utc": "",
    "last_reason": "",
    "last_applied_count": 0,
    "last_target_version": "",
    "rollout_status": "stable",
    "candidate_version": "",
    "candidate_baseline_version": "",
    "candidate_episode_budget": 0,
    "candidate_started_utc": "",
    "last_rollout_decision": "",
}


def load_controls(path: str | Path) -> dict[str, dict[str, float | int]]:
    controls = deepcopy(DEFAULT_CONTROLS)
    path = Path(path)
    if not path.exists():
        return controls

    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        return controls

    for section, section_values in payload.items():
        if section in controls and isinstance(section_values, dict):
            controls[section].update(section_values)
    return controls


def save_controls(path: str | Path, controls: dict[str, dict[str, float | int]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(controls, ensure_ascii=False, indent=2), encoding="utf-8")


def load_control_state(path: str | Path) -> dict[str, object]:
    state = deepcopy(DEFAULT_CONTROL_STATE)
    path = Path(path)
    if not path.exists():
        return state

    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        return state

    state.update(payload)
    return state


def save_control_state(path: str | Path, state: dict[str, object]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")


def load_control_versions(path: str | Path) -> list[dict[str, object]]:
    path = Path(path)
    if not path.exists():
        return []

    entries: list[dict[str, object]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        payload = json.loads(line)
        if isinstance(payload, dict):
            entries.append(payload)
    return entries


def append_control_version(path: str | Path, entry: dict[str, object]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry, ensure_ascii=False) + "\n")


def make_control_version_entry(
    controls: dict[str, dict[str, float | int]],
    history_dir: str | Path,
    reason: str,
    *,
    parent_version: str = "",
    restored_from: str = "",
    action_types: list[str] | None = None,
    applied_count: int = 0,
    source_actions: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    created_at = datetime.now(timezone.utc).isoformat()
    version_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    history_dir = Path(history_dir)
    history_dir.mkdir(parents=True, exist_ok=True)
    snapshot_path = history_dir / f"{version_id}.json"
    snapshot_path.write_text(json.dumps(controls, ensure_ascii=False, indent=2), encoding="utf-8")

    return {
        "version_id": version_id,
        "created_at_utc": created_at,
        "reason": reason,
        "parent_version": parent_version,
        "restored_from": restored_from,
        "applied_count": applied_count,
        "action_types": list(action_types or []),
        "source_actions": list(source_actions or []),
        "snapshot_path": str(snapshot_path),
    }


def find_control_version(
    entries: list[dict[str, object]],
    version_id: str,
) -> dict[str, object] | None:
    for entry in entries:
        if entry.get("version_id") == version_id:
            return entry
    return None


def load_controls_from_version(entry: dict[str, object]) -> dict[str, dict[str, float | int]]:
    snapshot_path = Path(str(entry.get("snapshot_path", "")))
    return load_controls(snapshot_path)


def update_control_state_for_entry(
    state: dict[str, object],
    entry: dict[str, object],
    *,
    previous_version: str = "",
) -> dict[str, object]:
    next_state = deepcopy(state)
    next_state["current_version"] = entry.get("version_id", "")
    next_state["previous_version"] = previous_version
    next_state["history_count"] = int(next_state.get("history_count", 0)) + 1
    next_state["last_updated_utc"] = entry.get("created_at_utc", "")
    next_state["last_reason"] = entry.get("reason", "")
    next_state["last_applied_count"] = int(entry.get("applied_count", 0))
    next_state["last_target_version"] = entry.get("restored_from", "")
    reason = str(entry.get("reason", ""))
    if reason == "apply_slow_path_actions" and int(entry.get("applied_count", 0)) > 0:
        next_state["rollout_status"] = "candidate"
        next_state["candidate_version"] = entry.get("version_id", "")
        next_state["candidate_baseline_version"] = previous_version
        next_state["candidate_episode_budget"] = int(next_state.get("candidate_episode_budget", 0) or 0)
        next_state["candidate_started_utc"] = entry.get("created_at_utc", "")
        next_state["last_rollout_decision"] = ""
    elif reason == "rollback":
        next_state["rollout_status"] = "stable"
        next_state["candidate_version"] = ""
        next_state["candidate_baseline_version"] = ""
        next_state["candidate_episode_budget"] = 0
        next_state["candidate_started_utc"] = ""
        next_state["last_rollout_decision"] = "rollback"
    return next_state
