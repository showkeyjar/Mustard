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
    update_control_state_for_entry,
)


def main() -> int:
    controls_path = Path(os.environ.get("CARM_CONTROLS_PATH", "data/control/runtime_controls.json"))
    versions_path = Path(os.environ.get("CARM_CONTROL_VERSIONS_PATH", "data/control/control_versions.jsonl"))
    state_path = Path(os.environ.get("CARM_CONTROL_STATE_PATH", "data/control/control_state.json"))
    history_dir = Path(os.environ.get("CARM_CONTROL_HISTORY_DIR", "data/control/history"))
    audit_path = Path(os.environ.get("CARM_ROLLBACK_AUDIT_PATH", "data/control/rollback_actions.jsonl"))
    target_version = os.environ.get("CARM_TARGET_VERSION", "").strip()

    control_state = load_control_state(state_path)
    control_versions = load_control_versions(versions_path)
    current_version = str(control_state.get("current_version", ""))

    if not control_versions or not current_version:
        print("No control history available for rollback.")
        return 0

    if not target_version:
        target_version = str(control_state.get("previous_version", ""))
        if not target_version and len(control_versions) >= 2:
            target_version = str(control_versions[-2].get("version_id", ""))

    if not target_version or target_version == current_version:
        print("No eligible previous control version found.")
        return 0

    target_entry = find_control_version(control_versions, target_version)
    if target_entry is None:
        print(f"Target control version not found: {target_version}")
        return 1

    restored_controls = load_controls_from_version(target_entry)
    save_controls(controls_path, restored_controls)

    rollback_entry = make_control_version_entry(
        restored_controls,
        history_dir,
        "rollback",
        parent_version=current_version,
        restored_from=target_version,
        action_types=["rollback"],
        applied_count=0,
    )
    append_control_version(versions_path, rollback_entry)
    next_state = update_control_state_for_entry(control_state, rollback_entry, previous_version=current_version)
    save_control_state(state_path, next_state)

    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_record = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "from_version": current_version,
        "target_version": target_version,
        "result_version": rollback_entry["version_id"],
    }
    with audit_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(audit_record, ensure_ascii=False) + "\n")

    print(f"Rolled back runtime controls from {current_version} to {target_version}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
