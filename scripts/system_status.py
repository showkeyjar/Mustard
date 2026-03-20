from __future__ import annotations

import json
import os
from pathlib import Path

from carm.runtime_controls import load_control_state, load_control_versions, load_controls


def load_json(path: Path) -> dict[str, object]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def count_jsonl(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())


def main() -> int:
    experience_path = Path("data/experience/episodes.jsonl")
    review_path = Path("data/review/reviews.jsonl")
    consolidated_path = Path("data/review/consolidated_recommendations.json")
    control_metrics_path = Path(os.environ.get("CARM_CONTROL_METRICS_PATH", "data/control/control_version_metrics.json"))
    policy_state_path = Path("data/experience/policy_state.json")
    concept_state_path = Path("data/experience/concept_state.json")
    core_state_path = Path("data/experience/core_state.json")
    controls_path = Path(os.environ.get("CARM_CONTROLS_PATH", "data/control/runtime_controls.json"))
    control_state_path = Path(os.environ.get("CARM_CONTROL_STATE_PATH", "data/control/control_state.json"))
    control_versions_path = Path(os.environ.get("CARM_CONTROL_VERSIONS_PATH", "data/control/control_versions.jsonl"))

    consolidated = load_json(consolidated_path)
    policy_state = load_json(policy_state_path)
    concept_state = load_json(concept_state_path)
    core_state = load_json(core_state_path)
    control_metrics = load_json(control_metrics_path)
    controls = load_controls(controls_path)
    control_state = load_control_state(control_state_path)
    control_versions = load_control_versions(control_versions_path)

    print("CARM System Status")
    print(f"- episodes: {count_jsonl(experience_path)}")
    print(f"- reviews: {count_jsonl(review_path)}")
    print(f"- policy_actions: {len(policy_state.get('action_weights', {}))}")
    print(f"- concept_tokens: {len(concept_state.get('token_action_weights', {}))}")
    print(f"- core_slots: {len(core_state.get('feature_weights', {}))}")

    glance_summary = consolidated.get("glance_summary", {})
    print(f"- glance_avg_help: {glance_summary.get('average_help_rate', 0.0)}")
    print(f"- controls: {controls}")
    print(f"- current_control_version: {control_state.get('current_version', '')}")
    print(f"- previous_control_version: {control_state.get('previous_version', '')}")
    print(f"- control_history_count: {len(control_versions)}")
    print(f"- last_control_reason: {control_state.get('last_reason', '')}")
    print(f"- last_control_update: {control_state.get('last_updated_utc', '')}")
    print(f"- rollout_status: {control_state.get('rollout_status', 'stable')}")
    print(f"- candidate_version: {control_state.get('candidate_version', '')}")
    print(f"- candidate_baseline_version: {control_state.get('candidate_baseline_version', '')}")
    print(f"- candidate_episode_budget: {control_state.get('candidate_episode_budget', 0)}")
    if control_versions:
        last_entry = control_versions[-1]
        print(
            f"- last_control_actions: {','.join(str(item) for item in last_entry.get('action_types', []))}"
        )
    version_metrics = control_metrics.get("version_metrics", {})
    sample_metrics = control_metrics.get("sample_metrics", {})
    current_version = str(control_state.get("current_version", ""))
    if current_version and current_version in version_metrics:
        current_metrics = version_metrics[current_version]
        print(f"- current_version_success_rate: {current_metrics.get('success_rate', 0.0)}")
        print(f"- current_version_avg_value: {current_metrics.get('avg_value_score', 0.0)}")
        print(f"- current_version_avg_steps: {current_metrics.get('avg_step_count', 0.0)}")
    if current_version and current_version in sample_metrics:
        current_samples = sample_metrics[current_version]
        print(f"- current_version_sample_count: {current_samples.get('sample_count', 0)}")
        by_tag = current_samples.get("by_tag", {})
        if isinstance(by_tag, dict):
            for tag, metrics in list(by_tag.items())[:3]:
                if isinstance(metrics, dict):
                    print(
                        f"- sample_tag[{tag}]: count={metrics.get('sample_count', 0)} "
                        f"tool_match={metrics.get('tool_match_rate', 0.0)}"
                    )
    comparison = control_metrics.get("comparison", {})
    if comparison:
        print(f"- control_delta_success_rate: {comparison.get('delta_success_rate', 0.0)}")
        print(f"- control_delta_avg_value: {comparison.get('delta_avg_value_score', 0.0)}")
        print(f"- control_delta_avg_steps: {comparison.get('delta_avg_step_count', 0.0)}")

    slow_path_actions = consolidated.get("slow_path_actions", [])
    print(f"- slow_path_actions: {len(slow_path_actions)}")
    for action in slow_path_actions[:5]:
        if isinstance(action, dict):
            print(
                f"  * {action.get('type', '')} -> {action.get('target_module', '')}: "
                f"{action.get('proposal', '')}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
