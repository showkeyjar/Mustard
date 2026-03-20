from __future__ import annotations

import json
import os
from collections import Counter, defaultdict
from pathlib import Path

from carm.normalize import normalize_episode_payload
from carm.runtime_controls import load_control_state


def load_episodes(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []

    episodes: list[dict[str, object]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        payload = json.loads(line)
        if isinstance(payload, dict):
            episodes.append(normalize_episode_payload(payload))
    return episodes


def load_samples(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []

    samples: list[dict[str, object]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        payload = json.loads(line)
        if isinstance(payload, dict):
            samples.append(payload)
    return samples


def build_metrics(
    episodes: list[dict[str, object]],
    samples: list[dict[str, object]],
    control_state: dict[str, object],
) -> dict[str, object]:
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for episode in episodes:
        outcome = episode.get("outcome_signature", {})
        features = episode.get("episode_features", {})
        version = str(outcome.get("control_version") or features.get("control_version") or "unversioned")
        grouped[version].append(episode)

    version_metrics: dict[str, dict[str, float | int]] = {}
    sample_metrics: dict[str, dict[str, object]] = {}
    for version, version_episodes in grouped.items():
        episode_count = len(version_episodes)
        success_count = sum(1 for episode in version_episodes if episode.get("success"))
        total_value = sum(float(episode.get("value_score", 0.0) or 0.0) for episode in version_episodes)
        total_steps = sum(len(episode.get("steps", [])) for episode in version_episodes)
        total_uncertainty = sum(
            float(episode.get("outcome_signature", {}).get("uncertainty", 1.0) or 1.0)
            for episode in version_episodes
        )
        tool_episodes = sum(
            1 for episode in version_episodes if str(episode.get("episode_features", {}).get("used_tool", "")).strip()
        )
        glance_episodes = sum(
            1
            for episode in version_episodes
            if any(bool(step.get("glance_used")) for step in episode.get("steps", []))
        )
        version_metrics[version] = {
            "episode_count": episode_count,
            "success_rate": round(success_count / episode_count, 4) if episode_count else 0.0,
            "avg_value_score": round(total_value / episode_count, 4) if episode_count else 0.0,
            "avg_step_count": round(total_steps / episode_count, 4) if episode_count else 0.0,
            "avg_uncertainty": round(total_uncertainty / episode_count, 4) if episode_count else 1.0,
            "tool_usage_rate": round(tool_episodes / episode_count, 4) if episode_count else 0.0,
            "glance_usage_rate": round(glance_episodes / episode_count, 4) if episode_count else 0.0,
        }
    sample_grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for sample in samples:
        version = str(sample.get("control_version", "") or "unversioned")
        sample_grouped[version].append(sample)
    for version, version_samples in sample_grouped.items():
        by_tag: dict[str, dict[str, object]] = {}
        tags = sorted({str(sample.get("tag", "")).strip() or "untagged" for sample in version_samples})
        for tag in tags:
            tag_samples = [sample for sample in version_samples if (str(sample.get("tag", "")).strip() or "untagged") == tag]
            expected_count = sum(1 for sample in tag_samples if str(sample.get("expected_tool", "")).strip())
            matched_count = sum(1 for sample in tag_samples if bool(sample.get("tool_match")))
            tool_counter = Counter(str(sample.get("used_tool", "")).strip() or "none" for sample in tag_samples)
            by_tag[tag] = {
                "sample_count": len(tag_samples),
                "tool_match_rate": round(matched_count / expected_count, 4) if expected_count else 0.0,
                "expected_tool_count": expected_count,
                "top_used_tools": tool_counter.most_common(3),
            }
        sample_metrics[version] = {
            "sample_count": len(version_samples),
            "by_tag": by_tag,
        }

    current_version = str(control_state.get("current_version", ""))
    previous_version = str(control_state.get("previous_version", ""))
    comparison: dict[str, float | int | str] = {}
    if current_version and previous_version and current_version in version_metrics and previous_version in version_metrics:
        current = version_metrics[current_version]
        previous = version_metrics[previous_version]
        comparison = {
            "current_version": current_version,
            "previous_version": previous_version,
            "delta_success_rate": round(float(current["success_rate"]) - float(previous["success_rate"]), 4),
            "delta_avg_value_score": round(float(current["avg_value_score"]) - float(previous["avg_value_score"]), 4),
            "delta_avg_step_count": round(float(current["avg_step_count"]) - float(previous["avg_step_count"]), 4),
            "delta_avg_uncertainty": round(float(current["avg_uncertainty"]) - float(previous["avg_uncertainty"]), 4),
        }

    return {
        "episode_count": len(episodes),
        "sample_count": len(samples),
        "version_metrics": version_metrics,
        "sample_metrics": sample_metrics,
        "comparison": comparison,
    }


def main() -> int:
    experience_path = Path(os.environ.get("CARM_EXPERIENCE_PATH", "data/experience/episodes.jsonl"))
    control_state_path = Path(os.environ.get("CARM_CONTROL_STATE_PATH", "data/control/control_state.json"))
    samples_path = Path(os.environ.get("CARM_CONTROL_SAMPLES_PATH", "data/control/control_cycle_samples.jsonl"))
    output_path = Path(os.environ.get("CARM_CONTROL_METRICS_PATH", "data/control/control_version_metrics.json"))

    episodes = load_episodes(experience_path)
    samples = load_samples(samples_path)
    control_state = load_control_state(control_state_path)
    payload = build_metrics(episodes, samples, control_state)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Evaluated {len(episodes)} episode(s) into {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
