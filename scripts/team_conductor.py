from __future__ import annotations

import json
import hashlib
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from carm.pretrain_data import generate_task_pool


DEFAULT_CONFIG_PATH = Path("configs/team_cycle.json")

REQUIRED_DIRECTORIES = [
    Path("team"),
    Path("memory"),
    Path("memory/daily"),
    Path("backlog/opportunities"),
    Path("backlog/incidents"),
    Path("backlog/proposals"),
    Path("backlog/approved"),
    Path("backlog/rejected"),
    Path("backlog/shipped"),
    Path("data/research"),
    Path("data/evolution/candidates"),
    Path("data/evolution/lineages"),
    Path("data/evolution/runs"),
]


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _read_json(path: Path) -> dict[str, object]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _count_jsonl(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())


def _slug(value: str) -> str:
    lowered = value.lower()
    cleaned = "".join(char if char.isalnum() else "_" for char in lowered)
    compact = "_".join(part for part in cleaned.split("_") if part)
    return compact or "item"


def _read_existing_proposal_titles(root: Path) -> set[str]:
    titles: set[str] = set()
    proposal_dir = root / "backlog" / "proposals"
    if not proposal_dir.exists():
        return titles

    for path in proposal_dir.glob("*.md"):
        lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
        if not lines:
            continue
        first = lines[0].strip()
        if first.startswith("# ") and len(first) > 2:
            titles.add(first[2:].strip())
    return titles


def _write_if_changed(path: Path, content: str) -> None:
    if path.exists() and path.read_text(encoding="utf-8") == content:
        return
    path.write_text(content, encoding="utf-8")


def _normalize_text(value: str) -> str:
    value = value.strip().lower()
    value = re.sub(r"\s+", " ", value)
    return value


def _content_hash(path: Path) -> str:
    if not path.exists():
        return ""
    normalized = _normalize_text(path.read_text(encoding="utf-8", errors="ignore"))
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _load_jsonl(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []
    items: list[dict[str, object]] = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
            if isinstance(payload, dict):
                items.append(payload)
        except json.JSONDecodeError:
            continue
    return items


def _append_jsonl(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _sync_real_prompt_eval_pipeline(root: Path) -> dict[str, object]:
    """Build candidates -> merge into config -> run evaluation.
    Returns pipeline stats and latest real_prompt_eval payload when available.
    """
    stats: dict[str, object] = {"candidates": 0, "merged_prompt_count": 0}
    candidates_path = root / "data" / "eval" / "real_prompt_candidates.json"
    eval_config_path = root / "configs" / "real_prompt_eval.json"

    try:
        subprocess.run(["python", "-m", "scripts.build_real_prompt_candidates"], cwd=str(root), check=True, capture_output=True, text=True)
    except Exception as exc:
        stats["pipeline_error"] = f"build_candidates_failed:{exc}"
        return stats

    candidates_payload = _read_json(candidates_path)
    candidates = candidates_payload.get("prompts", []) if isinstance(candidates_payload, dict) else []
    if not isinstance(candidates, list):
        candidates = []
    stats["candidates"] = len(candidates)

    # Candidate quality filter: drop noisy observer-learning prompts and unstable tool labels
    filtered_candidates: list[dict[str, object]] = []
    dropped: list[dict[str, object]] = []
    allowed_tools = {"search", "calculator", "bigmodel_proxy"}

    def _quality_reasons(item: dict[str, object]) -> list[str]:
        prompt = str(item.get("prompt", ""))
        expected_tool = str(item.get("expected_tool", "")).strip()
        reasons: list[str] = []
        if prompt.startswith("观察学习任务"):
            reasons.append("observer_learning_noise")
        if expected_tool not in allowed_tools:
            reasons.append(f"tool_not_allowed:{expected_tool or 'empty'}")
        if len(prompt) < 8:
            reasons.append("prompt_too_short")
        return reasons

    for item in candidates:
        if not isinstance(item, dict):
            continue
        prompt = str(item.get("prompt", ""))
        expected_tool = str(item.get("expected_tool", "")).strip()
        reasons = _quality_reasons(item)

        if reasons:
            dropped.append({
                "id": str(item.get("id", "")),
                "expected_tool": expected_tool,
                "reasons": reasons,
                "prompt_preview": prompt[:80],
            })
            continue

        filtered_candidates.append(item)

    stats["filtered_candidates"] = len(filtered_candidates)
    stats["dropped_candidates"] = len(dropped)

    quality_report_path = root / "backlog" / "opportunities" / "candidate_quality_report.md"
    lines = [
        "# Candidate Quality Report",
        "",
        f"- total_candidates: {len(candidates)}",
        f"- filtered_candidates: {len(filtered_candidates)}",
        f"- dropped_candidates: {len(dropped)}",
        "",
        "## Drop Details",
    ]
    if dropped:
        for item in dropped:
            lines.append(f"- {item.get('id','')} | tool={item.get('expected_tool','')} | reasons={','.join(item.get('reasons', []))}")
    else:
        lines.append("- none")
    _write_if_changed(quality_report_path, "\n".join(lines) + "\n")
    stats["candidate_quality_report"] = str(quality_report_path)

    existing_payload = _read_json(eval_config_path)
    existing_prompts = existing_payload.get("prompts", []) if isinstance(existing_payload, dict) else []
    if not isinstance(existing_prompts, list):
        existing_prompts = []

    merged: list[dict[str, object]] = []
    seen: set[str] = set()

    def _prompt_key(item: dict[str, object]) -> str:
        return _normalize_text(str(item.get("prompt", "")))

    for item in existing_prompts + filtered_candidates:
        if not isinstance(item, dict):
            continue
        reasons = _quality_reasons(item)
        if reasons:
            continue
        key = _prompt_key(item)
        if not key or key in seen:
            continue
        seen.add(key)
        merged.append(
            {
                "id": str(item.get("id", f"merged-{len(merged)+1:03d}")),
                "prompt": str(item.get("prompt", "")),
                "expected_tool": str(item.get("expected_tool", "search")),
                "logic_skill": str(item.get("logic_skill", "tool_selection")),
            }
        )

    merged = merged[:20]
    eval_config_path.write_text(json.dumps({"prompts": merged}, ensure_ascii=False, indent=2), encoding="utf-8")
    stats["merged_prompt_count"] = len(merged)

    try:
        proc = subprocess.run(["python", "-m", "scripts.evaluate_real_prompts"], cwd=str(root), check=True, capture_output=True, text=True)
        payload = json.loads(proc.stdout)
        latest_path = root / "data" / "eval" / "real_prompt_eval_latest.json"
        latest_path.parent.mkdir(parents=True, exist_ok=True)
        latest_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        stats["real_prompt_eval_path"] = str(latest_path)
        stats["real_prompt_eval"] = payload
    except Exception as exc:
        stats["pipeline_error"] = f"evaluate_real_prompts_failed:{exc}"

    return stats


def _track_eval_config_stability(root: Path) -> dict[str, object]:
    eval_path = root / "configs" / "real_prompt_eval.json"
    history_path = root / "data" / "team" / "eval_config_history.jsonl"

    current_hash = _content_hash(eval_path)
    history = _load_jsonl(history_path)
    last_hash = ""
    if history:
        last = history[-1]
        if isinstance(last, dict):
            last_hash = str(last.get("content_hash", ""))

    changed = bool(current_hash) and bool(last_hash) and current_hash != last_hash
    payload = {
        "timestamp_utc": utc_now().isoformat(),
        "path": str(eval_path),
        "content_hash": current_hash,
        "changed_vs_last": changed,
    }
    _append_jsonl(history_path, payload)

    return {
        "eval_config_path": str(eval_path),
        "eval_config_hash": current_hash,
        "eval_config_changed_vs_last": changed,
        "eval_config_history_path": str(history_path),
    }


def _cleanup_low_value_artifacts(root: Path) -> dict[str, int]:
    removed_research = 0
    opportunities = root / "backlog" / "opportunities"
    if opportunities.exists():
        for p in opportunities.glob("research_20*.md"):
            try:
                p.unlink()
                removed_research += 1
            except OSError:
                pass

    removed_proposals = 0
    proposals_dir = root / "backlog" / "proposals"
    if proposals_dir.exists():
        files = sorted(proposals_dir.glob("*.md"), key=lambda x: x.stat().st_mtime, reverse=True)
        seen: set[str] = set()
        for p in files:
            lines = p.read_text(encoding="utf-8", errors="ignore").splitlines()
            title = p.stem
            if lines:
                first = lines[0].strip()
                if first.startswith("# "):
                    title = first[2:].strip()
            key = _slug(title)
            target = proposals_dir / f"{key}.md"
            if key in seen:
                try:
                    p.unlink()
                    removed_proposals += 1
                except OSError:
                    pass
                continue
            seen.add(key)
            if p != target:
                if target.exists():
                    try:
                        p.unlink()
                        removed_proposals += 1
                    except OSError:
                        pass
                else:
                    p.rename(target)

    return {"removed_research": removed_research, "removed_proposals": removed_proposals}


def _build_signal_signature(signals: dict[str, object]) -> str:
    real_prompt_match = 0.0
    real_prompt_count = 0
    real_prompt_eval = signals.get("real_prompt_eval", {})
    if isinstance(real_prompt_eval, dict):
        summary = real_prompt_eval.get("summary", {})
        if isinstance(summary, dict):
            real_prompt_match = float(summary.get("pretrained_match_rate", 0.0) or 0.0)
            real_prompt_count = int(summary.get("prompt_count", 0) or 0)

    return "|".join(
        [
            str(int(signals.get("dataset_sample_count", 0) or 0)),
            str(int(signals.get("bridge_feedback", 0) or 0)),
            str(int(signals.get("frontier_observation_count", 0) or 0)),
            str(real_prompt_count),
            f"{real_prompt_match:.4f}",
        ]
    )


def _load_recursive_state(root: Path) -> dict[str, object]:
    path = root / "data" / "team" / "recursive_state.json"
    if not path.exists():
        return {"last_signature": "", "stagnation_rounds": 0, "last_mode": "normal"}
    payload = _read_json(path)
    if not isinstance(payload, dict):
        return {"last_signature": "", "stagnation_rounds": 0, "last_mode": "normal"}
    return payload


def _update_recursive_state(root: Path, signals: dict[str, object], config: dict[str, object]) -> dict[str, object]:
    policy = config.get("recursive_policy", {})
    if not isinstance(policy, dict):
        policy = {}

    enabled = bool(policy.get("enabled", True))
    pivot_rounds = int(policy.get("stagnation_rounds_to_pivot", 2) or 2)
    force_rounds = int(policy.get("force_max_landing_after_rounds", 3) or 3)

    state = _load_recursive_state(root)
    signature = _build_signal_signature(signals)
    last_signature = str(state.get("last_signature", ""))
    stagnation_rounds = int(state.get("stagnation_rounds", 0) or 0)

    if not enabled:
        mode = "normal"
        stagnation_rounds = 0
    else:
        stagnation_rounds = stagnation_rounds + 1 if signature == last_signature else 0
        if stagnation_rounds >= force_rounds:
            mode = "max_landing"
        elif stagnation_rounds >= pivot_rounds:
            mode = "pivot"
        else:
            mode = "normal"

    next_state = {
        "last_signature": signature,
        "stagnation_rounds": stagnation_rounds,
        "last_mode": mode,
        "updated_at_utc": utc_now().isoformat(),
    }

    target = root / "data" / "team" / "recursive_state.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    _write_if_changed(target, json.dumps(next_state, ensure_ascii=False, indent=2) + "\n")
    return next_state


def load_team_config(config_path: Path = DEFAULT_CONFIG_PATH) -> dict[str, object]:
    config = _read_json(config_path)
    if config:
        return config
    return {
        "team_name": "mustard-claw",
        "heartbeat": {"daily_digest": True, "max_new_proposals_per_cycle": 3},
        "risk_policy": {
            "human_gate_required_change_types": [
                "runtime_control",
                "desktop_behavior",
                "training_data_policy",
                "default_model_or_tooling",
            ]
        },
        "recursive_policy": {
            "enabled": True,
            "stagnation_rounds_to_pivot": 2,
            "force_max_landing_after_rounds": 3,
        },
        "eval_pipeline": {
            "auto_sync_real_prompt_eval": False,
            "min_new_candidates_to_merge": 4,
        },
        "focus_areas": [],
    }


def bootstrap_workspace(root: Path = Path(".")) -> None:
    for directory in REQUIRED_DIRECTORIES:
        (root / directory).mkdir(parents=True, exist_ok=True)


def collect_signals(root: Path = Path(".")) -> dict[str, object]:
    train_report = _read_json(root / "data/train_runs/auto_train_latest.json")
    control_state = _read_json(root / "data/control/control_state.json")
    control_metrics = _read_json(root / "data/control/control_version_metrics.json")

    evaluation = train_report.get("evaluation", {})
    if not isinstance(evaluation, dict):
        evaluation = {}
    pretrain_eval = evaluation.get("pretrain_eval", {})
    if not isinstance(pretrain_eval, dict):
        pretrain_eval = {}
    real_prompt_eval = evaluation.get("real_prompt_eval", {})
    if not isinstance(real_prompt_eval, dict):
        real_prompt_eval = {}

    latest_real_prompt_eval = _read_json(root / "data/eval/real_prompt_eval_latest.json")
    if isinstance(latest_real_prompt_eval, dict) and latest_real_prompt_eval.get("summary"):
        real_prompt_eval = latest_real_prompt_eval

    comparison = control_metrics.get("comparison", {})
    if not isinstance(comparison, dict):
        comparison = {}

    hard_bridge_feedback = _count_jsonl(root / "data/desktop/bridge_feedback.jsonl")
    soft_bridge_feedback = _count_jsonl(root / "data/research/soft_bridge_feedback.jsonl")
    frontier_observation_count = _count_jsonl(root / "data/research/frontier_observations.jsonl")
    recovery_variants_payload = _read_json(root / "data/evolution/research_recovery_variants.json")
    recovery_variants = recovery_variants_payload.get("variants", []) if isinstance(recovery_variants_payload, dict) else []
    if not isinstance(recovery_variants, list):
        recovery_variants = []

    return {
        "episodes": _count_jsonl(root / "data/experience/episodes.jsonl"),
        "reviews": _count_jsonl(root / "data/review/reviews.jsonl"),
        "bridge_feedback": hard_bridge_feedback + soft_bridge_feedback,
        "bridge_feedback_hard": hard_bridge_feedback,
        "bridge_feedback_soft": soft_bridge_feedback,
        "frontier_observation_count": frontier_observation_count,
        "research_recovery_variant_count": len(recovery_variants),
        "latest_train_run_id": train_report.get("run_id", ""),
        "dataset_sample_count": train_report.get("dataset", {}).get("sample_count", 0) if isinstance(train_report.get("dataset", {}), dict) else 0,
        "pretrain_eval": pretrain_eval,
        "real_prompt_eval": real_prompt_eval,
        "control_state": control_state,
        "control_comparison": comparison,
    }


def _arbiter_direction_review(signals: dict[str, object], config: dict[str, object]) -> dict[str, object]:
    decision_policy = config.get("decision_policy", {})
    if not isinstance(decision_policy, dict):
        decision_policy = {}

    min_real_prompt = float(decision_policy.get("min_real_prompt_match_rate", 0.9) or 0.9)
    min_pretrain = float(decision_policy.get("min_pretrain_tool_match_rate", 0.95) or 0.95)

    reasons: list[str] = []
    adjust_reasons: list[str] = []

    pretrain_eval = signals.get("pretrain_eval", {})
    if isinstance(pretrain_eval, dict):
        pretrained = pretrain_eval.get("pretrained", {})
        if isinstance(pretrained, dict):
            pretrain_match = float(pretrained.get("tool_match_rate", 0.0) or 0.0)
            if pretrain_match < min_pretrain:
                reasons.append(f"pretrain_tool_match_below_threshold:{pretrain_match:.4f}<{min_pretrain:.2f}")

    real_prompt_eval = signals.get("real_prompt_eval", {})
    if isinstance(real_prompt_eval, dict):
        summary = real_prompt_eval.get("summary", {})
        if isinstance(summary, dict):
            real_prompt_match = float(summary.get("pretrained_match_rate", 0.0) or 0.0)
            if real_prompt_match < min_real_prompt:
                reasons.append(f"real_prompt_match_below_threshold:{real_prompt_match:.4f}<{min_real_prompt:.2f}")

    control_state = signals.get("control_state", {})
    if isinstance(control_state, dict) and str(control_state.get("rollout_status", "stable")) == "candidate":
        reasons.append("runtime_candidate_active")

    research_quality = signals.get("research_quality", {})
    if not isinstance(research_quality, dict):
        research_quality = {}
    recursive_state = signals.get("recursive_state", {})
    if not isinstance(recursive_state, dict):
        recursive_state = {}

    stagnation_rounds = int(recursive_state.get("stagnation_rounds", 0) or 0)
    new_failure_pattern_count = int(research_quality.get("new_failure_pattern_count", 0) or 0)
    quality_exploration_active = bool(research_quality.get("quality_exploration_active", False))
    high_signal_count = int(research_quality.get("high_signal_count", 0) or 0)

    if stagnation_rounds >= 10 and new_failure_pattern_count == 0:
        adjust_reasons.append(f"stagnation_without_new_failure_pattern:{stagnation_rounds}")

    if stagnation_rounds >= 10 and quality_exploration_active and high_signal_count > 0:
        adjust_reasons.append("exploration_without_direction_change")

    researcher_artifact_text = str(signals.get("researcher_artifact_text", "") or "").lower()
    if "blind spot remains" in researcher_artifact_text or "sampling blind spot" in researcher_artifact_text:
        adjust_reasons.append("blind_spot_persists")

    if reasons:
        verdict = "uncertain_needs_human"
    elif adjust_reasons:
        verdict = "direction_adjust"
    else:
        verdict = "direction_correct"

    all_reasons = reasons + [reason for reason in adjust_reasons if reason not in reasons]
    return {
        "owner": str(decision_policy.get("owner", "arbiter")),
        "verdict": verdict,
        "reasons": all_reasons,
        "escalate_to_human": verdict == "uncertain_needs_human" and bool(decision_policy.get("escalate_on_uncertain_direction", True)),
    }


def _evaluate_deep_cycle(signals: dict[str, object], config: dict[str, object], proposal_count: int) -> list[str]:
    policy = config.get("deep_cycle_policy", {})
    if not isinstance(policy, dict) or not bool(policy.get("enabled", False)):
        return []

    failures: list[str] = []

    if bool(policy.get("require_at_least_one_proposal", True)) and proposal_count <= 0:
        failures.append("no_actionable_proposal")

    real_prompt_eval = signals.get("real_prompt_eval", {})
    summary = real_prompt_eval.get("summary", {}) if isinstance(real_prompt_eval, dict) else {}
    if not isinstance(summary, dict):
        summary = {}

    prompt_count = int(summary.get("prompt_count", 0) or 0)
    min_prompt_count = int(policy.get("min_real_prompt_count", 20) or 20)
    if prompt_count < min_prompt_count:
        failures.append(f"real_prompt_count_too_low:{prompt_count}<{min_prompt_count}")

    if bool(policy.get("require_positive_real_prompt_delta", True)):
        real_delta = float((real_prompt_eval or {}).get("delta_tool_match_rate", 0.0) or 0.0) if isinstance(real_prompt_eval, dict) else 0.0

        quality_positive_signal = False
        quality_focus_eval = signals.get("quality_focus_eval_result", {})
        if isinstance(quality_focus_eval, dict):
            focus_payload = quality_focus_eval.get("payload", {})
            focus_summary = focus_payload.get("summary", {}) if isinstance(focus_payload, dict) else {}
            if isinstance(focus_summary, dict):
                focus_baseline = float(focus_summary.get("baseline_match_rate", 0.0) or 0.0)
                focus_pretrained = float(focus_summary.get("pretrained_match_rate", 0.0) or 0.0)
                if focus_pretrained > focus_baseline:
                    quality_positive_signal = True

        quality_stabilization = signals.get("quality_stabilization", {})
        if isinstance(quality_stabilization, dict):
            high_signal_count = int(quality_stabilization.get("high_signal_count", 0) or 0)
            if high_signal_count > 0:
                quality_positive_signal = True

        if real_delta <= 0.0 and not quality_positive_signal:
            failures.append(f"real_prompt_delta_not_positive:{real_delta:.4f}")

    if bool(policy.get("require_non_negative_control_success_delta", True)):
        control_comparison = signals.get("control_comparison", {})
        success_delta = float(control_comparison.get("delta_success_rate", 0.0) or 0.0) if isinstance(control_comparison, dict) else 0.0
        if success_delta < 0.0:
            failures.append(f"control_success_regressed:{success_delta:.4f}")

    return failures


def build_daily_digest(signals: dict[str, object], config: dict[str, object]) -> dict[str, object]:
    focus_areas = config.get("focus_areas", [])
    if not isinstance(focus_areas, list):
        focus_areas = []

    alerts: list[str] = []
    real_prompt_eval = signals.get("real_prompt_eval", {})
    if isinstance(real_prompt_eval, dict) and real_prompt_eval:
        delta = float(real_prompt_eval.get("delta_tool_match_rate", 0.0) or 0.0)
        if delta < 0:
            alerts.append("real_prompt_tool_match_regressed")

    control_comparison = signals.get("control_comparison", {})
    if isinstance(control_comparison, dict):
        success_delta = float(control_comparison.get("delta_success_rate", 0.0) or 0.0)
        if success_delta < 0:
            alerts.append("runtime_control_success_regressed")

    control_state = signals.get("control_state", {})
    if isinstance(control_state, dict) and str(control_state.get("rollout_status", "stable")) == "candidate":
        alerts.append("candidate_rollout_active")

    direction_review = _arbiter_direction_review(signals, config)
    if bool(direction_review.get("escalate_to_human", False)):
        alerts.append("arbiter_uncertain_needs_human")

    return {
        "timestamp_utc": utc_now().isoformat(),
        "team_name": config.get("team_name", "mustard-claw"),
        "focus_areas": focus_areas,
        "signals": signals,
        "alerts": alerts,
        "direction_review": direction_review,
    }


def _build_proactive_proposals(signals: dict[str, object], gated_types: list[object]) -> list[dict[str, object]]:
    proposals: list[dict[str, object]] = []

    dataset_sample_count = int(signals.get("dataset_sample_count", 0) or 0)
    if dataset_sample_count < 250:
        proposals.append(
            {
                "title": "Scale pretraining corpus from real episodes",
                "problem": "当前预训练样本规模偏小，可能限制两段式学习上限。",
                "evidence": [f"dataset_sample_count={dataset_sample_count}"],
                "change_type": "evaluation_or_dataset",
                "proposed_change": "执行自动训练流水线并将高价值真实 episode 回流到预训练集，提升创新链路覆盖。",
                "risk_level": "low",
                "evaluation_plan": [
                    "python -m scripts.auto_train",
                    "python -m scripts.evaluate_pretraining",
                    "python -m scripts.evaluate_real_prompts",
                ],
                "rollback_plan": "仅新增训练产物与评测报告，不修改默认运行时策略。",
                "needs_human_approval": "evaluation_or_dataset" in gated_types,
            }
        )

    real_prompt_eval = signals.get("real_prompt_eval", {})
    prompt_count = 0
    if isinstance(real_prompt_eval, dict):
        summary = real_prompt_eval.get("summary", {})
        if isinstance(summary, dict):
            prompt_count = int(summary.get("prompt_count", 0) or 0)
    if prompt_count < 12:
        proposals.append(
            {
                "title": "Expand real-prompt regression coverage",
                "problem": "真实回归样本覆盖偏窄，难以持续发现创新能力退化。",
                "evidence": [f"real_prompt_prompt_count={prompt_count}"],
                "change_type": "evaluation_or_dataset",
                "proposed_change": "从近期高价值 episode 构建候选真实回归集，并合并到 real_prompt_eval 基准。",
                "risk_level": "low",
                "evaluation_plan": [
                    "python -m scripts.build_real_prompt_candidates",
                    "python -m scripts.evaluate_real_prompts",
                ],
                "rollback_plan": "仅修改评测集配置，可回退到上一版配置文件。",
                "needs_human_approval": "evaluation_or_dataset" in gated_types,
            }
        )

    bridge_feedback = int(signals.get("bridge_feedback", 0) or 0)
    if bridge_feedback == 0:
        proposals.append(
            {
                "title": "Kickstart desktop-bridge feedback loop",
                "problem": "桌面桥梁暂无反馈样本，创新闭环缺少用户纠偏信号。",
                "evidence": [f"bridge_feedback={bridge_feedback}"],
                "change_type": "desktop_behavior",
                "proposed_change": "优先采集并整理 bridge useful/misread 反馈，形成可训练的反馈样本包。",
                "risk_level": "medium",
                "evaluation_plan": [
                    "python -m scripts.desktop_agent_control snapshot",
                    "python -m scripts.desktop_bridge_chat",
                ],
                "rollback_plan": "仅进行反馈采集与标注，不调整桌面采样或主动追问默认策略。",
                "needs_human_approval": "desktop_behavior" in gated_types,
            }
        )

    frontier_observation_count = int(signals.get("frontier_observation_count", 0) or 0)
    if frontier_observation_count < 2:
        proposals.append(
            {
                "title": "Track frontier reasoning-small-model updates",
                "problem": "前沿小模型研究观察样本不足，容易重复踩坑。",
                "evidence": [f"frontier_observation_count={frontier_observation_count}"],
                "change_type": "research_tracking",
                "proposed_change": "Researcher 本周补齐 DeepSeek / MiniMax 等前沿观察，并形成可借鉴结论标签。",
                "risk_level": "low",
                "evaluation_plan": [
                    "更新 docs/dev/research_frontier_watchlist.md",
                    "沉淀观察到 data/research/frontier_observations.jsonl",
                ],
                "rollback_plan": "仅新增研究记录，不影响运行时。",
                "needs_human_approval": "research_tracking" in gated_types,
            }
        )

    return proposals


def _build_adaptive_proposals(signals: dict[str, object], gated_types: list[object], recursive_state: dict[str, object]) -> list[dict[str, object]]:
    proposals: list[dict[str, object]] = []

    real_prompt_eval = signals.get("real_prompt_eval", {})
    rows: list[dict[str, object]] = []
    summary: dict[str, object] = {}
    if isinstance(real_prompt_eval, dict):
        raw_rows = real_prompt_eval.get("rows", [])
        if isinstance(raw_rows, list):
            rows = [r for r in raw_rows if isinstance(r, dict)]
        raw_summary = real_prompt_eval.get("summary", {})
        if isinstance(raw_summary, dict):
            summary = raw_summary

    mismatches = [
        r for r in rows
        if ("pretrained_match" in r and not bool(r.get("pretrained_match", True)))
        or ("pretrained_tool_match" in r and not bool(r.get("pretrained_tool_match", True)))
    ]

    if mismatches:
        top = mismatches[0]
        sample_id = str(top.get("id", "unknown"))
        logic_skill = str(top.get("logic_skill", "unknown"))
        expected_tool = str(top.get("expected_tool", ""))
        actual_tool = str(top.get("pretrained_used_tool", top.get("pretrained_used_tool", "")))
        proposals.append(
            {
                "title": f"Fix mismatch path {sample_id} ({logic_skill})",
                "problem": "真实回归中出现明确的工具路径错配，说明当前策略对该场景泛化不足。",
                "evidence": [
                    f"sample_id={sample_id}",
                    f"logic_skill={logic_skill}",
                    f"expected_tool={expected_tool}",
                    f"actual_tool={actual_tool}",
                ],
                "from_failure_pattern": "eval_coverage_too_low",
                "from_top_gap": "eval_coverage_too_low",
                "change_type": "evaluation_or_dataset",
                "proposed_change": "围绕该错配样例补充 2-3 条同类对抗样本，并修正候选构建的工具标签规则后重跑评测。",
                "expected_metric_delta": "该 logic_skill 的 tool_match_rate 提升 >=0.1，且 mismatch_case_count 下降",
                "risk_level": "low",
                "evaluation_plan": [
                    "python -m scripts.build_real_prompt_candidates",
                    "python -m scripts.evaluate_real_prompts",
                ],
                "rollback_plan": "回退新增样本与标签变更，恢复上一版 real_prompt_eval 配置。",
                "relative_to_last_round": f"从泛化覆盖提案切换为针对 {sample_id} 的精确修复。",
                "scenario_fit": "用户在真实任务里遇到工具选错/步骤错配时的高频场景。",
                "needs_human_approval": "evaluation_or_dataset" in gated_types,
            }
        )

    prompt_count = int(summary.get("prompt_count", 0) or 0)
    target_count = 20
    if prompt_count < target_count:
        proposals.append(
            {
                "title": f"Expand real prompt set from {prompt_count} to {target_count}",
                "problem": "真实回归样本覆盖不足，导致团队在同一窄样本上反复提出相似方案。",
                "evidence": [f"real_prompt_prompt_count={prompt_count}", f"target={target_count}"],
                "from_failure_pattern": "eval_coverage_too_low",
                "from_top_gap": "eval_coverage_too_low",
                "change_type": "evaluation_or_dataset",
                "proposed_change": "按 logic_skill 分桶补齐样本（每桶至少 2 条），优先补当前低覆盖桶。",
                "expected_metric_delta": "prompt_count 达到 20，且 by_logic_skill 覆盖更均衡",
                "risk_level": "low",
                "evaluation_plan": [
                    "python -m scripts.build_real_prompt_candidates",
                    "python -m scripts.evaluate_real_prompts",
                ],
                "rollback_plan": "仅回退新增样本，不触碰运行时默认策略。",
                "relative_to_last_round": "从模板化扩容改为按 logic_skill 定向扩容。",
                "scenario_fit": "真实多任务场景下，避免模型只对少数模板题有效。",
                "needs_human_approval": "evaluation_or_dataset" in gated_types,
            }
        )

    return proposals


def _proposal_topic_key(proposal: dict[str, object]) -> str:
    text = " ".join(
        [
            str(proposal.get("title", "")),
            str(proposal.get("problem", "")),
            str(proposal.get("proposed_change", "")),
            str(proposal.get("from_failure_pattern", "")),
            str(proposal.get("from_top_gap", "")),
        ]
    ).lower()
    if any(token in text for token in ["bridge", "feedback", "desktop"]):
        return "bridge_feedback"
    if any(token in text for token in ["frontier", "research intake", "watchlist", "observation"]):
        return "frontier_intake"
    if any(token in text for token in ["sampling blind spot", "high-information", "real prompt", "coverage"]):
        return "real_prompt_sampling"
    if any(token in text for token in ["runtime rollout", "candidate version", "runtime_control"]):
        return "runtime_control"
    return str(proposal.get("title", "")).strip().lower()


def _proposal_information_score(proposal: dict[str, object]) -> int:
    score = 0
    if proposal.get("from_failure_pattern"):
        score += 3
    if proposal.get("from_top_gap"):
        score += 2
    if proposal.get("expected_metric_delta"):
        score += 2
    if proposal.get("relative_to_last_round"):
        score += 1
    if proposal.get("scenario_fit"):
        score += 1
    if proposal.get("architect_handoff") and proposal.get("architect_handoff") != "direct_execute_if_format_passes":
        score += 2
    if proposal.get("needs_human_approval"):
        score += 1
    evaluation_plan = proposal.get("evaluation_plan", [])
    if isinstance(evaluation_plan, list):
        score += min(2, len(evaluation_plan))
    evidence = proposal.get("evidence", [])
    if isinstance(evidence, list) and evidence:
        score += 1
    return score


def _dedupe_and_prioritize_proposals(proposals: list[dict[str, object]]) -> list[dict[str, object]]:
    best_by_topic: dict[str, dict[str, object]] = {}
    for proposal in proposals:
        key = _proposal_topic_key(proposal)
        current = best_by_topic.get(key)
        if current is None:
            best_by_topic[key] = proposal
            continue
        current_score = _proposal_information_score(current)
        next_score = _proposal_information_score(proposal)
        if next_score > current_score:
            best_by_topic[key] = proposal
        elif next_score == current_score:
            if bool(proposal.get("needs_human_approval", False)) and not bool(current.get("needs_human_approval", False)):
                best_by_topic[key] = proposal

    ordered_topics = []
    seen = set()
    for proposal in proposals:
        key = _proposal_topic_key(proposal)
        if key not in seen:
            ordered_topics.append(key)
            seen.add(key)
    return [best_by_topic[key] for key in ordered_topics if key in best_by_topic]


def _load_active_failure_pattern_ids(root: Path) -> set[str]:
    path = root / "backlog" / "incidents" / "auto_failure_patterns.json"
    if not path.exists():
        return set()
    payload = _read_json(path)
    patterns = payload.get("patterns", []) if isinstance(payload, dict) else []
    active: set[str] = set()
    if isinstance(patterns, list):
        for item in patterns:
            if isinstance(item, dict):
                value = str(item.get("id", "")).strip()
                if value:
                    active.add(value)
    return active


def _load_research_recovery_state(root: Path) -> dict[str, object]:
    path = root / "data" / "team" / "research_recovery_state.json"
    if not path.exists():
        return {
            "frontier_intake_runs": 0,
            "soft_feedback_runs": 0,
            "high_information_sampling_runs": 0,
            "last_triggered": [],
            "updated_at_utc": "",
        }
    payload = _read_json(path)
    if not isinstance(payload, dict):
        return {
            "frontier_intake_runs": 0,
            "soft_feedback_runs": 0,
            "high_information_sampling_runs": 0,
            "last_triggered": [],
            "updated_at_utc": "",
        }
    return payload


def _select_research_recovery_operators(
    research_quality: dict[str, object],
    active_patterns: set[str],
    config: dict[str, object],
) -> list[str]:
    policy = config.get("research_recovery_policy", {})
    if not isinstance(policy, dict) or not bool(policy.get("enabled", False)):
        return []
    reasons = research_quality.get("reasons", []) if isinstance(research_quality, dict) else []
    if not isinstance(reasons, list):
        reasons = []
    operators: list[str] = []
    if (
        bool(policy.get("allow_frontier_intake_operator", True))
        and ("frontier_zero_signal_persistence" in reasons or "frontier_research_blindspot" in active_patterns)
    ):
        operators.append("frontier_intake_operator")
    if (
        bool(policy.get("allow_soft_feedback_operator", True))
        and ("bridge_zero_feedback_persistence" in reasons or "no_tool_feedback_loop" in active_patterns)
    ):
        operators.append("soft_feedback_operator")
    if (
        bool(policy.get("allow_high_information_sampling_operator", True))
        and ("no_new_failure_pattern" in reasons or "sampling_blind_spot" in active_patterns)
    ):
        operators.append("high_information_sampling_operator")
    return operators


def _run_frontier_intake_operator(root: Path, config: dict[str, object], signals: dict[str, object]) -> dict[str, object]:
    policy = config.get("research_recovery_policy", {}) if isinstance(config, dict) else {}
    max_records = int(policy.get("max_frontier_records_per_round", 3) or 3)
    path = root / "data" / "research" / "frontier_observations.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    topics = [
        "small reasoning model routing",
        "tool-use stability under ambiguity",
        "conflict-aware answer suppression",
    ]
    generated = 0
    for idx, topic in enumerate(topics[:max_records], start=1):
        _append_jsonl(path, {
            "id": f"frontier-auto-{utc_now().strftime('%Y%m%d%H%M%S')}-{idx}",
            "topic": topic,
            "source": "auto_recovery_operator",
            "label": "pending_label",
            "reason": "frontier_zero_signal_persistence",
            "created_at_utc": utc_now().isoformat(),
        })
        generated += 1
    return {"operator": "frontier_intake_operator", "generated_count": generated, "path": str(path)}


def _run_soft_feedback_operator(root: Path, signals: dict[str, object], config: dict[str, object]) -> dict[str, object]:
    policy = config.get("research_recovery_policy", {}) if isinstance(config, dict) else {}
    max_records = int(policy.get("max_soft_feedback_records_per_round", 5) or 5)
    path = root / "data" / "research" / "soft_bridge_feedback.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    real_prompt_eval = signals.get("real_prompt_eval", {})
    if isinstance(real_prompt_eval, dict):
        raw_rows = real_prompt_eval.get("rows", [])
        if isinstance(raw_rows, list):
            rows = [item for item in raw_rows if isinstance(item, dict)]
    generated = 0
    for row in rows:
        matched = row.get("baseline_match")
        if matched is True:
            continue
        kind = "misread"
        if str(row.get("logic_skill", "")) == "conflict_detection":
            kind = "missing_context"
        elif str(row.get("logic_skill", "")) == "tool_selection":
            kind = "overreach"
        _append_jsonl(path, {
            "id": f"soft-feedback-{row.get('id','unknown')}",
            "kind": kind,
            "source_case": str(row.get("id", "unknown")),
            "reason": f"baseline used {row.get('baseline_used_tool','')} while expected {row.get('expected_tool','')}",
            "created_at_utc": utc_now().isoformat(),
        })
        generated += 1
        if generated >= max_records:
            break
    return {"operator": "soft_feedback_operator", "generated_count": generated, "path": str(path)}


def _mutation_strategies_for_logic_skill(logic_skill: str) -> list[str]:
    mapping = {
        "conflict_detection": ["contradictory_authority", "missing_evidence", "conflicting_sources"],
        "comparison": ["ranking_flip", "partial_overlap", "conflicting_sources"],
        "tool_selection": ["calculator_vs_search", "code_vs_search", "tool_boundary_shift"],
        "termination_judgment": ["ambiguous_stop", "one_more_step_needed", "premature_finish"],
        "result_integration": ["cross_source_summary", "missing_evidence", "partial_merge"],
    }
    return mapping.get(logic_skill, ["conflicting_sources", "missing_evidence", "tool_boundary_shift"])


def _select_high_information_parent_rows(signals: dict[str, object], limit: int) -> list[dict[str, object]]:
    source_dataset = "real_prompt_eval"
    payload = signals.get("quality_stabilization", {})
    rows = payload.get("rows", []) if isinstance(payload, dict) else []
    if isinstance(rows, list) and rows:
        source_dataset = "quality_stabilization"
    if not isinstance(rows, list) or not rows:
        focus_eval = signals.get("quality_focus_eval_result", {})
        if isinstance(focus_eval, dict):
            focus_payload = focus_eval.get("payload", {})
            if isinstance(focus_payload, dict):
                focus_rows = focus_payload.get("rows", [])
                if isinstance(focus_rows, list) and focus_rows:
                    rows = focus_rows
                    source_dataset = "quality_focus_eval_result"
    if not isinstance(rows, list) or not rows:
        try:
            focus_path = Path(str(signals.get("quality_focus_eval_result_path", "")))
            if focus_path.exists():
                focus_payload = _read_json(focus_path)
                rows = focus_payload.get("rows", []) if isinstance(focus_payload, dict) else []
                if isinstance(rows, list) and rows:
                    source_dataset = "quality_focus_eval_result"
        except Exception:
            rows = rows if isinstance(rows, list) else []
    if not isinstance(rows, list) or not rows:
        real_prompt_eval = signals.get("real_prompt_eval", {})
        rows = real_prompt_eval.get("rows", []) if isinstance(real_prompt_eval, dict) else []
        source_dataset = "real_prompt_eval"
    candidates = []
    for item in rows:
        if isinstance(item, dict):
            enriched = dict(item)
            enriched.setdefault("source_dataset", source_dataset)
            candidates.append(enriched)
    preferred_skills = {
        "conflict_detection": 5,
        "comparison": 4,
        "tool_selection": 3,
        "termination_judgment": 2,
        "result_integration": 1,
    }

    def _rank(row: dict[str, object]) -> tuple[int, int, int]:
        baseline_fail_pretrained_pass = int(not bool(row.get("baseline_match", True)) and bool(row.get("pretrained_match", True)))
        separation = int(row.get("separation_score", 0) or 0)
        skill_bonus = preferred_skills.get(str(row.get("logic_skill", "")), 0)
        return (baseline_fail_pretrained_pass, separation, skill_bonus)

    ranked = sorted(candidates, key=_rank, reverse=True)
    selected = ranked[:limit]
    for index, row in enumerate(selected, start=1):
        row.setdefault("source_rank", index)
    return selected


def _cluster_recovery_variants(variants: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[str, dict[str, object]] = {}
    for item in variants:
        if not isinstance(item, dict):
            continue
        logic_skill = str(item.get("logic_skill", "")).strip()
        mutation = str(item.get("mutation", "")).strip()
        source_score = int(item.get("source_score", 0) or 0)
        if logic_skill == "conflict_detection":
            pattern_id = "repeated_conflict_detection_gap"
        elif logic_skill == "tool_selection":
            pattern_id = "tool_boundary_sampling_gap"
        elif logic_skill == "comparison":
            pattern_id = "comparison_under_conflicting_sources"
        elif logic_skill == "termination_judgment":
            pattern_id = "termination_judgment_sampling_gap"
        else:
            pattern_id = f"{logic_skill or 'generic'}_sampling_gap"
        cluster = grouped.setdefault(
            pattern_id,
            {
                "pattern_id": pattern_id,
                "logic_skill": logic_skill or "generic",
                "count": 0,
                "avg_source_score": 0.0,
                "top_mutations": [],
            },
        )
        cluster["count"] = int(cluster.get("count", 0) or 0) + 1
        previous_total = float(cluster.get("avg_source_score", 0.0) or 0.0) * (cluster["count"] - 1)
        cluster["avg_source_score"] = round((previous_total + source_score) / cluster["count"], 2)
        mutations = cluster.get("top_mutations", [])
        if isinstance(mutations, list) and mutation and mutation not in mutations:
            mutations.append(mutation)
            cluster["top_mutations"] = mutations[:3]
    return list(grouped.values())


def _run_high_information_sampling_operator(root: Path, signals: dict[str, object], config: dict[str, object]) -> dict[str, object]:
    policy = config.get("research_recovery_policy", {}) if isinstance(config, dict) else {}
    max_records = int(policy.get("max_high_information_variants_per_round", 4) or 4)
    path = root / "data" / "evolution" / "research_recovery_variants.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = _select_high_information_parent_rows(signals, max_records)
    variants: list[dict[str, object]] = []
    for idx, row in enumerate(rows):
        logic_skill = str(row.get("logic_skill", "comparison"))
        mutations = _mutation_strategies_for_logic_skill(logic_skill)
        mutation = mutations[idx % len(mutations)]
        baseline_fail_pretrained_pass = not bool(row.get("baseline_match", True)) and bool(row.get("pretrained_match", True))
        variants.append({
            "id": f"recovery-variant-{idx+1:02d}",
            "parent_case": str(row.get("id", f"case-{idx+1}")),
            "logic_skill": logic_skill,
            "mutation": mutation,
            "mutation_reason": f"parent had logic_skill={logic_skill} and high recovery value under {mutation}",
            "source_score": int(row.get("separation_score", 0) or 0),
            "selection_reason": "baseline_fail_pretrained_pass" if baseline_fail_pretrained_pass else "high_signal_parent",
            "source_dataset": str(row.get("source_dataset", "real_prompt_eval")),
            "source_rank": int(row.get("source_rank", idx + 1) or (idx + 1)),
            "expected_tool": str(row.get("expected_tool", "search")),
        })
    clusters = _cluster_recovery_variants(variants)
    path.write_text(json.dumps({"variants": variants, "clusters": clusters}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {"operator": "high_information_sampling_operator", "generated_count": len(variants), "path": str(path), "clusters": clusters}


def build_proposals(
    digest: dict[str, object],
    config: dict[str, object],
    recursive_state: dict[str, object] | None = None,
) -> list[dict[str, object]]:
    risk_policy = config.get("risk_policy", {})
    if not isinstance(risk_policy, dict):
        risk_policy = {}
    gated_types = risk_policy.get("human_gate_required_change_types", [])
    if not isinstance(gated_types, list):
        gated_types = []

    proposals: list[dict[str, object]] = []
    signals = digest.get("signals", {})
    if not isinstance(signals, dict):
        return proposals

    recursive_mode = "normal"
    stagnation_rounds = 0
    if isinstance(recursive_state, dict):
        recursive_mode = str(recursive_state.get("last_mode", "normal"))
        stagnation_rounds = int(recursive_state.get("stagnation_rounds", 0) or 0)

    research_quality = digest.get("research_quality", {})
    if not isinstance(research_quality, dict):
        research_quality = {}
    new_failure_pattern_ids = research_quality.get("new_failure_pattern_ids", [])
    if not isinstance(new_failure_pattern_ids, list):
        new_failure_pattern_ids = []
    active_failure_pattern_ids = set(new_failure_pattern_ids)
    active_failure_pattern_ids.update(_load_active_failure_pattern_ids(Path(digest.get("workspace_root", "."))))

    reasons = research_quality.get("reasons", [])
    if not isinstance(reasons, list):
        reasons = []
    if "bridge_zero_feedback_persistence" in reasons:
        active_failure_pattern_ids.add("no_tool_feedback_loop")
    if "frontier_zero_signal_persistence" in reasons:
        active_failure_pattern_ids.add("frontier_research_blindspot")
    if "no_new_failure_pattern" in reasons:
        active_failure_pattern_ids.add("sampling_blind_spot")
    for specific_pattern in [
        "repeated_conflict_detection_gap",
        "tool_boundary_sampling_gap",
        "comparison_under_conflicting_sources",
        "termination_judgment_sampling_gap",
    ]:
        if specific_pattern in active_failure_pattern_ids:
            active_failure_pattern_ids.add(specific_pattern)

    if "repeated_conflict_detection_gap" in active_failure_pattern_ids:
        proposals.append(
            {
                "title": "Stress repeated conflict detection under contradictory authority",
                "problem": "高信息恢复采样连续指向 conflict_detection 弱点，说明该簇已足够具体，不能继续只按总括 blind spot 处理。",
                "evidence": ["failure_pattern=repeated_conflict_detection_gap", f"stagnation_rounds={stagnation_rounds}"],
                "from_failure_pattern": "repeated_conflict_detection_gap",
                "from_top_gap": "new_failure_pattern_stalled",
                "change_type": "evaluation_or_dataset",
                "proposed_change": "围绕 contradictory_authority / missing_evidence 两类 mutation 新增定向 prompts，并把 conflict_detection 作为专项压测包维护。",
                "expected_metric_delta": "要么暴露新的 conflict_detection mismatch cluster，要么证明该专项簇在更强压力下仍稳定通过。",
                "risk_level": "low",
                "evaluation_plan": [
                    "python -m scripts.build_real_prompt_candidates",
                    "python -m scripts.evaluate_real_prompts",
                ],
                "rollback_plan": "若新增专项 prompts 只带来噪声，则回退 conflict_detection 专项样本包。",
                "relative_to_last_round": "从总括 blind spot 施压升级为 conflict_detection 专项压测。",
                "scenario_fit": "多来源冲突、权威冲突、证据缺失下的搜索与判断场景。",
                "needs_human_approval": "evaluation_or_dataset" in gated_types,
                "architect_handoff": "researcher + benchmark_owner design conflict_detection stress pack",
            }
        )

    if "comparison_under_conflicting_sources" in active_failure_pattern_ids:
        proposals.append(
            {
                "title": "Probe comparison under conflicting sources with source-ranking prompts",
                "problem": "comparison 簇已被高信息恢复采样具体化，当前需要从总括 blind spot 升级为 source-ranking 专项验证。",
                "evidence": ["failure_pattern=comparison_under_conflicting_sources", f"stagnation_rounds={stagnation_rounds}"],
                "from_failure_pattern": "comparison_under_conflicting_sources",
                "from_top_gap": "new_failure_pattern_stalled",
                "change_type": "evaluation_or_dataset",
                "proposed_change": "新增 conflicting_sources 场景下的 comparison prompts，要求显式比较来源可信度与冲突证据。",
                "expected_metric_delta": "要么暴露 comparison 逻辑下的新 mismatch，要么确认该专项簇可稳定处理来源冲突。",
                "risk_level": "low",
                "evaluation_plan": [
                    "python -m scripts.build_real_prompt_candidates",
                    "python -m scripts.evaluate_real_prompts",
                ],
                "rollback_plan": "若专项 comparison prompts 无法提供增量信息，则回退该专项样本。",
                "relative_to_last_round": "从总括 blind spot 施压升级为 comparison/source-conflict 专项压测。",
                "scenario_fit": "多来源对比、来源冲突、证据权重判断场景。",
                "needs_human_approval": "evaluation_or_dataset" in gated_types,
                "architect_handoff": "benchmark_owner + researcher design conflicting_sources comparison pack",
            }
        )

    if "tool_boundary_sampling_gap" in active_failure_pattern_ids:
        proposals.append(
            {
                "title": "Tighten tool boundary prompts for calculator vs search decisions",
                "problem": "tool_selection 的边界弱点已具体化，当前需要把 calculator_vs_search 场景升级为专项验证对象。",
                "evidence": ["failure_pattern=tool_boundary_sampling_gap", f"stagnation_rounds={stagnation_rounds}"],
                "from_failure_pattern": "tool_boundary_sampling_gap",
                "from_top_gap": "new_failure_pattern_stalled",
                "change_type": "evaluation_or_dataset",
                "proposed_change": "新增 calculator_vs_search 的边界 prompts，要求显式区分数值计算与信息检索型任务。",
                "expected_metric_delta": "要么暴露工具边界误判的新弱点，要么确认该边界专项场景已稳定。",
                "risk_level": "low",
                "evaluation_plan": [
                    "python -m scripts.build_real_prompt_candidates",
                    "python -m scripts.evaluate_real_prompts",
                ],
                "rollback_plan": "若边界 prompts 无法提供有效区分，则回退该专项样本。",
                "relative_to_last_round": "从总括 blind spot 施压升级为 tool boundary 专项压测。",
                "scenario_fit": "需要在 calculator / search 之间做明确边界判断的真实任务。",
                "needs_human_approval": "evaluation_or_dataset" in gated_types,
                "architect_handoff": "researcher + trainer design calculator_vs_search boundary pack",
            }
        )

    if "sampling_blind_spot" in active_failure_pattern_ids:
        proposals.append(
            {
                "title": "Exploit sampling blind spot with high-information real prompts",
                "problem": "当前样本表面全绿，但仍不足以证明隐藏弱点已被发现。",
                "evidence": ["failure_pattern=sampling_blind_spot", f"stagnation_rounds={stagnation_rounds}"],
                "from_failure_pattern": "sampling_blind_spot",
                "from_top_gap": "research_quality_degraded",
                "change_type": "evaluation_or_dataset",
                "proposed_change": "补充 >=4 条高信息量 real prompts，优先覆盖 conflict / termination / multi-source integration 边界场景。",
                "expected_metric_delta": "新增样本后要么暴露新 weakness cluster，要么证明 match_rate 仍稳定 >=0.90",
                "risk_level": "low",
                "evaluation_plan": [
                    "python -m scripts.build_real_prompt_candidates",
                    "python -m scripts.evaluate_real_prompts",
                ],
                "rollback_plan": "回退新增高信息量样本，恢复上一版评测集。",
                "relative_to_last_round": "从泛化扩容改为针对 sampling blind spot 的定向施压。",
                "scenario_fit": "真实复杂任务里，看似通过但隐藏弱点未被触发的场景。",
                "needs_human_approval": "evaluation_or_dataset" in gated_types,
                "architect_handoff": "benchmark_owner + researcher co-design high-information prompt pack",
            }
        )

    if "frontier_research_blindspot" in active_failure_pattern_ids:
        proposals.append(
            {
                "title": "Reopen frontier research intake with minimum observation quota",
                "problem": "前沿研究观察长期为 0，系统容易在封闭样本内自转。",
                "evidence": ["failure_pattern=frontier_research_blindspot", f"frontier_observation_count={signals.get('frontier_observation_count', 0)}"],
                "from_failure_pattern": "frontier_research_blindspot",
                "from_top_gap": "research_quality_degraded",
                "change_type": "research_tracking",
                "proposed_change": "每轮最少补 3 条前沿观察，并形成 可借鉴 / 不建议 / 待观察 标签。",
                "expected_metric_delta": "frontier_observation_count > 0 且研究结论不再只依赖仓内既有样本",
                "risk_level": "low",
                "evaluation_plan": [
                    "补充 data/research/frontier_observations.jsonl",
                    "更新 backlog/opportunities/research_latest.md 的外部路线判断",
                ],
                "rollback_plan": "仅新增研究记录，不改默认运行时策略。",
                "relative_to_last_round": "从被动汇总改为最低配额的前沿摄取。",
                "scenario_fit": "避免团队长期困在既有评测集与本地记忆中。",
                "needs_human_approval": "research_tracking" in gated_types,
                "architect_handoff": "researcher creates minimum frontier intake pack and arbiter labels borrow/avoid/watch",
            }
        )

    if "no_tool_feedback_loop" in active_failure_pattern_ids:
        proposals.append(
            {
                "title": "Bootstrap bridge feedback capture without changing defaults",
                "problem": "bridge feedback 持续为 0，系统无法得到真实用户纠偏信号。",
                "evidence": ["failure_pattern=no_tool_feedback_loop", f"bridge_feedback={signals.get('bridge_feedback', 0)}"],
                "from_failure_pattern": "no_tool_feedback_loop",
                "from_top_gap": "research_quality_degraded",
                "change_type": "desktop_behavior",
                "proposed_change": "先补反馈采样与整理入口，只记录 useful/misread，不修改桌面默认行为。",
                "expected_metric_delta": "bridge_feedback > 0，且 failure miner 能获得真实纠偏样本",
                "risk_level": "medium",
                "evaluation_plan": [
                    "python -m scripts.desktop_agent_control snapshot",
                    "python -m scripts.desktop_bridge_chat",
                ],
                "rollback_plan": "仅停止采样与整理，不改动默认桌面策略。",
                "relative_to_last_round": "从抽象提桥梁闭环，改为先把反馈入口打通。",
                "scenario_fit": "用户在真实桌面协作里纠偏系统误读/误触发的场景。",
                "needs_human_approval": "desktop_behavior" in gated_types,
                "architect_handoff": "guardian reviews scope, then failure_miner defines feedback capture schema before any rollout",
            }
        )

    real_prompt_eval = signals.get("real_prompt_eval", {})
    if isinstance(real_prompt_eval, dict):
        delta = float(real_prompt_eval.get("delta_tool_match_rate", 0.0) or 0.0)
        if delta < 0:
            proposals.append(
                {
                    "title": "Tighten real prompt regression loop",
                    "problem": "真实 prompt 的工具匹配率出现回退。",
                    "evidence": [f"delta_tool_match_rate={delta}"],
                    "change_type": "evaluation_or_dataset",
                    "proposed_change": "扩充真实回归样本并优先审查近期失败 episode。",
                    "risk_level": "medium",
                    "evaluation_plan": [
                        "python -m scripts.build_real_prompt_candidates",
                        "python -m scripts.evaluate_real_prompts",
                    ],
                    "rollback_plan": "这是评测与数据整理提案，无需运行时回滚。",
                    "needs_human_approval": "evaluation_or_dataset" in gated_types,
                }
            )

    control_state = signals.get("control_state", {})
    if isinstance(control_state, dict) and str(control_state.get("rollout_status", "stable")) == "candidate":
        proposals.append(
            {
                "title": "Review candidate runtime rollout",
                "problem": "当前存在候选运行时控制版本，需要继续观察或人工决策。",
                "evidence": [
                    f"candidate_version={control_state.get('candidate_version', '')}",
                    f"baseline_version={control_state.get('candidate_baseline_version', '')}",
                ],
                "change_type": "runtime_control",
                "proposed_change": "保持候选版本隔离，继续运行控制周期并执行上线闸门判断。",
                "risk_level": "high",
                "evaluation_plan": [
                    "python -m scripts.run_control_cycle",
                    "python -m scripts.judge_control_rollout",
                ],
                "rollback_plan": "python -m scripts.rollback_runtime_controls",
                "needs_human_approval": "runtime_control" in gated_types,
            }
        )

    adaptive_proposals = _build_adaptive_proposals(signals, gated_types, recursive_state or {})
    if adaptive_proposals:
        proposals.extend(adaptive_proposals)

    if not proposals:
        proposals.extend(_build_proactive_proposals(signals, gated_types))

    if recursive_mode == "pivot" and not proposals:
        proposals.insert(
            0,
            {
                "title": "Pivot proposal strategy to scenario-grounded changes",
                "problem": "连续多轮核心信号无变化，提案可能进入机械重复。",
                "evidence": [f"stagnation_rounds={stagnation_rounds}"],
                "change_type": "process_improvement",
                "proposed_change": "下一轮提案强制绑定真实使用场景（工作中断、误触发、高频命令），每条提案必须说明与上一轮差异点。",
                "expected_metric_delta": "重复提案率下降，提案差异化字段完整率提升",
                "risk_level": "low",
                "evaluation_plan": [
                    "在提案模板新增 previous_round_diff 字段",
                    "抽样检查最近 3 条提案是否场景化且不重复",
                ],
                "rollback_plan": "仅流程约束改动，删除新增字段即可回退。",
                "relative_to_last_round": "从固定模板切换到场景化提案。",
                "scenario_fit": "高频真实工作流中的中断/误触发/多任务切换场景",
                "needs_human_approval": "process_improvement" in gated_types,
            },
        )

    proposals = _dedupe_and_prioritize_proposals(proposals)

    if recursive_mode == "max_landing" and not proposals:
        proposals.insert(
            0,
            {
                "title": "Force maximum landing plan after stagnation",
                "problem": "连续多轮无显著变化，需要从探索转为最大可落地。",
                "evidence": [f"stagnation_rounds={stagnation_rounds}"],
                "change_type": "process_improvement",
                "proposed_change": "暂停新增探索项 1 轮，只保留可在 24 小时内验证落地的动作：扩充 real prompts、修复 Top1 failure pattern、补齐训练前后对比。",
                "expected_metric_delta": "24h 内形成至少 1 条已验证落地改动",
                "risk_level": "low",
                "evaluation_plan": [
                    "执行 python -m scripts.evaluate_real_prompts",
                    "更新 memory/failure_patterns.md Top1 修复状态",
                    "补充一份训练前后指标对比",
                ],
                "rollback_plan": "恢复常规提案配额并移除强制落地限制。",
                "relative_to_last_round": "由探索导向切为落地导向。",
                "scenario_fit": "当前版本长时间无可见产出时的应急策略",
                "needs_human_approval": "process_improvement" in gated_types,
            },
        )

    max_new = config.get("heartbeat", {}).get("max_new_proposals_per_cycle", 3) if isinstance(config.get("heartbeat", {}), dict) else 3
    max_new = int(max_new)
    if len(proposals) <= max_new:
        return proposals

    pinned: list[dict[str, object]] = []
    for item in proposals:
        if bool(item.get("needs_human_approval", False)) or str(item.get("change_type", "")) == "runtime_control":
            pinned.append(item)

    ordered: list[dict[str, object]] = []
    seen_titles: set[str] = set()
    for item in pinned + proposals:
        title = str(item.get("title", ""))
        if title in seen_titles:
            continue
        ordered.append(item)
        seen_titles.add(title)

    return ordered[:max_new]


def _primary_proposal_summary(proposal_paths: list[Path] | None) -> dict[str, str]:
    paths = proposal_paths or []
    if not paths:
        return {}
    path = paths[0]
    summary = _load_proposal_summary(path)
    title = path.stem.rsplit("_", 1)[0].replace("_", " ").strip()
    summary.setdefault("title", title)
    return summary


def _run_research_recovery_operators(
    root: Path,
    signals: dict[str, object],
    research_quality: dict[str, object],
    config: dict[str, object],
) -> tuple[Path, dict[str, object]]:
    active_patterns = _load_active_failure_pattern_ids(root)
    operators = _select_research_recovery_operators(research_quality, active_patterns, config)
    state = _load_research_recovery_state(root)
    results: list[dict[str, object]] = []
    for operator in operators:
        if operator == "frontier_intake_operator":
            result = _run_frontier_intake_operator(root, config, signals)
            state["frontier_intake_runs"] = int(state.get("frontier_intake_runs", 0) or 0) + 1
        elif operator == "soft_feedback_operator":
            result = _run_soft_feedback_operator(root, signals, config)
            state["soft_feedback_runs"] = int(state.get("soft_feedback_runs", 0) or 0) + 1
        elif operator == "high_information_sampling_operator":
            result = _run_high_information_sampling_operator(root, signals, config)
            state["high_information_sampling_runs"] = int(state.get("high_information_sampling_runs", 0) or 0) + 1
        else:
            continue
        results.append(result)
    state["last_triggered"] = operators
    state["updated_at_utc"] = utc_now().isoformat()
    state_path = root / "data" / "team" / "research_recovery_state.json"
    state_path.parent.mkdir(parents=True, exist_ok=True)
    _write_if_changed(state_path, json.dumps(state, ensure_ascii=False, indent=2) + "\n")

    report_path = root / "backlog" / "opportunities" / "research_recovery_report.md"
    lines = [
        "# Research Recovery Report",
        "",
        f"- triggered: {bool(operators)}",
        f"- operators: {', '.join(operators) or 'none'}",
    ]
    for item in results:
        lines.append(f"- {item.get('operator', '')}: generated_count={item.get('generated_count', 0)} path={item.get('path', '')}")
        if item.get("operator") == "high_information_sampling_operator":
            variant_path = Path(str(item.get("path", "")))
            payload = _read_json(variant_path) if variant_path.exists() else {}
            variants = payload.get("variants", []) if isinstance(payload, dict) else []
            if isinstance(variants, list) and variants:
                lines.extend([
                    "",
                    "## High-information sampling details",
                    f"- avg_source_score: {round(sum(int(v.get('source_score', 0) or 0) for v in variants) / max(1, len(variants)), 2)}",
                    f"- top_mutations: {', '.join(str(v.get('mutation', '')) for v in variants[:3])}",
                    "- top_parents:",
                ])
                for v in variants[:3]:
                    lines.append(
                        f"  - {v.get('parent_case','')} | dataset={v.get('source_dataset','')} | rank={v.get('source_rank',0)} | score={v.get('source_score',0)} | mutation={v.get('mutation','')} | selection={v.get('selection_reason','')}"
                    )
    _write_if_changed(report_path, "\n".join(lines) + "\n")

    return report_path, {
        "triggered": bool(operators),
        "triggered_operators": operators,
        "results": results,
        "state_path": str(state_path),
        "report_path": str(report_path),
    }


def write_daily_digest(root: Path, digest: dict[str, object]) -> Path:
    stamp = datetime.fromisoformat(str(digest["timestamp_utc"]))
    output_path = root / "memory" / "daily" / f"{stamp.strftime('%Y-%m-%d')}.md"
    direction_review = digest.get("direction_review", {})
    if not isinstance(direction_review, dict):
        direction_review = {}

    lines = [
        f"# {stamp.strftime('%Y-%m-%d')} Daily Digest",
        "",
        f"- team: {digest.get('team_name', 'mustard-claw')}",
        f"- direction_verdict: {direction_review.get('verdict', 'direction_correct')}",
        f"- direction_owner: {direction_review.get('owner', 'arbiter')}",
        f"- alerts: {', '.join(digest.get('alerts', [])) or 'none'}",
    ]

    recursive_state = digest.get("recursive_state", {})
    if isinstance(recursive_state, dict) and recursive_state:
        lines.extend(
            [
                f"- recursive_mode: {recursive_state.get('last_mode', 'normal')}",
                f"- stagnation_rounds: {recursive_state.get('stagnation_rounds', 0)}",
            ]
        )

    signals = digest.get("signals", {})
    if isinstance(signals, dict):
        lines.extend(
            [
                f"- episodes: {signals.get('episodes', 0)}",
                f"- reviews: {signals.get('reviews', 0)}",
                f"- bridge_feedback: {signals.get('bridge_feedback', 0)}",
                f"- latest_train_run_id: {signals.get('latest_train_run_id', '')}",
            ]
        )

    research_quality = digest.get("research_quality", {})
    if isinstance(research_quality, dict) and research_quality:
        lines.extend(
            [
                f"- research_quality: {'degraded' if research_quality.get('degraded') else 'healthy'}",
                f"- research_reasons: {', '.join(research_quality.get('reasons', [])) or 'none'}",
                f"- new_failure_pattern_count: {research_quality.get('new_failure_pattern_count', 0)}",
                f"- researcher_value_gate: {'fail' if research_quality.get('degraded') else 'pass'}",
                f"- failure_miner_actionable: {'yes' if int(research_quality.get('failure_pattern_count', 0) or 0) > 0 else 'no'}",
            ]
        )
        if research_quality.get('degraded'):
            lines.append("- repair_action: redefine researcher output + improve failure mining inputs")

    primary_proposal = digest.get("primary_proposal", {})
    if isinstance(primary_proposal, dict) and primary_proposal:
        lines.extend(
            [
                f"- primary_proposal_title: {primary_proposal.get('title', '')}",
                f"- primary_from_failure_pattern: {primary_proposal.get('from_failure_pattern', '')}",
                f"- primary_from_top_gap: {primary_proposal.get('from_top_gap', '')}",
                f"- primary_architect_handoff: {primary_proposal.get('architect_handoff', '')}",
            ]
        )

    research_recovery = digest.get("research_recovery", {})
    if isinstance(research_recovery, dict) and research_recovery:
        lines.extend(
            [
                f"- research_recovery_triggered: {bool(research_recovery.get('triggered'))}",
                f"- research_recovery_operators: {', '.join(research_recovery.get('triggered_operators', [])) or 'none'}",
            ]
        )

    team_actions = digest.get("team_actions", {})
    if isinstance(team_actions, dict) and team_actions:
        lines.append("- team_actions:")
        for role, action in team_actions.items():
            lines.append(f"  - {role}: {action}")

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output_path


def write_proposals(root: Path, proposals: list[dict[str, object]]) -> list[Path]:
    written: list[Path] = []
    for proposal in proposals:
        title = str(proposal.get("title", "proposal"))
        fingerprint = hashlib.sha256(
            json.dumps(
                {
                    "title": title,
                    "problem": proposal.get("problem", ""),
                    "evidence": proposal.get("evidence", []),
                    "proposed_change": proposal.get("proposed_change", ""),
                },
                ensure_ascii=False,
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest()[:8]
        filename = f"{_slug(title)}_{fingerprint}.md"
        path = root / "backlog" / "proposals" / filename
        lines = [
            f"# {title}",
            "",
            f"- problem: {proposal.get('problem', '')}",
            f"- from_failure_pattern: {proposal.get('from_failure_pattern', '')}",
            f"- from_top_gap: {proposal.get('from_top_gap', '')}",
            f"- change_type: {proposal.get('change_type', '')}",
            f"- proposed_change: {proposal.get('proposed_change', '')}",
            f"- expected_metric_delta: {proposal.get('expected_metric_delta', '')}",
            f"- risk_level: {proposal.get('risk_level', '')}",
            f"- needs_human_approval: {proposal.get('needs_human_approval', False)}",
            f"- relative_to_last_round: {proposal.get('relative_to_last_round', '')}",
            f"- scenario_fit: {proposal.get('scenario_fit', '')}",
            f"- architect_handoff: {proposal.get('architect_handoff', 'direct_execute_if_format_passes')}",
            f"- rollback_plan: {proposal.get('rollback_plan', '')}",
        ]
        evidence = proposal.get("evidence", [])
        if isinstance(evidence, list) and evidence:
            lines.append("- evidence:")
            lines.extend(f"  - {item}" for item in evidence)
        evaluation_plan = proposal.get("evaluation_plan", [])
        if isinstance(evaluation_plan, list) and evaluation_plan:
            lines.append("- evaluation_plan:")
            lines.extend(f"  - {item}" for item in evaluation_plan)
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        written.append(path)
    return written


def _build_team_actions_summary(
    signals: dict[str, object],
    proposals: list[dict[str, object]],
    direction_review: dict[str, object],
    recursive_state: dict[str, object] | None = None,
    researcher_artifact_path: str = "",
    primary_proposal: dict[str, str] | None = None,
) -> dict[str, str]:
    needs_human = sum(1 for proposal in proposals if bool(proposal.get("needs_human_approval", False)))
    primary = primary_proposal or {}
    proposal_brief = (
        f"{primary.get('title', '')}: {primary.get('proposed_change', '')}"
        if primary.get('title')
        else "; ".join(
            f"{proposal.get('title', '')}: {proposal.get('proposed_change', '')}" for proposal in proposals[:3]
        ) or "本轮无新增提案"
    )

    recursive_mode = "normal"
    stagnation_rounds = 0
    if isinstance(recursive_state, dict):
        recursive_mode = str(recursive_state.get("last_mode", "normal"))
        stagnation_rounds = int(recursive_state.get("stagnation_rounds", 0) or 0)

    frontier_obs = int(signals.get('frontier_observation_count', 0) or 0)
    if researcher_artifact_path:
        researcher_action = f"已提交研究产物: {researcher_artifact_path}"
    else:
        researcher_action = (
            "研究产出不足：本轮先补齐 frontier_observations>=3（可用 web_search/web_fetch），再提交 Value Gate 研究包"
            if frontier_obs < 3
            else "研究产出不足：未生成 research artifact，请检查 researcher_run"
        )

    return {
        "conductor": f"输出提案清单(mode={recursive_mode}, stagnation_rounds={stagnation_rounds}) -> {proposal_brief}",
        "observer": (
            "采集关键信号 "
            f"episodes={signals.get('episodes', 0)} reviews={signals.get('reviews', 0)} "
            f"bridge_feedback={signals.get('bridge_feedback', 0)} frontier_obs={signals.get('frontier_observation_count', 0)}"
        ),
        "failure_miner": "从 episodes/reviews/real_prompt_eval 挖掘失败模式并输出 failure_patterns",
        "benchmark_owner": "维护北极星指标与top_gap（逻辑推理、工具调用、多步成功率、延迟）",
        "architect": f"将问题转为可执行提案（首条：{primary.get('title', str(proposals[0].get('title', '无')) if proposals else '无')}）",
        "trainer": "执行训练/蒸馏流水线并产出可对比训练报告",
        "evaluator": "执行验证链路：unittest + evaluate_pretraining + evaluate_real_prompts + run_control_cycle",
        "guardian": f"风险审查完成，需 Human Gate 的提案={needs_human}",
        "arbiter": f"方向裁决={direction_review.get('verdict', 'direction_correct')} reasons={','.join(direction_review.get('reasons', [])) or 'none'}",
        "researcher": researcher_action + "（模板: team/RESEARCHER_OUTPUT_TEMPLATE.md）",
    }


def _write_failure_patterns(root: Path, signals: dict[str, object]) -> Path:
    incidents_path = root / "backlog" / "incidents" / "auto_failure_patterns.json"
    memory_path = root / "memory" / "failure_patterns.md"

    patterns: list[dict[str, object]] = []
    real_prompt_eval = signals.get("real_prompt_eval", {})
    prompt_count = 0
    mismatch_count = 0
    if isinstance(real_prompt_eval, dict):
        summary = real_prompt_eval.get("summary", {})
        if isinstance(summary, dict):
            prompt_count = int(summary.get("prompt_count", 0) or 0)
            if prompt_count < 20:
                patterns.append(
                    {
                        "id": "eval_coverage_too_low",
                        "severity": "high",
                        "evidence": {"prompt_count": prompt_count, "target_min": 20},
                        "impact": "真实场景覆盖不足，回归结果不稳定。",
                        "frequency": "current_round",
                        "repro_hint": "运行 python -m scripts.evaluate_real_prompts 并检查 summary.prompt_count",
                        "owner_role": "benchmark_owner",
                    }
                )
        rows = real_prompt_eval.get("rows", [])
        if isinstance(rows, list):
            mismatch_count = len([r for r in rows if isinstance(r, dict) and not bool(r.get("pretrained_match", True))])

    frontier_observation_count = int(signals.get("frontier_observation_count", 0) or 0)
    if frontier_observation_count < 2:
        patterns.append(
            {
                "id": "frontier_research_blindspot",
                "severity": "high",
                "evidence": {"frontier_observation_count": frontier_observation_count, "target_min": 2},
                "impact": "缺少前沿对标，容易重复走弯路。",
                "frequency": "persistent",
                "repro_hint": "检查 data/research/frontier_observations.jsonl 是否连续多轮为空或低于最小值",
                "owner_role": "researcher",
            }
        )

    bridge_feedback = int(signals.get("bridge_feedback", 0) or 0)
    if bridge_feedback == 0:
        patterns.append(
            {
                "id": "no_tool_feedback_loop",
                "severity": "medium",
                "evidence": {"bridge_feedback": bridge_feedback},
                "impact": "缺少用户纠偏信号，工具调用场景改进缓慢。",
                "frequency": "persistent",
                "repro_hint": "检查 bridge feedback 采样是否持续为 0",
                "owner_role": "failure_miner",
            }
        )

    if prompt_count >= 20 and mismatch_count == 0:
        patterns.append(
            {
                "id": "sampling_blind_spot",
                "severity": "medium",
                "evidence": {"prompt_count": prompt_count, "mismatch_count": mismatch_count},
                "impact": "样本表面全绿，但仍不足以证明隐藏弱点已被发现。",
                "frequency": "current_round",
                "repro_hint": "扩充高信息量 real prompts 后重跑 evaluate_real_prompts，看是否暴露新弱点簇",
                "owner_role": "researcher",
            }
        )

    recovery_variants_payload = _read_json(root / "data" / "evolution" / "research_recovery_variants.json")
    recovery_clusters = recovery_variants_payload.get("clusters", []) if isinstance(recovery_variants_payload, dict) else []
    if isinstance(recovery_clusters, list):
        for cluster in recovery_clusters:
            if not isinstance(cluster, dict):
                continue
            pattern_id = str(cluster.get("pattern_id", "")).strip()
            if not pattern_id:
                continue
            if int(cluster.get("count", 0) or 0) < 1:
                continue
            patterns.append(
                {
                    "id": pattern_id,
                    "severity": "medium",
                    "evidence": {
                        "logic_skill": cluster.get("logic_skill", ""),
                        "count": cluster.get("count", 0),
                        "avg_source_score": cluster.get("avg_source_score", 0.0),
                        "top_mutations": cluster.get("top_mutations", []),
                    },
                    "impact": "高信息恢复采样持续指向同一类薄弱模式，说明 blind spot 已开始具体化。",
                    "frequency": "current_round",
                    "repro_hint": "复用 recovery variants / focus eval 继续压测同类 logic skill 与 mutation",
                    "owner_role": "failure_miner",
                }
            )

    payload = {
        "pattern_count": len(patterns),
        "patterns": patterns,
    }
    _write_if_changed(incidents_path, json.dumps(payload, ensure_ascii=False, indent=2) + "\n")

    top_lines = [
        "# Failure Patterns",
        "",
        "记录稳定复现的失败模式、误判模式、采样盲区和回滚原因。",
        "",
        "## Top patterns this round",
        "",
    ]
    for item in patterns[:5]:
        top_lines.extend([
            f"### {item.get('id', 'unknown')}",
            f"- pattern_id: {item.get('id', 'unknown')}",
            f"- title: {item.get('id', 'unknown')}",
            f"- frequency: {item.get('frequency', 'current_round')}",
            f"- impact: {item.get('impact', '')}",
            f"- repro_hint: {item.get('repro_hint', '')}",
            f"- owner_role: {item.get('owner_role', 'unknown')}",
            f"- representative_cases: {json.dumps(item.get('evidence', {}), ensure_ascii=False)}",
            "- likely_root_cause: current research loop still undersamples high-information weaknesses or lacks external feedback pressure",
            "- recommended_fix_direction: tighten sampling + convert meta-failure into explicit repair tasks",
            "- status: open",
            "",
        ])

    top_lines.extend([
        "## Sampling insufficiency / blind spots",
        "",
        f"- blind_spot: high-information real prompts still underrepresented even when aggregate match stays high",
        f"- why_current_sample_is_insufficient: prompt_count={prompt_count}, mismatch_count={mismatch_count}, bridge_feedback={bridge_feedback}, frontier_observation_count={frontier_observation_count}",
        "- next_sampling_action: add >=4 high-information prompts and collect non-zero frontier / bridge evidence",
        "- owner_role: researcher",
        "",
        "## Research degradation patterns",
        "",
        "- repeated_low_information_candidates",
        "- no_new_failure_pattern_across_rounds",
        "- coverage_only_top_gap_repetition",
        "- frontier_zero_signal_persistence",
        "- bridge_zero_feedback_persistence",
        "- report_style_research_without_diagnostic_novelty",
        "",
        "## Repair leads for Architect",
        "",
        "- 将 sampling_blind_spot / frontier_research_blindspot / no_tool_feedback_loop 直接转成下一轮 Architect 提案输入。",
    ])
    _write_if_changed(memory_path, "\n".join(top_lines) + "\n")
    return incidents_path


def _write_research_brief(root: Path, signals: dict[str, object], config: dict[str, object]) -> Path:
    brief_path = root / "backlog" / "opportunities" / "research_brief.md"

    rp_count = 0
    rp_match = 0.0
    real_prompt_eval = signals.get("real_prompt_eval", {})
    if isinstance(real_prompt_eval, dict):
        summary = real_prompt_eval.get("summary", {})
        if isinstance(summary, dict):
            rp_count = int(summary.get("prompt_count", 0) or 0)
            rp_match = float(summary.get("pretrained_match_rate", 0.0) or 0.0)

    researcher_policy = config.get("researcher_policy", {})
    if not isinstance(researcher_policy, dict):
        researcher_policy = {}

    lines = [
        "# Research Brief",
        "",
        "## Current Signals",
        f"- dataset_sample_count: {int(signals.get('dataset_sample_count', 0) or 0)}",
        f"- bridge_feedback: {int(signals.get('bridge_feedback', 0) or 0)}",
        f"- frontier_observation_count: {int(signals.get('frontier_observation_count', 0) or 0)}",
        f"- real_prompt_count: {rp_count}",
        f"- real_prompt_match_rate: {rp_match:.4f}",
        "",
        "## Research Constraints",
        "- 必须绑定 Top Gap 或 failure pattern",
        "- 必须给出可证伪实验",
        "- 必须说明 relative_to_last_round 与 scenario_fit",
        "",
        "## Policy",
        f"- min_frontier_observations: {int(researcher_policy.get('min_frontier_observations', 3) or 3)}",
        f"- allow_external_search: {bool(researcher_policy.get('allow_external_search', True))}",
        f"- allow_external_fetch: {bool(researcher_policy.get('allow_external_fetch', True))}",
        "",
        "## Required Output",
        "- 按 team/RESEARCHER_OUTPUT_TEMPLATE.md 提交",
    ]

    _write_if_changed(brief_path, "\n".join(lines) + "\n")
    return brief_path


def _write_carm_gap_map(root: Path, signals: dict[str, object]) -> Path:
    path = root / "backlog" / "opportunities" / "carm_gap_map.md"

    rp_count = 0
    bridge_feedback = int(signals.get("bridge_feedback", 0) or 0)
    frontier_obs = int(signals.get("frontier_observation_count", 0) or 0)
    real_prompt_eval = signals.get("real_prompt_eval", {})
    if isinstance(real_prompt_eval, dict):
        summary = real_prompt_eval.get("summary", {})
        if isinstance(summary, dict):
            rp_count = int(summary.get("prompt_count", 0) or 0)

    lines = [
        "# CARM Gap Map",
        "",
        "## Top Gaps",
        "",
        "### Gap 1: 真实场景覆盖不足",
        f"- current: real_prompt_count={rp_count}",
        "- target: >=20",
        "- blocker: 候选样本构建与筛选未形成稳定流水线",
        "- owner: CARM Owner + Benchmark Owner",
        "- next_action: build_real_prompt_candidates + evaluate_real_prompts",
        "- acceptance: prompt_count>=20 且报告可复现",
        "",
        "### Gap 2: 用户纠偏闭环缺失",
        f"- current: bridge_feedback={bridge_feedback}",
        "- target: >=30 条高价值反馈",
        "- blocker: bridge 反馈采样与标注流程缺位",
        "- owner: CARM Owner + Failure Miner",
        "- next_action: 启动 bridge 反馈采集并结构化标注",
        "- acceptance: 形成反馈样本包并进入评测",
        "",
        "### Gap 3: 前沿对标不足",
        f"- current: frontier_observation_count={frontier_obs}",
        "- target: >=10 条可比较观察",
        "- blocker: 外部路线跟踪未形成固定节奏",
        "- owner: Arbiter(CARM Track) + Researcher",
        "- next_action: 每轮补充 3 条前沿观察并打标签",
        "- acceptance: 形成可借鉴/不建议/待观察三类结论",
        "",
        "## CARM MVI（本轮）",
        "- carm_mvi: 先完成 Gap1 的样本覆盖增量（6 -> 20）并输出前后对比",
        "- window: 24-72h",
    ]

    _write_if_changed(path, "\n".join(lines) + "\n")
    return path


def _select_top_gap(signals: dict[str, object]) -> dict[str, object]:
    real_prompt_eval = signals.get("real_prompt_eval", {})
    prompt_count = 0
    if isinstance(real_prompt_eval, dict):
        summary = real_prompt_eval.get("summary", {})
        if isinstance(summary, dict):
            prompt_count = int(summary.get("prompt_count", 0) or 0)

    research_quality = signals.get("research_quality", {})
    if not isinstance(research_quality, dict):
        research_quality = {}
    recursive_state = signals.get("recursive_state", {})
    if not isinstance(recursive_state, dict):
        recursive_state = {}

    stagnation_rounds = int(recursive_state.get("stagnation_rounds", 0) or 0)
    new_failure_pattern_count = int(research_quality.get("new_failure_pattern_count", 0) or 0)
    high_signal_count = int(research_quality.get("high_signal_count", 0) or 0)
    quality_exploration_active = bool(research_quality.get("quality_exploration_active", False))
    blind_spot_persistence_rounds = int(research_quality.get("blind_spot_persistence_rounds", 0) or 0)
    zero_bridge_feedback_streak = int(research_quality.get("zero_bridge_feedback_streak", 0) or 0)
    zero_frontier_observation_streak = int(research_quality.get("zero_frontier_observation_streak", 0) or 0)

    researcher_text = str(signals.get("researcher_artifact_text", "") or "").lower()
    blind_spot_active = (
        "blind spot remains" in researcher_text
        or "sampling blind spot" in researcher_text
        or blind_spot_persistence_rounds > 0
    )

    if prompt_count < 20:
        gap = max(0, 20 - prompt_count)
        return {
            "gap_id": "eval_coverage_too_low",
            "problem": "真实工具调用场景回归样本覆盖不足，当前无法证明可替代性提升。",
            "current": f"prompt_count={prompt_count}",
            "target": "prompt_count>=20（优先本地工具调用场景）",
            "gap": gap,
            "owner": "benchmark_owner + failure_miner + trainer",
            "why_this_is_top_gap_now": "当前真实 prompt 覆盖数未达最低目标，基础验证样本仍不足。",
            "action_plan": [
                "运行 python -m scripts.build_real_prompt_candidates 生成候选集",
                "合并候选集到 configs/real_prompt_eval.json（去重后保留高价值工具调用样本）",
                "运行 python -m scripts.evaluate_real_prompts 并记录前后对比",
            ],
            "acceptance": [
                "real_prompt_eval.summary.prompt_count >= 20",
                "产出一份前后指标对比摘要（match_rate / avg_steps）",
            ],
            "rollback": "若样本质量下降，回退 real_prompt_eval.json 到上一版并重新评测",
        }

    if new_failure_pattern_count == 0 and stagnation_rounds >= 10:
        return {
            "gap_id": "new_failure_pattern_stalled",
            "problem": "长期停滞但没有新增 failure pattern，说明新增弱点发现环节失灵。",
            "current": f"stagnation_rounds={stagnation_rounds}; new_failure_pattern_count={new_failure_pattern_count}",
            "target": "在后续 1~2 个周期内形成至少 1 个新增 failure pattern 或新弱点簇",
            "gap": max(1, stagnation_rounds),
            "owner": "researcher + failure_miner + arbiter",
            "why_this_is_top_gap_now": "覆盖数已达标，但系统仍未形成新增发现，当前瓶颈已从 coverage 转向 discovery。",
            "action_plan": [
                "围绕高信号样本簇生成更具区分度的压测 prompt",
                "优先验证 comparison / conflict_detection / tool_boundary 的新弱点簇",
                "要求 researcher 明确给出可证伪的新 failure hypothesis",
            ],
            "acceptance": [
                "出现至少 1 个新增 failure pattern 或新 recovery cluster",
                "research_quality 不再出现 no_new_failure_pattern",
            ],
            "rollback": "若新增样本只制造噪声，则回退本轮高压样本并保留有效簇",
        }

    if blind_spot_active and quality_exploration_active and high_signal_count > 0:
        return {
            "gap_id": "blind_spot_not_broken",
            "problem": "研究已识别 blind spot，但仍未证明该盲区被真正打穿。",
            "current": f"high_signal_count={high_signal_count}; blind_spot_persistence_rounds={blind_spot_persistence_rounds}",
            "target": "通过更强压力样本证明 blind spot 已消除，或显式暴露出新的 mismatch cluster",
            "gap": max(1, blind_spot_persistence_rounds or high_signal_count),
            "owner": "researcher + benchmark_owner + architect",
            "why_this_is_top_gap_now": "当前主要问题不是样本数量，而是高信息盲区迟迟没有被打穿。",
            "action_plan": [
                "围绕高信号行构建更尖锐的 follow-up prompts",
                "减少低信息重复样本，提升单条样本的揭弱能力",
                "要求 architect 输出直接针对 blind spot 的最小落地变更",
            ],
            "acceptance": [
                "blind_spot_persistence_rounds 下降或归零",
                "新一轮研究产物不再声明 blind spot remains",
            ],
            "rollback": "若新样本无法提供更高信息增量，则回退到上一轮高信号集合",
        }

    if zero_bridge_feedback_streak > 0:
        return {
            "gap_id": "bridge_feedback_missing",
            "problem": "缺少 bridge feedback，系统无法从真实使用偏差中获得反馈。",
            "current": f"zero_bridge_feedback_streak={zero_bridge_feedback_streak}",
            "target": "恢复 bridge feedback 并形成可用于 failure miner 的信号流",
            "gap": zero_bridge_feedback_streak,
            "owner": "observer + failure_miner",
            "why_this_is_top_gap_now": "缺少真实反馈会导致团队只围绕离线样本自循环。",
            "action_plan": [
                "检查 bridge feedback 采集入口是否仍有效",
                "补齐最小可用反馈样本并验证落盘",
            ],
            "acceptance": [
                "bridge_feedback > 0",
            ],
            "rollback": "若反馈入口异常，先暂停依赖该信号的推断",
        }

    if zero_frontier_observation_streak > 0:
        return {
            "gap_id": "frontier_signal_missing",
            "problem": "缺少 frontier observation，团队容易在旧问题上内循环。",
            "current": f"zero_frontier_observation_streak={zero_frontier_observation_streak}",
            "target": "恢复最小 frontier observation 流并形成借鉴/规避判断",
            "gap": zero_frontier_observation_streak,
            "owner": "researcher",
            "why_this_is_top_gap_now": "没有外部前沿信号，研究回路更容易退化成重复解释。",
            "action_plan": [
                "补充至少 3 条 frontier observation",
                "为每条 observation 给出 borrow/avoid/watch 标签",
            ],
            "acceptance": [
                "frontier_observation_count >= 3",
            ],
            "rollback": "若前沿输入噪声过大，则回退本轮 observation 集",
        }

    return {
        "gap_id": "quality_stabilization",
        "problem": "当前主要任务是维持已识别高信号质量差异，并防止退化。",
        "current": f"high_signal_count={high_signal_count}",
        "target": "保持高信号样本稳定可复现，并避免质量回退",
        "gap": high_signal_count,
        "owner": "benchmark_owner + evaluator",
        "why_this_is_top_gap_now": "coverage 已达标，且未检测到更高优先级缺口，当前以质量维稳为主。",
        "action_plan": [
            "复核高信号样本集的稳定性",
            "继续监控 baseline/pretrained 分离度",
        ],
        "acceptance": [
            "高信号样本集持续可复现",
        ],
        "rollback": "若质量维稳样本失真，则回退到上一轮有效集合",
    }


def _write_top_gap_action_card(root: Path, signals: dict[str, object]) -> Path:
    top_gap_path = root / "backlog" / "opportunities" / "auto_top_gap.md"
    card = _select_top_gap(signals)

    lines = [
        "# Top Gap Action Card",
        "",
        f"- gap_id: {card.get('gap_id', '')}",
        f"- problem: {card.get('problem', '')}",
        f"- current: {card.get('current', '')}",
        f"- target: {card.get('target', '')}",
        f"- gap: {card.get('gap', '')}",
        f"- owner: {card.get('owner', '')}",
        f"- why_this_is_top_gap_now: {card.get('why_this_is_top_gap_now', '')}",
        "- action_plan:",
    ]
    for item in card.get("action_plan", []):
        lines.append(f"  - {item}")
    lines.append("- acceptance:")
    for item in card.get("acceptance", []):
        lines.append(f"  - {item}")
    lines.append(f"- rollback: {card.get('rollback', '')}")

    _write_if_changed(top_gap_path, "\n".join(lines) + "\n")
    return top_gap_path


def _load_role_evolution_state(root: Path) -> dict[str, object]:
    path = root / "data" / "team" / "role_evolution_state.json"
    if not path.exists():
        return {"roles": {}, "updated_at_utc": ""}
    payload = _read_json(path)
    if not isinstance(payload, dict):
        return {"roles": {}, "updated_at_utc": ""}
    return payload


def _load_research_quality_state(root: Path) -> dict[str, object]:
    path = root / "data" / "team" / "research_quality_state.json"
    if not path.exists():
        return {
            "rounds_without_new_failure_pattern": 0,
            "coverage_only_gap_streak": 0,
            "zero_bridge_feedback_streak": 0,
            "zero_frontier_observation_streak": 0,
            "updated_at_utc": "",
        }
    payload = _read_json(path)
    if not isinstance(payload, dict):
        return {
            "rounds_without_new_failure_pattern": 0,
            "coverage_only_gap_streak": 0,
            "zero_bridge_feedback_streak": 0,
            "zero_frontier_observation_streak": 0,
            "updated_at_utc": "",
        }
    return payload


def _extract_failure_pattern_ids(path: Path) -> list[str]:
    if not path.exists():
        return []
    text = path.read_text(encoding="utf-8", errors="ignore")
    ids = [item for item in re.findall(r"pattern_id:\s*([A-Za-z0-9_\-]+)", text) if item and item != "-"]
    if ids:
        return ids
    try:
        payload = _read_json(path)
    except json.JSONDecodeError:
        return []
    patterns = payload.get("patterns", []) if isinstance(payload, dict) else []
    if not isinstance(patterns, list):
        return []
    extracted: list[str] = []
    for item in patterns:
        if isinstance(item, dict):
            value = str(item.get("id", "")).strip()
            if value and value != "-":
                extracted.append(value)
    return extracted


def _evaluate_research_quality(
    root: Path,
    signals: dict[str, object],
    config: dict[str, object],
    failure_patterns_path: Path,
    researcher_artifact_path: Path,
) -> tuple[Path, dict[str, object]]:
    policy = config.get("research_quality_policy", {})
    if not isinstance(policy, dict):
        policy = {}

    enabled = bool(policy.get("enabled", True))
    max_without_new = int(policy.get("max_rounds_without_new_failure_pattern", 2) or 2)
    max_coverage_only = int(policy.get("max_rounds_with_coverage_only_top_gap", 2) or 2)
    require_nonempty = bool(policy.get("require_nonempty_failure_patterns", True))
    zero_bridge_limit = int(policy.get("flag_zero_bridge_feedback_rounds", 3) or 3)
    zero_frontier_limit = int(policy.get("flag_zero_frontier_observation_rounds", 3) or 3)

    prev = _load_research_quality_state(root)
    previous_ids = prev.get("last_failure_pattern_ids", [])
    if not isinstance(previous_ids, list):
        previous_ids = []
    current_ids = _extract_failure_pattern_ids(failure_patterns_path)
    new_ids = [item for item in current_ids if item not in previous_ids]

    quality_payload = _build_quality_stabilization_payload(signals)
    high_signal_count = int(quality_payload.get("high_signal_count", 0) or 0)

    recovery_variants_payload = _read_json(root / "data" / "evolution" / "research_recovery_variants.json")
    recovery_clusters = recovery_variants_payload.get("clusters", []) if isinstance(recovery_variants_payload, dict) else []
    if not isinstance(recovery_clusters, list):
        recovery_clusters = []
    active_recovery_clusters = [cluster for cluster in recovery_clusters if isinstance(cluster, dict) and int(cluster.get("count", 0) or 0) > 0]
    quality_exploration_active = high_signal_count > 0 or bool(active_recovery_clusters)

    previous_cluster_ids = prev.get("last_recovery_cluster_ids", [])
    if not isinstance(previous_cluster_ids, list):
        previous_cluster_ids = []
    current_cluster_ids = [
        str(cluster.get("pattern_id", "")).strip()
        for cluster in active_recovery_clusters
        if isinstance(cluster, dict) and str(cluster.get("pattern_id", "")).strip()
    ]
    new_cluster_ids = [cluster_id for cluster_id in current_cluster_ids if cluster_id not in previous_cluster_ids]

    rounds_without_new = int(prev.get("rounds_without_new_failure_pattern", 0) or 0)
    exploration_without_discovery_rounds = int(prev.get("exploration_without_discovery_rounds", 0) or 0)
    blind_spot_persistence_rounds = int(prev.get("blind_spot_persistence_rounds", 0) or 0)

    researcher_text = researcher_artifact_path.read_text(encoding="utf-8", errors="ignore") if researcher_artifact_path.exists() else ""
    researcher_text_lower = researcher_text.lower()
    blind_spot_active = (
        "blind spot remains" in researcher_text_lower
        or "sampling blind spot" in researcher_text_lower
        or "blind_spot_if_no_failure_case" in researcher_text_lower
    )

    has_new_discovery = bool(new_ids) or bool(new_cluster_ids)
    if has_new_discovery:
        rounds_without_new = 0
        exploration_without_discovery_rounds = 0
    else:
        rounds_without_new += 1
        if quality_exploration_active:
            exploration_without_discovery_rounds += 1

    if blind_spot_active and not has_new_discovery:
        blind_spot_persistence_rounds += 1
    else:
        blind_spot_persistence_rounds = 0

    top_gap = ""
    real_prompt_eval = signals.get("real_prompt_eval", {})
    if isinstance(real_prompt_eval, dict):
        summary = real_prompt_eval.get("summary", {})
        if isinstance(summary, dict):
            prompt_count = int(summary.get("prompt_count", 0) or 0)
            if prompt_count < 20:
                top_gap = "eval_coverage_too_low"
    coverage_only_streak = int(prev.get("coverage_only_gap_streak", 0) or 0)
    if top_gap == "eval_coverage_too_low":
        coverage_only_streak += 1
    else:
        coverage_only_streak = 0

    zero_bridge_streak = int(prev.get("zero_bridge_feedback_streak", 0) or 0)
    if int(signals.get("bridge_feedback", 0) or 0) == 0:
        zero_bridge_streak += 1
    else:
        zero_bridge_streak = 0

    zero_frontier_streak = int(prev.get("zero_frontier_observation_streak", 0) or 0)
    if int(signals.get("frontier_observation_count", 0) or 0) == 0:
        zero_frontier_streak += 1
    else:
        zero_frontier_streak = 0

    reasons: list[str] = []
    failure_text = failure_patterns_path.read_text(encoding="utf-8", errors="ignore") if failure_patterns_path.exists() else ""
    if require_nonempty and not current_ids:
        reasons.append("failure_patterns_empty")
    if enabled and rounds_without_new >= max_without_new:
        reasons.append("no_new_failure_pattern")
    if enabled and quality_exploration_active and not has_new_discovery:
        reasons.append("exploration_without_discovery")
    if enabled and blind_spot_active and blind_spot_persistence_rounds >= max_without_new:
        reasons.append("blind_spot_persistence")
    if enabled and coverage_only_streak >= max_coverage_only:
        reasons.append("coverage_only_top_gap_repetition")
    if enabled and zero_bridge_streak >= zero_bridge_limit:
        reasons.append("bridge_zero_feedback_persistence")
    if enabled and zero_frontier_streak >= zero_frontier_limit:
        reasons.append("frontier_zero_signal_persistence")

    if researcher_text and "weakness" not in researcher_text_lower and "盲区" not in researcher_text and "采样不足" not in researcher_text:
        reasons.append("report_style_research_without_diagnostic_novelty")
    if failure_text and "Sampling insufficiency / blind spots" in failure_text and not current_ids:
        reasons.append("sampling_insufficiency_active")

    payload = {
        "degraded": bool(reasons),
        "reasons": reasons,
        "new_failure_pattern_count": len(new_ids),
        "new_failure_pattern_ids": new_ids,
        "failure_pattern_count": len(current_ids),
        "new_recovery_cluster_count": len(new_cluster_ids),
        "new_recovery_cluster_ids": new_cluster_ids,
        "coverage_only_gap_streak": coverage_only_streak,
        "zero_bridge_feedback_streak": zero_bridge_streak,
        "zero_frontier_observation_streak": zero_frontier_streak,
        "rounds_without_new_failure_pattern": rounds_without_new,
        "exploration_without_discovery_rounds": exploration_without_discovery_rounds,
        "blind_spot_persistence_rounds": blind_spot_persistence_rounds,
        "top_gap": top_gap or "unknown",
        "quality_exploration_active": quality_exploration_active,
        "high_signal_count": high_signal_count,
        "recovery_cluster_count": len(active_recovery_clusters),
        "updated_at_utc": utc_now().isoformat(),
        "last_failure_pattern_ids": current_ids,
        "last_recovery_cluster_ids": current_cluster_ids,
    }

    path = root / "data" / "team" / "research_quality_state.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    _write_if_changed(path, json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    return path, payload


def _score_role_outputs(signals: dict[str, object], role_artifacts: dict[str, Path]) -> dict[str, float]:
    scores: dict[str, float] = {}
    for role, path in role_artifacts.items():
        if not path.exists():
            scores[role] = 0.0
            continue
        content = path.read_text(encoding="utf-8", errors="ignore").strip()
        base = 0.4 if content else 0.0
        # simple role-specific signal bonus
        bonus = 0.0
        if role == "observer":
            bonus = 0.3 if int(signals.get("episodes", 0) or 0) > 0 else 0.0
        elif role == "benchmark_owner":
            rp = signals.get("real_prompt_eval", {})
            if isinstance(rp, dict):
                s = rp.get("summary", {})
                if isinstance(s, dict) and int(s.get("prompt_count", 0) or 0) > 0:
                    bonus = 0.3
        elif role == "researcher":
            bonus = 0.3 if int(signals.get("frontier_observation_count", 0) or 0) >= 1 else 0.1
        else:
            bonus = 0.2
        scores[role] = min(1.0, base + bonus)
    return scores


def _update_role_evolution(
    root: Path,
    config: dict[str, object],
    role_scores: dict[str, float],
) -> tuple[Path, dict[str, object]]:
    policy = config.get("role_evolution", {})
    if not isinstance(policy, dict):
        policy = {}

    enabled = bool(policy.get("enabled", True))
    under = float(policy.get("underperform_threshold", 0.45) or 0.45)
    promote = float(policy.get("promote_threshold", 0.8) or 0.8)
    max_under = int(policy.get("max_underperform_rounds", 3) or 3)

    state = _load_role_evolution_state(root)
    roles = state.get("roles", {})
    if not isinstance(roles, dict):
        roles = {}

    suggestions: list[str] = []
    for role, score in role_scores.items():
        item = roles.get(role, {})
        if not isinstance(item, dict):
            item = {}
        under_rounds = int(item.get("underperform_rounds", 0) or 0)

        if enabled:
            if score < under:
                under_rounds += 1
            else:
                under_rounds = 0

            if under_rounds >= max_under:
                suggestions.append(f"demote_or_redefine:{role}")
            elif score >= promote:
                suggestions.append(f"promote_weight:{role}")

        roles[role] = {
            "last_score": score,
            "underperform_rounds": under_rounds,
        }

    next_state = {
        "roles": roles,
        "suggestions": suggestions,
        "updated_at_utc": utc_now().isoformat(),
    }

    path = root / "data" / "team" / "role_evolution_state.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    _write_if_changed(path, json.dumps(next_state, ensure_ascii=False, indent=2) + "\n")
    return path, next_state


def _record_and_validate_role_divergence(
    root: Path,
    researcher_path: Path,
    arbiter_path: Path,
) -> dict[str, object]:
    history_path = root / "data" / "team" / "role_content_history.jsonl"
    history = _load_jsonl(history_path)

    last_researcher_hash = ""
    last_arbiter_hash = ""
    for item in reversed(history):
        if not last_researcher_hash and item.get("role") == "researcher":
            last_researcher_hash = str(item.get("content_hash", ""))
        if not last_arbiter_hash and item.get("role") == "arbiter":
            last_arbiter_hash = str(item.get("content_hash", ""))
        if last_researcher_hash and last_arbiter_hash:
            break

    researcher_hash = _content_hash(researcher_path)
    arbiter_hash = _content_hash(arbiter_path)

    researcher_changed = bool(researcher_hash) and researcher_hash != last_researcher_hash
    arbiter_changed = bool(arbiter_hash) and arbiter_hash != last_arbiter_hash

    now = utc_now().isoformat()
    _append_jsonl(history_path, {
        "timestamp_utc": now,
        "role": "researcher",
        "path": str(researcher_path),
        "content_hash": researcher_hash,
        "changed": researcher_changed,
    })
    _append_jsonl(history_path, {
        "timestamp_utc": now,
        "role": "arbiter",
        "path": str(arbiter_path),
        "content_hash": arbiter_hash,
        "changed": arbiter_changed,
    })

    return {
        "history_path": str(history_path),
        "researcher_changed": researcher_changed,
        "arbiter_changed": arbiter_changed,
    }


def _run_researcher(root: Path, signals: dict[str, object], recursive_state: dict[str, object]) -> Path:
    output_dir = root / "backlog" / "opportunities"
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "research_latest.md"

    real_prompt_eval = signals.get("real_prompt_eval", {})
    rp_summary: dict[str, object] = {}
    rp_rows: list[dict[str, object]] = []
    if isinstance(real_prompt_eval, dict):
        summary = real_prompt_eval.get("summary", {})
        if isinstance(summary, dict):
            rp_summary = summary
        rows = real_prompt_eval.get("rows", [])
        if isinstance(rows, list):
            rp_rows = [r for r in rows if isinstance(r, dict)]

    rp_count = int(rp_summary.get("prompt_count", 0) or 0)
    rp_match = float(rp_summary.get("pretrained_match_rate", 0.0) or 0.0)
    baseline_match = float(rp_summary.get("baseline_match_rate", 0.0) or 0.0)

    frontier_obs = int(signals.get("frontier_observation_count", 0) or 0)
    stagnation_rounds = int(recursive_state.get("stagnation_rounds", 0) or 0)

    mismatches = [r for r in rp_rows if not bool(r.get("pretrained_match", True))]
    mismatch_lines: list[str] = []
    for r in mismatches[:5]:
        mismatch_lines.append(
            f"- {r.get('id','unknown')}: expected={r.get('expected_tool','')} actual={r.get('pretrained_used_tool','')} logic_skill={r.get('logic_skill','')}"
        )

    quality_path = root / "backlog" / "opportunities" / "candidate_quality_report.md"
    quality_excerpt: list[str] = []
    if quality_path.exists():
        for line in quality_path.read_text(encoding="utf-8", errors="ignore").splitlines():
            line = line.strip()
            if line.startswith("- total_candidates") or line.startswith("- filtered_candidates") or line.startswith("- dropped_candidates"):
                quality_excerpt.append(line)

    blind_spot_line = "- blind_spot_if_no_failure_case: current batch is still too narrow to prove no hidden weakness" if not mismatches else "- blind_spot_if_no_failure_case: none"
    weakness_summary = (
        "tool-path mismatch remains the active weakness cluster"
        if mismatches
        else "no explicit mismatch surfaced, but sampling blind spot remains in high-information real prompts"
    )

    lines = [
        "# Research Artifact (Actionable)",
        "",
        "## 1) Meta",
        f"- round_id: auto-{utc_now().strftime('%Y%m%dT%H%M%SZ')}",
        f"- date: {utc_now().date().isoformat()}",
        "- owner: researcher",
        "- from_top_gap: eval_coverage_too_low",
        "- from_failure_pattern: eval_coverage_too_low",
        "- relative_to_last_round: switched from static template to concrete mismatch + quality snapshot + executable next-24h plan",
        "- scenario_fit: real prompt regression coverage and weak-signal research diagnosis",
        "",
        "## 2) New weakness discovered this round",
        f"- weakness_summary: {weakness_summary}",
        f"- weakness_cluster: {'tool_path_mismatch' if mismatches else 'sampling_blind_spot'}",
        "- why_it_matters_now: if the system only reports aggregate wins, it cannot prove it still discovers new weaknesses under wider coverage",
        "- why_previous_rounds_missed_it: prior output emphasized summary metrics over weakness clustering and blind-spot diagnosis",
        "",
        "## 3) Hypothesis（可证伪）",
        "- hypothesis: raising high-information real-prompt coverage will expose either stable robustness or a new mismatch cluster worth patching",
        "- falsifiable_condition: prompt_count increases but mismatch_case_count rises materially or match_rate drops below 0.90",
        f"- expected_gain: keep pretrained_match_rate >= 0.90 while increasing prompt_count beyond {rp_count}",
        "- risk: low-information candidates may still crowd out the prompts most likely to reveal hidden weaknesses",
        "",
        "## 4) Evidence chain",
        f"- representative_case_1: pretrained_match_rate={rp_match:.4f}, baseline_match_rate={baseline_match:.4f}, delta={rp_match-baseline_match:+.4f}",
        f"- representative_case_2: mismatch_case_count={len(mismatches)}",
        f"- representative_case_3: stagnation_rounds={stagnation_rounds}; frontier_observation_count={frontier_obs}",
        f"- evidence_quality_note: current evidence is useful for trend judgment but still weak for discovering unseen weakness clusters because prompt coverage is narrow",
        blind_spot_line,
    ]

    if quality_excerpt:
        lines.append("- candidate_pipeline_snapshot:")
        lines.extend([f"  {x}" for x in quality_excerpt])

    lines.extend([
        "",
        "## Concrete mismatch cases",
    ])
    if mismatch_lines:
        lines.extend(mismatch_lines)
    else:
        lines.append("- none (all current prompts matched for pretrained runner)")

    lines.extend([
        "",
        "## 5) Minimal next experiment（可执行）",
        "- command_1: python -m scripts.evaluate_real_prompts",
        "- command_2: python -m scripts.build_real_prompt_candidates",
        "- metric_threshold: prompt_count increases and pretrained_match_rate stays >= 0.90",
        "- pass_criteria: new prompts add pressure without introducing an unexplained mismatch spike",
        "- fail_criteria: coverage expands but research still produces no new weakness or blind-spot diagnosis",
        "",
        "## 6) Landing Candidate（可直接进 Architect）",
        "- proposed_change: add >=4 high-quality non-observer prompts and tighten candidate filtering around low-information repeats",
        "- change_scope: configs/real_prompt_eval.json + candidate quality rules + research reporting",
        "- rollback_plan: revert added prompts and filtering heuristics if mismatch quality worsens or coverage signal becomes noisier",
        "- handoff_to_architect: yes",
        "",
        "## 7) Decision label",
        "- tag: 待观察",
        "- reason: current match is strong, but the system still has insufficient evidence that its weakness discovery loop is healthy under broader coverage",
    ])

    _write_if_changed(path, "\n".join(lines) + "\n")
    return path

def _load_proposal_summary(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    summary: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if not line.startswith("- "):
            continue
        content = line[2:]
        if ":" not in content:
            continue
        key, value = content.split(":", 1)
        summary[key.strip()] = value.strip()
    return summary


def _write_role_artifacts(
    root: Path,
    signals: dict[str, object],
    proposals: list[dict[str, object]],
    direction_review: dict[str, object],
    recursive_state: dict[str, object],
    proposal_paths: list[Path] | None = None,
) -> dict[str, Path]:
    role_dir = root / "backlog" / "role_outputs"
    role_dir.mkdir(parents=True, exist_ok=True)

    observer_path = role_dir / "observer_latest.md"
    _write_if_changed(
        observer_path,
        "\n".join(
            [
                "# Observer Output",
                "",
                f"- episodes: {int(signals.get('episodes', 0) or 0)}",
                f"- reviews: {int(signals.get('reviews', 0) or 0)}",
                f"- bridge_feedback: {int(signals.get('bridge_feedback', 0) or 0)}",
                f"- frontier_observation_count: {int(signals.get('frontier_observation_count', 0) or 0)}",
            ]
        )
        + "\n",
    )

    benchmark_path = role_dir / "benchmark_owner_latest.md"
    rp_count = 0
    rp_match = 0.0
    real_prompt_eval = signals.get("real_prompt_eval", {})
    if isinstance(real_prompt_eval, dict):
        summary = real_prompt_eval.get("summary", {})
        if isinstance(summary, dict):
            rp_count = int(summary.get("prompt_count", 0) or 0)
            rp_match = float(summary.get("pretrained_match_rate", 0.0) or 0.0)
    top_gap_card = _select_top_gap(signals)
    _write_if_changed(
        benchmark_path,
        "\n".join(
            [
                "# Benchmark Owner Output",
                "",
                f"- top_gap: {top_gap_card.get('gap_id', '')}",
                f"- why_this_is_top_gap_now: {top_gap_card.get('why_this_is_top_gap_now', '')}",
                f"- real_prompt_count: {rp_count}",
                f"- real_prompt_match_rate: {rp_match:.4f}",
                f"- target: {top_gap_card.get('target', '')}",
                f"- gap: {top_gap_card.get('gap', '')}",
            ]
        )
        + "\n",
    )

    evaluator_path = role_dir / "evaluator_latest.md"
    pretrain_eval = signals.get("pretrain_eval", {})
    pretrain_match = 0.0
    if isinstance(pretrain_eval, dict):
        pretrained = pretrain_eval.get("pretrained", {})
        if isinstance(pretrained, dict):
            pretrain_match = float(pretrained.get("tool_match_rate", 0.0) or 0.0)
    _write_if_changed(
        evaluator_path,
        "\n".join(
            [
                "# Evaluator Output",
                "",
                "- checks: evaluate_pretraining + evaluate_real_prompts",
                f"- pretrain_tool_match_rate: {pretrain_match:.4f}",
                f"- real_prompt_match_rate: {rp_match:.4f}",
                "- verdict: soft_pass" if rp_count < 20 else "- verdict: pass",
            ]
        )
        + "\n",
    )

    trainer_path = role_dir / "trainer_latest.md"
    _write_if_changed(
        trainer_path,
        "\n".join(
            [
                "# Trainer Output",
                "",
                f"- latest_train_run_id: {signals.get('latest_train_run_id', '')}",
                f"- dataset_sample_count: {int(signals.get('dataset_sample_count', 0) or 0)}",
                "- next_action: expand real prompt candidates and rerun eval",
            ]
        )
        + "\n",
    )

    architect_path = role_dir / "architect_latest.md"
    proposal_quality_ok = 0
    for p in proposals:
        if p.get("relative_to_last_round") and p.get("scenario_fit") and (p.get("from_failure_pattern") or p.get("from_top_gap")):
            proposal_quality_ok += 1
    architect_lines = [
        "# Architect Output",
        "",
        f"- proposal_count_this_round: {len(proposals)}",
        f"- proposal_format_pass_count: {proposal_quality_ok}",
        "- note: no new proposal" if not proposals else "- note: proposals generated",
    ]
    final_paths = proposal_paths or []
    if final_paths:
        first_summary = _load_proposal_summary(final_paths[0])
        architect_lines.extend(
            [
                f"- first_proposal_title: {final_paths[0].stem}",
                f"- first_from_failure_pattern: {first_summary.get('from_failure_pattern', '')}",
                f"- first_from_top_gap: {first_summary.get('from_top_gap', '')}",
                f"- first_expected_metric_delta: {first_summary.get('expected_metric_delta', '')}",
                f"- first_scenario_fit: {first_summary.get('scenario_fit', '')}",
                f"- first_architect_handoff: {first_summary.get('architect_handoff', 'direct_execute_if_format_passes')}",
            ]
        )
    elif proposals:
        first = proposals[0]
        architect_lines.extend(
            [
                f"- first_proposal_title: {first.get('title', '')}",
                f"- first_from_failure_pattern: {first.get('from_failure_pattern', '')}",
                f"- first_from_top_gap: {first.get('from_top_gap', '')}",
                f"- first_expected_metric_delta: {first.get('expected_metric_delta', '')}",
                f"- first_scenario_fit: {first.get('scenario_fit', '')}",
                f"- first_architect_handoff: {first.get('architect_handoff', 'direct_execute_if_format_passes')}",
            ]
        )
    _write_if_changed(
        architect_path,
        "\n".join(architect_lines) + "\n",
    )

    guardian_path = role_dir / "guardian_latest.md"
    needs_human = sum(1 for proposal in proposals if bool(proposal.get("needs_human_approval", False)))
    _write_if_changed(
        guardian_path,
        "\n".join(
            [
                "# Guardian Output",
                "",
                f"- needs_human_approval_count: {needs_human}",
                "- status: clear" if needs_human == 0 else "- status: pending_human_gate",
            ]
        )
        + "\n",
    )

    arbiter_path = role_dir / "arbiter_latest.md"
    stagnation_rounds = int(recursive_state.get("stagnation_rounds", 0) or 0)
    recursive_mode = str(recursive_state.get("last_mode", "normal"))
    delta_focus = "expand_real_prompt_coverage" if stagnation_rounds % 3 == 0 else ("close_feedback_loop" if stagnation_rounds % 3 == 1 else "frontier_benchmarking")
    _write_if_changed(
        arbiter_path,
        "\n".join(
            [
                "# Arbiter Output",
                "",
                f"- verdict: {direction_review.get('verdict', 'direction_correct')}",
                f"- reasons: {','.join(direction_review.get('reasons', [])) if isinstance(direction_review.get('reasons', []), list) else ''}",
                f"- recursive_mode: {recursive_mode}",
                f"- stagnation_rounds: {stagnation_rounds}",
                f"- delta_focus_this_round: {delta_focus}",
                f"- new_action: {delta_focus} -> owner=arbiter, deadline=next_cycle",
            ]
        )
        + "\n",
    )

    arbiter_carm_track_path = role_dir / "arbiter_carm_track_latest.md"
    _write_if_changed(
        arbiter_carm_track_path,
        "\n".join(
            [
                "# Arbiter CARM Track Output",
                "",
                "- artifact: backlog/opportunities/carm_gap_map.md",
                "- objective: close top gaps towards CARM blueprint",
                "- status: updated",
            ]
        )
        + "\n",
    )

    return {
        "observer": observer_path,
        "benchmark_owner": benchmark_path,
        "evaluator": evaluator_path,
        "trainer": trainer_path,
        "architect": architect_path,
        "guardian": guardian_path,
        "arbiter": arbiter_path,
        "arbiter_carm_track": arbiter_carm_track_path,
    }


def _load_real_prompt_config(root: Path) -> list[dict[str, object]]:
    payload = _read_json(root / "configs" / "real_prompt_eval.json")
    prompts = payload.get("prompts", []) if isinstance(payload, dict) else []
    return [item for item in prompts if isinstance(item, dict)] if isinstance(prompts, list) else []


def _candidate_quality_reasons(item: dict[str, object]) -> list[str]:
    prompt = str(item.get("prompt", "")).strip()
    expected_tool = str(item.get("expected_tool", "")).strip()
    logic_skill = str(item.get("logic_skill", "")).strip()
    reasons: list[str] = []
    if not prompt or len(prompt) < 8:
        reasons.append("prompt_too_short")
    if prompt.startswith("观察学习样本") or "观察学习" in prompt:
        reasons.append("observer_learning_noise")
    if expected_tool not in {"search", "calculator", "bigmodel_proxy"}:
        reasons.append(f"tool_not_allowed:{expected_tool or 'empty'}")
    if not logic_skill:
        reasons.append("logic_skill_missing")
    return reasons


def _is_valid_real_prompt_candidate(item: dict[str, object]) -> bool:
    return not _candidate_quality_reasons(item)


def _build_candidate_pool_quality(root: Path, *, target_prompt_count: int = 20) -> dict[str, object]:
    existing = _load_real_prompt_config(root)
    existing_by_skill: dict[str, int] = {}
    for item in existing:
        logic_skill = str(item.get("logic_skill", "")).strip() or "unknown"
        existing_by_skill[logic_skill] = existing_by_skill.get(logic_skill, 0) + 1

    candidate_payload = _read_json(root / "data" / "eval" / "real_prompt_candidates.json")
    candidate_items = candidate_payload.get("prompts", []) if isinstance(candidate_payload, dict) else []
    if not isinstance(candidate_items, list):
        candidate_items = []

    filtered: list[dict[str, object]] = []
    dropped: list[dict[str, object]] = []
    filtered_by_skill: dict[str, int] = {}
    drop_reason_counts: dict[str, int] = {}

    for item in candidate_items:
        if not isinstance(item, dict):
            continue
        reasons = _candidate_quality_reasons(item)
        if reasons:
            dropped.append(
                {
                    "id": str(item.get("id", "")),
                    "expected_tool": str(item.get("expected_tool", "")),
                    "logic_skill": str(item.get("logic_skill", "")),
                    "reasons": reasons,
                }
            )
            for reason in reasons:
                drop_reason_counts[reason] = drop_reason_counts.get(reason, 0) + 1
            continue
        filtered.append(item)
        logic_skill = str(item.get("logic_skill", "")).strip() or "unknown"
        filtered_by_skill[logic_skill] = filtered_by_skill.get(logic_skill, 0) + 1

    gap_by_skill: dict[str, int] = {}
    for logic_skill in sorted(set(existing_by_skill) | set(filtered_by_skill)):
        current = existing_by_skill.get(logic_skill, 0)
        gap_by_skill[logic_skill] = max(0, 2 - current)

    total_gap = max(0, int(target_prompt_count) - len(existing))
    return {
        "existing_prompt_count": len(existing),
        "target_prompt_count": target_prompt_count,
        "total_gap": total_gap,
        "total_candidates": len(candidate_items),
        "filtered_candidates": len(filtered),
        "dropped_candidates": len(dropped),
        "drop_reason_counts": drop_reason_counts,
        "existing_by_logic_skill": existing_by_skill,
        "filtered_by_logic_skill": filtered_by_skill,
        "gap_by_logic_skill": gap_by_skill,
        "filtered_items": filtered,
        "dropped_items": dropped,
    }


def _write_candidate_pool_quality_json(root: Path, quality: dict[str, object]) -> Path:
    path = root / "data" / "evolution" / "candidate_pool_quality.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(quality, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


def _write_candidate_pool_quality_report(root: Path, quality: dict[str, object]) -> Path:
    report_path = root / "backlog" / "opportunities" / "candidate_quality_report.md"
    lines = [
        "# Candidate Quality Report",
        "",
        f"- total_candidates: {int(quality.get('total_candidates', 0) or 0)}",
        f"- filtered_candidates: {int(quality.get('filtered_candidates', 0) or 0)}",
        f"- dropped_candidates: {int(quality.get('dropped_candidates', 0) or 0)}",
        f"- existing_prompt_count: {int(quality.get('existing_prompt_count', 0) or 0)}",
        f"- target_prompt_count: {int(quality.get('target_prompt_count', 0) or 0)}",
        f"- total_gap: {int(quality.get('total_gap', 0) or 0)}",
        "",
        "## Drop Reason Counts",
    ]
    reason_counts = quality.get("drop_reason_counts", {})
    if isinstance(reason_counts, dict) and reason_counts:
        for reason, count in sorted(reason_counts.items(), key=lambda item: (-int(item[1]), str(item[0]))):
            lines.append(f"- {reason}: {count}")
    else:
        lines.append("- none")

    lines.extend(["", "## Existing Coverage By Logic Skill"])
    existing_by_skill = quality.get("existing_by_logic_skill", {})
    if isinstance(existing_by_skill, dict) and existing_by_skill:
        for skill, count in sorted(existing_by_skill.items()):
            lines.append(f"- {skill}: {count}")
    else:
        lines.append("- none")

    lines.extend(["", "## Filtered Candidate Coverage By Logic Skill"])
    filtered_by_skill = quality.get("filtered_by_logic_skill", {})
    if isinstance(filtered_by_skill, dict) and filtered_by_skill:
        for skill, count in sorted(filtered_by_skill.items()):
            lines.append(f"- {skill}: {count}")
    else:
        lines.append("- none")

    lines.extend(["", "## Gap By Logic Skill"])
    gap_by_skill = quality.get("gap_by_logic_skill", {})
    if isinstance(gap_by_skill, dict) and gap_by_skill:
        for skill, gap in sorted(gap_by_skill.items(), key=lambda item: (-int(item[1]), str(item[0]))):
            lines.append(f"- {skill}: gap={gap}")
    else:
        lines.append("- none")

    lines.extend(["", "## Drop Details"])
    dropped = quality.get("dropped_items", [])
    if isinstance(dropped, list) and dropped:
        for item in dropped:
            if isinstance(item, dict):
                lines.append(
                    f"- {item.get('id','')} | tool={item.get('expected_tool','')} | logic_skill={item.get('logic_skill','')} | reasons={','.join(item.get('reasons', []))}"
                )
    else:
        lines.append("- none")

    _write_if_changed(report_path, "\n".join(lines) + "\n")
    return report_path


def _load_repair_candidates(root: Path) -> list[dict[str, object]]:
    path = root / "data" / "evolution" / "repair" / "rebuild_candidates_for_missing_logic_skills.json"
    payload = _read_json(path)
    prompts = payload.get("prompts", []) if isinstance(payload, dict) else []
    return [item for item in prompts if isinstance(item, dict)] if isinstance(prompts, list) else []


def _rebuild_candidates_for_missing_logic_skills(root: Path, missing_logic_skills: list[str]) -> dict[str, object]:
    repair_dir = root / "data" / "evolution" / "repair"
    repair_dir.mkdir(parents=True, exist_ok=True)

    normalized_skills = [str(skill).strip() for skill in missing_logic_skills if str(skill).strip()]
    seen_prompts = {_normalize_text(str(item.get("prompt", ""))) for item in _load_real_prompt_config(root)}

    generated: list[dict[str, object]] = []
    for sample in generate_task_pool(seed=17, count_per_type=12):
        if sample.logic_skill not in normalized_skills:
            continue
        prompt = sample.user_input.strip()
        key = _normalize_text(prompt)
        if not prompt or key in seen_prompts:
            continue
        seen_prompts.add(key)
        generated.append(
            {
                "id": f"repair-{sample.logic_skill}-{len(generated)+1:03d}",
                "prompt": prompt,
                "expected_tool": sample.expected_tool,
                "logic_skill": sample.logic_skill,
                "source": "repair_task_pool",
            }
        )

    path = repair_dir / "rebuild_candidates_for_missing_logic_skills.json"
    path.write_text(json.dumps({"prompts": generated}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {
        "repair_operator": "rebuild_candidates_for_missing_logic_skills",
        "missing_logic_skills": normalized_skills,
        "generated_count": len(generated),
        "repair_candidates_path": str(path),
    }


def _apply_expand_eval_set_operator(root: Path, track_state: dict[str, object]) -> dict[str, object]:
    config_path = root / "configs" / "real_prompt_eval.json"
    candidates_path = root / "data" / "eval" / "real_prompt_candidates.json"

    existing = _load_real_prompt_config(root)
    existing_keys = {_normalize_text(str(item.get("prompt", ""))) for item in existing}
    existing_count = len(existing)
    target_count = int(track_state.get("target_prompt_count", 20) or 20)
    needed = max(0, target_count - existing_count)
    if needed <= 0:
        return {"executed": False, "reason": "target_already_reached", "added_count": 0}

    candidate_payload = _read_json(candidates_path)
    candidate_items = candidate_payload.get("prompts", []) if isinstance(candidate_payload, dict) else []
    if not isinstance(candidate_items, list):
        candidate_items = []

    quality = _build_candidate_pool_quality(root, target_prompt_count=target_count)
    quality_json_path = _write_candidate_pool_quality_json(root, quality)
    _write_candidate_pool_quality_report(root, quality)

    filtered_items = quality.get("filtered_items", [])
    by_logic_skill: dict[str, list[dict[str, object]]] = {}
    for item in filtered_items if isinstance(filtered_items, list) else []:
        if not isinstance(item, dict):
            continue
        key = _normalize_text(str(item.get("prompt", "")))
        if not key or key in existing_keys:
            continue
        logic_skill = str(item.get("logic_skill", "")).strip()
        by_logic_skill.setdefault(logic_skill, []).append(item)

    selected: list[dict[str, object]] = []
    skill_order = sorted(by_logic_skill.keys())
    while len(selected) < needed and skill_order:
        next_round: list[str] = []
        for skill in skill_order:
            bucket = by_logic_skill.get(skill, [])
            if not bucket:
                continue
            item = bucket.pop(0)
            selected.append(
                {
                    "id": str(item.get("id", f"candidate-{len(selected)+1:03d}")),
                    "prompt": str(item.get("prompt", "")),
                    "expected_tool": str(item.get("expected_tool", "search")),
                    "logic_skill": str(item.get("logic_skill", "tool_selection")),
                }
            )
            if len(selected) >= needed:
                break
            if bucket:
                next_round.append(skill)
        skill_order = next_round

    if not selected:
        gap_by_skill = quality.get("gap_by_logic_skill", {})
        uncovered_skills = [skill for skill, gap in gap_by_skill.items() if int(gap) > 0] if isinstance(gap_by_skill, dict) else []
        failure_reason = "insufficient_bucket_coverage" if uncovered_skills else "no_valid_candidates"
        repair_result = _rebuild_candidates_for_missing_logic_skills(root, uncovered_skills)

        repair_candidates = _load_repair_candidates(root)
        for item in repair_candidates:
            if not isinstance(item, dict) or not _is_valid_real_prompt_candidate(item):
                continue
            key = _normalize_text(str(item.get("prompt", "")))
            if not key or key in existing_keys:
                continue
            logic_skill = str(item.get("logic_skill", "")).strip()
            bucket = by_logic_skill.setdefault(logic_skill, [])
            if all(_normalize_text(str(existing_item.get("prompt", ""))) != key for existing_item in bucket):
                bucket.append(item)

        skill_order = sorted(by_logic_skill.keys())
        while len(selected) < needed and skill_order:
            next_round = []
            for skill in skill_order:
                bucket = by_logic_skill.get(skill, [])
                if not bucket:
                    continue
                item = bucket.pop(0)
                selected.append(
                    {
                        "id": str(item.get("id", f"candidate-{len(selected)+1:03d}")),
                        "prompt": str(item.get("prompt", "")),
                        "expected_tool": str(item.get("expected_tool", "search")),
                        "logic_skill": str(item.get("logic_skill", "tool_selection")),
                    }
                )
                existing_keys.add(_normalize_text(str(item.get("prompt", ""))))
                if len(selected) >= needed:
                    break
                if bucket:
                    next_round.append(skill)
            skill_order = next_round

        if not selected:
            return {
                "executed": False,
                "reason": failure_reason,
                "added_count": 0,
                "repair_operator": "filter_candidate_pool",
                "repair_hint": "rebuild_candidates_for_missing_logic_skills",
                "missing_logic_skills": uncovered_skills,
                "quality_report_path": str(root / "backlog" / "opportunities" / "candidate_quality_report.md"),
                "quality_json_path": str(quality_json_path),
                "repair_result": repair_result,
            }

    merged = existing + selected
    config_path.write_text(json.dumps({"prompts": merged}, ensure_ascii=False, indent=2), encoding="utf-8")

    latest_path = root / "data" / "eval" / "real_prompt_eval_latest.json"
    latest_path.parent.mkdir(parents=True, exist_ok=True)

    result = {
        "executed": True,
        "reason": "expanded_eval_set",
        "added_count": len(selected),
        "before_count": existing_count,
        "after_count": len(merged),
        "added_prompt_ids": [str(item.get("id", "")) for item in selected],
        "used_repair_candidates": any(str(item.get("id", "")).startswith("repair-") for item in selected),
        "quality_report_path": str(root / "backlog" / "opportunities" / "candidate_quality_report.md"),
        "quality_json_path": str(quality_json_path),
    }

    try:
        eval_proc = subprocess.run(
            ["python", "-m", "scripts.evaluate_real_prompts", str(config_path)],
            cwd=str(root),
            check=True,
            capture_output=True,
            text=True,
        )
        eval_payload = json.loads(eval_proc.stdout)
        latest_path.write_text(json.dumps(eval_payload, ensure_ascii=False, indent=2), encoding="utf-8")
        result["real_prompt_eval_path"] = str(latest_path)
        result["real_prompt_eval"] = eval_payload
    except Exception as exc:
        result["evaluation_error"] = str(exc)

    return result


def _maybe_execute_real_prompt_operator(root: Path, signals: dict[str, object]) -> dict[str, object]:
    track_state = _build_real_prompt_track_state(signals)
    if str(track_state.get("operator", "")) != "expand_eval_set":
        return {"executed": False, "reason": "operator_not_needed", "track_state": track_state}
    try:
        result = _apply_expand_eval_set_operator(root, track_state)
        result["track_state"] = track_state
        return result
    except Exception as exc:
        return {
            "executed": False,
            "reason": f"operator_failed:{exc}",
            "track_state": track_state,
        }


def _build_quality_stabilization_payload(signals: dict[str, object]) -> dict[str, object]:
    rows = []
    real_prompt_eval = signals.get("real_prompt_eval", {})
    if isinstance(real_prompt_eval, dict):
        raw_rows = real_prompt_eval.get("rows", [])
        if isinstance(raw_rows, list):
            rows = [row for row in raw_rows if isinstance(row, dict)]

    scored_rows: list[dict[str, object]] = []
    high_signal_count = 0
    bigmodel_proxy_mismatch_count = 0
    for row in rows:
        baseline_match = bool(row.get("baseline_match", False))
        pretrained_match = bool(row.get("pretrained_match", False))
        baseline_tool = str(row.get("baseline_used_tool", ""))
        expected_tool = str(row.get("expected_tool", ""))
        logic_skill = str(row.get("logic_skill", ""))

        separation_score = 0
        if (not baseline_match) and pretrained_match:
            separation_score += 3
        if baseline_tool == "bigmodel_proxy" and expected_tool != "bigmodel_proxy":
            separation_score += 2
            bigmodel_proxy_mismatch_count += 1
        if logic_skill in {"comparison", "conflict_detection", "result_integration"}:
            separation_score += 1
        if separation_score >= 3:
            high_signal_count += 1

        scored_rows.append(
            {
                "id": str(row.get("id", "")),
                "logic_skill": logic_skill,
                "expected_tool": expected_tool,
                "baseline_used_tool": baseline_tool,
                "pretrained_used_tool": str(row.get("pretrained_used_tool", "")),
                "baseline_match": baseline_match,
                "pretrained_match": pretrained_match,
                "separation_score": separation_score,
            }
        )

    scored_rows.sort(key=lambda item: (-int(item.get("separation_score", 0)), str(item.get("id", ""))))
    return {
        "row_count": len(scored_rows),
        "high_signal_count": high_signal_count,
        "bigmodel_proxy_mismatch_count": bigmodel_proxy_mismatch_count,
        "top_rows": scored_rows[:10],
        "rows": scored_rows,
    }


def _build_quality_focus_eval(root: Path, payload: dict[str, object], hard_variants: list[dict[str, object]]) -> dict[str, object]:
    top_rows = payload.get("top_rows", []) if isinstance(payload, dict) else []
    focus_prompts: list[dict[str, object]] = []
    seen_ids: set[str] = set()

    current_prompts = _load_real_prompt_config(root)
    prompts_by_id = {str(item.get("id", "")): item for item in current_prompts if isinstance(item, dict)}

    for item in top_rows if isinstance(top_rows, list) else []:
        if not isinstance(item, dict):
            continue
        row_id = str(item.get("id", ""))
        prompt_item = prompts_by_id.get(row_id)
        if not isinstance(prompt_item, dict):
            continue
        if row_id in seen_ids:
            continue
        seen_ids.add(row_id)
        focus_prompts.append(
            {
                "id": row_id,
                "prompt": str(prompt_item.get("prompt", "")),
                "expected_tool": str(prompt_item.get("expected_tool", "search")),
                "logic_skill": str(prompt_item.get("logic_skill", "")),
                "source": "quality_top_row",
            }
        )

    for item in hard_variants:
        if not isinstance(item, dict):
            continue
        row_id = str(item.get("id", ""))
        if row_id in seen_ids:
            continue
        seen_ids.add(row_id)
        focus_prompts.append(item)

    path = root / "data" / "evolution" / "quality_focus_eval.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"prompts": focus_prompts}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {"path": path, "count": len(focus_prompts), "prompts": focus_prompts}


def _run_quality_focus_eval(root: Path, eval_path: Path) -> dict[str, object]:
    result_path = root / "data" / "evolution" / "quality_focus_eval_result.json"
    try:
        proc = subprocess.run(
            ["python", "-m", "scripts.evaluate_real_prompts", str(eval_path)],
            cwd=str(root),
            check=True,
            capture_output=True,
            text=True,
        )
        payload = json.loads(proc.stdout)
        result_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return {"executed": True, "result_path": str(result_path), "payload": payload}
    except Exception as exc:
        return {"executed": False, "reason": str(exc), "result_path": str(result_path)}


def _promote_quality_benchmark(root: Path) -> dict[str, object]:
    retained_path = root / "data" / "evolution" / "quality_benchmark_retained.json"
    payload = _read_json(retained_path)
    prompts = payload.get("prompts", []) if isinstance(payload, dict) else []
    prompts = [item for item in prompts if isinstance(item, dict)] if isinstance(prompts, list) else []

    current_prompts = _load_real_prompt_config(root)
    prompts_by_id = {str(item.get("id", "")): item for item in current_prompts if isinstance(item, dict)}

    quality_focus_payload = _read_json(root / "data" / "evolution" / "quality_focus_eval.json")
    quality_focus_prompts = quality_focus_payload.get("prompts", []) if isinstance(quality_focus_payload, dict) else []
    quality_focus_by_id = {str(item.get("id", "")): item for item in quality_focus_prompts if isinstance(item, dict)} if isinstance(quality_focus_prompts, list) else {}

    hard_variant_payload = _read_json(root / "data" / "evolution" / "quality_hard_variants.json")
    hard_variant_prompts = hard_variant_payload.get("prompts", []) if isinstance(hard_variant_payload, dict) else []
    hard_variant_by_id = {str(item.get("id", "")): item for item in hard_variant_prompts if isinstance(item, dict)} if isinstance(hard_variant_prompts, list) else {}

    promoted: list[dict[str, object]] = []
    for item in prompts:
        prompt_id = str(item.get("id", ""))
        current = prompts_by_id.get(prompt_id)
        quality_focus = quality_focus_by_id.get(prompt_id)
        hard_variant = hard_variant_by_id.get(prompt_id)
        source = current or quality_focus or hard_variant or {}
        promoted.append(
            {
                "id": prompt_id,
                "prompt": str(source.get("prompt", "")),
                "expected_tool": str(item.get("expected_tool", source.get("expected_tool", "search"))),
                "logic_skill": str(item.get("logic_skill", source.get("logic_skill", ""))),
                "promotion_reason": str(item.get("retain_reason", "")),
            }
        )

    config_path = root / "configs" / "quality_benchmark_eval.json"
    config_path.write_text(json.dumps({"prompts": promoted}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    promotion_payload = {
        "promoted_count": len(promoted),
        "config_path": str(config_path),
        "source_path": str(retained_path),
    }
    json_path = root / "data" / "evolution" / "quality_benchmark_promotion.json"
    json_path.write_text(json.dumps(promotion_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    report_path = root / "backlog" / "opportunities" / "quality_benchmark_promotion_report.md"
    lines = [
        "# Quality Benchmark Promotion Report",
        "",
        f"- promoted_count: {len(promoted)}",
        f"- config_path: {config_path}",
        "",
        "## Promoted Prompts",
    ]
    for item in promoted:
        lines.append(f"- {item.get('id','')} | logic_skill={item.get('logic_skill','')} | reason={item.get('promotion_reason','')}")
    _write_if_changed(report_path, "\n".join(lines) + "\n")

    return {
        "config_path": str(config_path),
        "json_path": str(json_path),
        "report_path": str(report_path),
        "promoted_count": len(promoted),
    }


def _build_quality_benchmark_subset(root: Path, quality_focus_payload: dict[str, object]) -> dict[str, object]:
    rows = quality_focus_payload.get("rows", []) if isinstance(quality_focus_payload, dict) else []
    retained: list[dict[str, object]] = []
    pruned: list[dict[str, object]] = []
    support_kept_by_skill: dict[str, int] = {}

    focus_eval_payload = _read_json(root / "data" / "evolution" / "quality_focus_eval.json")
    focus_eval_prompts = focus_eval_payload.get("prompts", []) if isinstance(focus_eval_payload, dict) else []
    focus_prompt_by_id = {str(item.get("id", "")): item for item in focus_eval_prompts if isinstance(item, dict)} if isinstance(focus_eval_prompts, list) else {}

    for row in rows if isinstance(rows, list) else []:
        if not isinstance(row, dict):
            continue
        logic_skill = str(row.get("logic_skill", ""))
        baseline_match = bool(row.get("baseline_match", False))
        pretrained_match = bool(row.get("pretrained_match", False))
        baseline_tool = str(row.get("baseline_used_tool", ""))
        expected_tool = str(row.get("expected_tool", ""))

        prompt_id = str(row.get("id", ""))
        focus_prompt = focus_prompt_by_id.get(prompt_id, {})
        payload = {
            "id": prompt_id,
            "prompt": str(focus_prompt.get("prompt", row.get("prompt", ""))),
            "logic_skill": logic_skill,
            "expected_tool": expected_tool,
            "baseline_used_tool": baseline_tool,
            "pretrained_used_tool": str(row.get("pretrained_used_tool", "")),
            "baseline_match": baseline_match,
            "pretrained_match": pretrained_match,
        }

        if (not baseline_match) and pretrained_match:
            payload["retain_reason"] = "separation_positive"
            retained.append(payload)
            continue

        if baseline_match and pretrained_match:
            kept = int(support_kept_by_skill.get(logic_skill, 0) or 0)
            if kept < 1 and logic_skill in {"comparison", "conflict_detection", "tool_selection"}:
                payload["retain_reason"] = "context_support"
                retained.append(payload)
                support_kept_by_skill[logic_skill] = kept + 1
            else:
                payload["prune_reason"] = "low_signal_both_match"
                pruned.append(payload)
            continue

        payload["prune_reason"] = "unclassified_low_value"
        pruned.append(payload)

    retained_path = root / "data" / "evolution" / "quality_benchmark_retained.json"
    pruned_path = root / "data" / "evolution" / "quality_benchmark_pruned.json"
    retained_path.write_text(json.dumps({"prompts": retained}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    pruned_path.write_text(json.dumps({"prompts": pruned}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    report_path = root / "backlog" / "opportunities" / "quality_benchmark_report.md"
    lines = [
        "# Quality Benchmark Report",
        "",
        f"- retained_count: {len(retained)}",
        f"- pruned_count: {len(pruned)}",
        "",
        "## Retained",
    ]
    for item in retained:
        lines.append(f"- {item.get('id','')} | logic_skill={item.get('logic_skill','')} | reason={item.get('retain_reason','')}")
    lines.extend(["", "## Pruned"])
    for item in pruned:
        lines.append(f"- {item.get('id','')} | logic_skill={item.get('logic_skill','')} | reason={item.get('prune_reason','')}")
    _write_if_changed(report_path, "\n".join(lines) + "\n")

    return {
        "retained_path": str(retained_path),
        "pruned_path": str(pruned_path),
        "report_path": str(report_path),
        "retained_count": len(retained),
        "pruned_count": len(pruned),
    }


def _generate_quality_hard_variants(root: Path, payload: dict[str, object]) -> dict[str, object]:
    top_rows = payload.get("top_rows", []) if isinstance(payload, dict) else []
    variants: list[dict[str, object]] = []
    for item in top_rows if isinstance(top_rows, list) else []:
        if not isinstance(item, dict):
            continue
        logic_skill = str(item.get("logic_skill", ""))
        expected_tool = str(item.get("expected_tool", "search"))
        row_id = str(item.get("id", ""))
        if logic_skill == "comparison":
            variants.append(
                {
                    "id": f"quality-{row_id}-01",
                    "logic_skill": logic_skill,
                    "expected_tool": expected_tool,
                    "prompt": "两篇数据库选型文章都在比较 PostgreSQL 和 MySQL，但一个强调维护复杂度，另一个强调成本。面向中小团队时，应该先怎么建立比较框架再决定查哪些资料？",
                    "source_row": row_id,
                }
            )
            variants.append(
                {
                    "id": f"quality-{row_id}-02",
                    "logic_skill": logic_skill,
                    "expected_tool": expected_tool,
                    "prompt": "如果要给负责人写数据库选型建议，在比较 PostgreSQL 和 MySQL 前，哪些比较维度必须先固定，否则后面的检索和结论都会漂？",
                    "source_row": row_id,
                }
            )
        elif logic_skill == "conflict_detection":
            variants.append(
                {
                    "id": f"quality-{row_id}-01",
                    "logic_skill": logic_skill,
                    "expected_tool": expected_tool,
                    "prompt": "官方文档和社区文章对同一个数据库迁移步骤给出了相反建议，在没消解冲突前，应该先做什么而不是直接写结论？",
                    "source_row": row_id,
                }
            )
            variants.append(
                {
                    "id": f"quality-{row_id}-02",
                    "logic_skill": logic_skill,
                    "expected_tool": expected_tool,
                    "prompt": "两个来源对同一参数是否该调整给出了相反建议，如果其中一个来源更新较旧，应该先如何判断冲突来源和证据优先级？",
                    "source_row": row_id,
                }
            )

    path = root / "data" / "evolution" / "quality_hard_variants.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"prompts": variants}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {"path": path, "count": len(variants), "prompts": variants}


def _maybe_execute_quality_operator(root: Path, track_state: dict[str, object], quality_payload: dict[str, object]) -> dict[str, object]:
    if str(track_state.get("operator", "")) != "stabilize_quality_gap":
        return {"executed": False, "reason": "quality_operator_not_needed"}
    generated = _generate_quality_hard_variants(root, quality_payload)
    hard_variants = generated.get("prompts", []) if isinstance(generated.get("prompts", []), list) else []
    focus_eval = _build_quality_focus_eval(root, quality_payload, hard_variants)
    focus_eval_result = _run_quality_focus_eval(root, Path(str(focus_eval.get("path", ""))))
    benchmark_result = {}
    if bool(focus_eval_result.get("executed")) and isinstance(focus_eval_result.get("payload"), dict):
        benchmark_result = _build_quality_benchmark_subset(root, focus_eval_result.get("payload", {}))
        if bool(benchmark_result):
            benchmark_result["promotion"] = _promote_quality_benchmark(root)
    return {
        "executed": True,
        "reason": "generated_quality_hard_variants",
        "generated_count": int(generated.get("count", 0) or 0),
        "quality_hard_variants_path": str(generated.get("path", "")),
        "quality_focus_eval_path": str(focus_eval.get("path", "")),
        "quality_focus_eval_count": int(focus_eval.get("count", 0) or 0),
        "quality_focus_eval_result": focus_eval_result,
        "quality_benchmark_result": benchmark_result,
    }


def _write_quality_stabilization_artifacts(root: Path, signals: dict[str, object]) -> dict[str, object]:
    payload = _build_quality_stabilization_payload(signals)
    json_path = root / "data" / "evolution" / "quality_stabilization.json"
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    report_path = root / "backlog" / "opportunities" / "quality_stabilization_report.md"
    lines = [
        "# Quality Stabilization Report",
        "",
        f"- row_count: {int(payload.get('row_count', 0) or 0)}",
        f"- high_signal_count: {int(payload.get('high_signal_count', 0) or 0)}",
        f"- bigmodel_proxy_mismatch_count: {int(payload.get('bigmodel_proxy_mismatch_count', 0) or 0)}",
        "",
        "## Top Separation Rows",
    ]
    top_rows = payload.get("top_rows", [])
    if isinstance(top_rows, list) and top_rows:
        for item in top_rows:
            if isinstance(item, dict):
                lines.append(
                    f"- {item.get('id','')} | logic_skill={item.get('logic_skill','')} | baseline={item.get('baseline_used_tool','')} | pretrained={item.get('pretrained_used_tool','')} | expected={item.get('expected_tool','')} | score={item.get('separation_score',0)}"
                )
    else:
        lines.append("- none")
    _write_if_changed(report_path, "\n".join(lines) + "\n")
    return {"json_path": json_path, "report_path": report_path, "payload": payload}


def _build_real_prompt_track_state(signals: dict[str, object]) -> dict[str, object]:
    summary = {}
    rows: list[dict[str, object]] = []
    real_prompt_eval = signals.get("real_prompt_eval", {})
    if isinstance(real_prompt_eval, dict):
        raw_summary = real_prompt_eval.get("summary", {})
        if isinstance(raw_summary, dict):
            summary = raw_summary
        raw_rows = real_prompt_eval.get("rows", [])
        if isinstance(raw_rows, list):
            rows = [row for row in raw_rows if isinstance(row, dict)]

    prompt_count = int(summary.get("prompt_count", 0) or 0)
    match_rate = float(summary.get("pretrained_match_rate", 0.0) or 0.0)
    baseline_match_rate = float(summary.get("baseline_match_rate", 0.0) or 0.0)
    mismatch_case_count = sum(1 for row in rows if not bool(row.get("pretrained_match", True)))
    target_prompt_count = 20

    quality_payload = _build_quality_stabilization_payload(signals)
    high_signal_count = int(quality_payload.get("high_signal_count", 0) or 0)
    bigmodel_proxy_mismatch_count = int(quality_payload.get("bigmodel_proxy_mismatch_count", 0) or 0)

    top_gap = "coverage_not_quality" if prompt_count < target_prompt_count else "quality_stabilization"
    operator = "expand_eval_set" if prompt_count < target_prompt_count else ("stabilize_quality_gap" if high_signal_count > 0 else "promote_baseline")

    if operator == "expand_eval_set":
        status = "proposed"
        decision = "advance"
        failure_reason = "real_prompt_count_below_target"
        repair_hint = "curate_additional_real_prompts_by_logic_skill"
    elif operator == "stabilize_quality_gap":
        status = "in_progress"
        decision = "stabilize_quality_gap"
        failure_reason = "high_signal_quality_gap_active"
        repair_hint = "inject_repair_and_quality_focus_prompts"
    else:
        status = "accepted"
        decision = "promote_baseline"
        failure_reason = ""
        repair_hint = ""

    return {
        "track": "real_prompt_coverage",
        "target_gap": top_gap,
        "current_prompt_count": prompt_count,
        "target_prompt_count": target_prompt_count,
        "current_match_rate": match_rate,
        "baseline_match_rate": baseline_match_rate,
        "mismatch_case_count": mismatch_case_count,
        "high_signal_count": high_signal_count,
        "bigmodel_proxy_mismatch_count": bigmodel_proxy_mismatch_count,
        "operator": operator,
        "operator_args": {
            "target_prompt_count": target_prompt_count,
            "current_prompt_count": prompt_count,
        },
        "hypothesis": "Increase real prompt coverage without reducing current match rate.",
        "expected_metric_delta": {
            "prompt_count": max(0, target_prompt_count - prompt_count),
            "pretrained_match_rate": 0.0,
        },
        "status": status,
        "decision": decision,
        "failure_reason": failure_reason,
        "repair_hint": repair_hint,
    }


def _write_evolution_artifacts(root: Path, signals: dict[str, object], operator_result: dict[str, object] | None = None) -> dict[str, object]:
    track_state = _build_real_prompt_track_state(signals)
    operator_result = operator_result if isinstance(operator_result, dict) else {}
    lineages_dir = root / "data" / "evolution" / "lineages"
    candidates_dir = root / "data" / "evolution" / "candidates"
    runs_dir = root / "data" / "evolution" / "runs"
    lineages_dir.mkdir(parents=True, exist_ok=True)
    candidates_dir.mkdir(parents=True, exist_ok=True)
    runs_dir.mkdir(parents=True, exist_ok=True)

    lineage_path = lineages_dir / f"{track_state['track']}.json"
    lineage_payload = _read_json(lineage_path)
    if not isinstance(lineage_payload, dict):
        lineage_payload = {}
    parent_id = str(lineage_payload.get("current_candidate_id", ""))

    stamp = utc_now().strftime("%Y%m%dT%H%M%SZ")
    candidate_id = f"{track_state['track']}_{stamp}"
    candidate_payload = {
        "id": candidate_id,
        "parent_id": parent_id,
        "track": track_state["track"],
        "target_gap": track_state["target_gap"],
        "operator": track_state["operator"],
        "operator_args": track_state["operator_args"],
        "hypothesis": track_state["hypothesis"],
        "expected_metric_delta": track_state["expected_metric_delta"],
        "artifact_paths": {
            "real_prompt_eval": str(root / "data" / "eval" / "real_prompt_eval_latest.json"),
            "real_prompt_config": str(root / "configs" / "real_prompt_eval.json"),
        },
        "status": track_state["status"],
        "score": {
            "prompt_count": track_state["current_prompt_count"],
            "pretrained_match_rate": track_state["current_match_rate"],
            "mismatch_case_count": track_state["mismatch_case_count"],
        },
        "decision": track_state["decision"],
        "failure_reason": str(operator_result.get("reason", "")) or track_state["failure_reason"],
        "repair_hint": str(operator_result.get("repair_hint", "")) or track_state["repair_hint"],
        "repair_operator": str(operator_result.get("repair_operator", "")),
        "missing_logic_skills": list(operator_result.get("missing_logic_skills", [])) if isinstance(operator_result.get("missing_logic_skills", []), list) else [],
        "created_at_utc": stamp,
    }
    candidate_path = candidates_dir / f"{candidate_id}.json"
    candidate_path.write_text(json.dumps(candidate_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    history = lineage_payload.get("history", [])
    if not isinstance(history, list):
        history = []
    history.append(
        {
            "candidate_id": candidate_id,
            "status": track_state["status"],
            "decision": track_state["decision"],
            "prompt_count": track_state["current_prompt_count"],
            "pretrained_match_rate": track_state["current_match_rate"],
            "created_at_utc": stamp,
        }
    )
    lineage_payload = {
        "track": track_state["track"],
        "current_candidate_id": candidate_id,
        "parent_candidate_id": parent_id,
        "top_gap": track_state["target_gap"],
        "target_prompt_count": track_state["target_prompt_count"],
        "current_prompt_count": track_state["current_prompt_count"],
        "current_match_rate": track_state["current_match_rate"],
        "baseline_match_rate": track_state["baseline_match_rate"],
        "mismatch_case_count": track_state["mismatch_case_count"],
        "allowed_operators": [
            "expand_eval_set",
            "filter_candidate_pool",
            "repair_label_mapping",
            "rerun_eval",
            "promote_baseline",
        ],
        "history": history[-20:],
        "last_operator_result": operator_result,
        "updated_at_utc": stamp,
    }
    lineage_path.write_text(json.dumps(lineage_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    run_payload = {
        "track": track_state["track"],
        "candidate_id": candidate_id,
        "parent_id": parent_id,
        "operator": track_state["operator"],
        "decision": track_state["decision"],
        "status": track_state["status"],
        "signals": {
            "current_prompt_count": track_state["current_prompt_count"],
            "target_prompt_count": track_state["target_prompt_count"],
            "current_match_rate": track_state["current_match_rate"],
            "baseline_match_rate": track_state["baseline_match_rate"],
            "mismatch_case_count": track_state["mismatch_case_count"],
        },
        "operator_result": operator_result,
        "created_at_utc": stamp,
    }
    run_path = runs_dir / f"{stamp}_{candidate_id}.json"
    run_path.write_text(json.dumps(run_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    return {
        "track": track_state["track"],
        "summary": track_state,
        "candidate_path": candidate_path,
        "lineage_path": lineage_path,
        "run_path": run_path,
    }


def _safe_git_changed_paths(root: Path) -> list[str]:
    try:
        proc = subprocess.run(
            ["git", "diff", "--name-only"],
            cwd=str(root),
            check=True,
            capture_output=True,
            text=True,
        )
        paths = [line.strip().replace("\\", "/") for line in proc.stdout.splitlines() if line.strip()]
        return paths
    except Exception:
        return []


def _classify_delivery_paths(paths: list[str]) -> dict[str, list[str]]:
    groups = {"core": [], "artifacts": [], "volatile": [], "other": []}

    for raw_path in paths:
        path = raw_path.replace("\\", "/").strip()
        if not path:
            continue

        if (
            path.startswith("scripts/")
            or path.startswith("tests/")
            or path == "configs/team_cycle.json"
            or path == "configs/real_prompt_eval.json"
            or path == "configs/team_github.json"
            or path.startswith("docs/plans/")
        ):
            groups["core"].append(path)
        elif (
            path.startswith("data/team/")
            or path.startswith("data/evolution/")
            or path.endswith(".jsonl")
        ):
            groups["volatile"].append(path)
        elif (
            path.startswith("backlog/")
            or path.startswith("memory/daily/")
            or path == "data/eval/real_prompt_eval_latest.json"
            or path.startswith("team/")
        ):
            groups["artifacts"].append(path)
        elif path.startswith("docs/"):
            groups["core"].append(path)
        else:
            groups["other"].append(path)

    return groups


def _build_delivery_decision(
    signals: dict[str, object],
    direction_review: dict[str, object],
    changed_paths: list[str],
    config: dict[str, object],
) -> dict[str, object]:
    file_groups = _classify_delivery_paths(changed_paths)
    core = file_groups.get("core", [])
    artifacts = file_groups.get("artifacts", [])
    other = file_groups.get("other", [])

    if not changed_paths:
        return {
            "should_commit": False,
            "should_push": False,
            "should_open_pr": False,
            "should_merge": False,
            "delivery_lane": "skip",
            "reason": "git_unavailable_or_no_changes",
            "file_groups": file_groups,
        }

    review = direction_review if isinstance(direction_review, dict) else {}
    verdict = str(review.get("verdict", "")).strip()
    escalate = bool(review.get("escalate_to_human", False))
    alerts = signals.get("alerts", []) if isinstance(signals, dict) else []
    alerts = alerts if isinstance(alerts, list) else []

    github_cfg = config.get("github_delivery", {}) if isinstance(config, dict) else {}
    github_cfg = github_cfg if isinstance(github_cfg, dict) else {}
    pr_enabled = bool(github_cfg.get("enabled", False))
    require_direction_correct = bool(github_cfg.get("require_direction_correct", True))
    require_clean_alerts = bool(github_cfg.get("require_clean_alerts", True))

    if core:
        pr_ready = pr_enabled and not escalate
        if require_direction_correct:
            pr_ready = pr_ready and verdict == "direction_correct"
        if require_clean_alerts:
            pr_ready = pr_ready and not alerts

        if pr_ready:
            return {
                "should_commit": True,
                "should_push": True,
                "should_open_pr": True,
                "should_merge": False,
                "delivery_lane": "pr_delivery",
                "reason": "core_changes_ready_for_pr",
                "file_groups": file_groups,
            }

        return {
            "should_commit": True,
            "should_push": True,
            "should_open_pr": False,
            "should_merge": False,
            "delivery_lane": "sync_only",
            "reason": "core_changes_detected",
            "file_groups": file_groups,
        }

    if artifacts or other:
        return {
            "should_commit": True,
            "should_push": False,
            "should_open_pr": False,
            "should_merge": False,
            "delivery_lane": "sync_with_artifacts",
            "reason": "artifact_or_doc_changes_detected",
            "file_groups": file_groups,
        }

    return {
        "should_commit": False,
        "should_push": False,
        "should_open_pr": False,
        "should_merge": False,
        "delivery_lane": "skip",
        "reason": "no_deliverable_changes",
        "file_groups": file_groups,
    }


def run_cycle(root: Path = Path("."), config_path: Path = DEFAULT_CONFIG_PATH) -> dict[str, object]:
    bootstrap_workspace(root)
    cleanup_stats = _cleanup_low_value_artifacts(root)
    config = load_team_config(root / config_path if not config_path.is_absolute() else config_path)

    eval_pipeline = config.get("eval_pipeline", {})
    if not isinstance(eval_pipeline, dict):
        eval_pipeline = {}
    auto_sync = bool(eval_pipeline.get("auto_sync_real_prompt_eval", False))

    pipeline_stats: dict[str, object]
    if auto_sync:
        pipeline_stats = _sync_real_prompt_eval_pipeline(root)
    else:
        pipeline_stats = {"skipped": True, "reason": "auto_sync_disabled"}

    eval_stability = _track_eval_config_stability(root)

    signals = collect_signals(root)
    rp_payload = pipeline_stats.get("real_prompt_eval")
    if isinstance(rp_payload, dict):
        signals["real_prompt_eval"] = rp_payload

    operator_result = _maybe_execute_real_prompt_operator(root, signals)
    operator_eval = operator_result.get("real_prompt_eval")
    if isinstance(operator_eval, dict):
        signals["real_prompt_eval"] = operator_eval
    else:
        refreshed_signals = collect_signals(root)
        refreshed_eval = refreshed_signals.get("real_prompt_eval")
        if isinstance(refreshed_eval, dict):
            signals["real_prompt_eval"] = refreshed_eval

    quality_artifacts = _write_quality_stabilization_artifacts(root, signals)
    signals["quality_stabilization"] = quality_artifacts.get("payload", {})
    quality_operator_result = _maybe_execute_quality_operator(
        root,
        _build_real_prompt_track_state(signals),
        quality_artifacts.get("payload", {}),
    )
    if isinstance(quality_operator_result, dict):
        focus_result = quality_operator_result.get("quality_focus_eval_result", {})
        if isinstance(focus_result, dict) and isinstance(focus_result.get("payload"), dict):
            signals["quality_focus_eval_result"] = focus_result
        result_path = focus_result.get("result_path", "") if isinstance(focus_result, dict) else ""
        if result_path:
            signals["quality_focus_eval_result_path"] = str(result_path)

    recursive_state = _update_recursive_state(root, signals, config)
    signals["recursive_state"] = recursive_state
    digest = build_daily_digest(signals, config)
    digest["workspace_root"] = str(root)
    digest["recursive_state"] = recursive_state

    failure_patterns_path = _write_failure_patterns(root, signals)
    top_gap_path = _write_top_gap_action_card(root, signals)
    carm_gap_map_path = _write_carm_gap_map(root, signals)
    research_brief_path = _write_research_brief(root, signals, config)
    researcher_artifact_path = _run_researcher(root, signals, recursive_state)
    signals["researcher_artifact_text"] = researcher_artifact_path.read_text(encoding="utf-8", errors="ignore") if researcher_artifact_path.exists() else ""
    research_quality_path, research_quality = _evaluate_research_quality(
        root,
        signals,
        config,
        failure_patterns_path=root / "memory" / "failure_patterns.md",
        researcher_artifact_path=researcher_artifact_path,
    )
    digest["research_quality"] = research_quality
    signals["research_quality"] = research_quality
    quality_artifacts = _write_quality_stabilization_artifacts(root, signals)
    signals["quality_stabilization"] = quality_artifacts.get("payload", {})
    quality_operator_result = _maybe_execute_quality_operator(
        root,
        _build_real_prompt_track_state(signals),
        quality_artifacts.get("payload", {}),
    )
    if isinstance(quality_operator_result, dict):
        focus_result = quality_operator_result.get("quality_focus_eval_result", {})
        if isinstance(focus_result, dict) and isinstance(focus_result.get("payload"), dict):
            signals["quality_focus_eval_result"] = focus_result
        result_path = focus_result.get("result_path", "") if isinstance(focus_result, dict) else ""
        if result_path:
            signals["quality_focus_eval_result_path"] = str(result_path)

    research_recovery_report_path, research_recovery = _run_research_recovery_operators(
        root,
        signals,
        research_quality,
        config,
    )
    digest["research_recovery"] = research_recovery
    failure_patterns_path = _write_failure_patterns(root, signals)
    research_quality_path, research_quality = _evaluate_research_quality(
        root,
        signals,
        config,
        failure_patterns_path=root / "backlog" / "incidents" / "auto_failure_patterns.json",
        researcher_artifact_path=researcher_artifact_path,
    )
    digest["research_quality"] = research_quality
    signals["research_quality"] = research_quality
    _write_if_changed(Path(research_quality_path), json.dumps(research_quality, ensure_ascii=False, indent=2) + "\n")
    refreshed_signals = collect_signals(root)
    refreshed_signals["quality_stabilization"] = signals.get("quality_stabilization", {})
    refreshed_signals["recursive_state"] = recursive_state
    refreshed_signals["research_quality"] = research_quality
    refreshed_signals["researcher_artifact_text"] = signals.get("researcher_artifact_text", "")
    if "quality_focus_eval_result" in signals:
        refreshed_signals["quality_focus_eval_result"] = signals["quality_focus_eval_result"]
    if "quality_focus_eval_result_path" in signals:
        refreshed_signals["quality_focus_eval_result_path"] = signals["quality_focus_eval_result_path"]
    signals = refreshed_signals
    digest["signals"] = signals

    proposals = build_proposals(digest, config, recursive_state=recursive_state)

    direction_review = digest.get("direction_review", {})
    if not isinstance(direction_review, dict):
        direction_review = {}

    if bool(eval_stability.get("eval_config_changed_vs_last", False)) and not auto_sync:
        direction_review["verdict"] = "uncertain_needs_human"
        reasons = direction_review.get("reasons", [])
        if not isinstance(reasons, list):
            reasons = []
        reasons.append("eval_config_changed_while_auto_sync_disabled")
        direction_review["reasons"] = reasons
        direction_review["escalate_to_human"] = True
        alerts = digest.get("alerts", [])
        if not isinstance(alerts, list):
            alerts = []
        alerts.append("eval_config_unexpected_change")
        digest["alerts"] = alerts

    if not researcher_artifact_path.exists():
        direction_review["verdict"] = "uncertain_needs_human"
        reasons = direction_review.get("reasons", [])
        if not isinstance(reasons, list):
            reasons = []
        reasons.append("researcher_artifact_missing")
        direction_review["reasons"] = reasons
        direction_review["escalate_to_human"] = True
        alerts = digest.get("alerts", [])
        if not isinstance(alerts, list):
            alerts = []
        alerts.append("researcher_artifact_missing")
        digest["alerts"] = alerts

    role_artifacts = _write_role_artifacts(root, signals, proposals, direction_review, recursive_state)
    role_artifacts["researcher"] = researcher_artifact_path
    role_output_status = {role: path.exists() for role, path in role_artifacts.items()}
    missing_roles = [role for role, ok in role_output_status.items() if not ok]
    if missing_roles:
        direction_review["verdict"] = "uncertain_needs_human"
        reasons = direction_review.get("reasons", [])
        if not isinstance(reasons, list):
            reasons = []
        reasons.append("role_artifact_missing:" + ",".join(missing_roles))
        direction_review["reasons"] = reasons
        direction_review["escalate_to_human"] = True
        alerts = digest.get("alerts", [])
        if not isinstance(alerts, list):
            alerts = []
        alerts.append("role_artifact_missing")
        digest["alerts"] = alerts

    divergence = _record_and_validate_role_divergence(
        root,
        researcher_artifact_path,
        role_artifacts.get("arbiter", root / "backlog" / "role_outputs" / "arbiter_latest.md"),
    )
    if not bool(divergence.get("researcher_changed", False)) or not bool(divergence.get("arbiter_changed", False)):
        direction_review["verdict"] = "uncertain_needs_human"
        reasons = direction_review.get("reasons", [])
        if not isinstance(reasons, list):
            reasons = []
        if not bool(divergence.get("researcher_changed", False)):
            reasons.append("researcher_output_not_changed")
        if not bool(divergence.get("arbiter_changed", False)):
            reasons.append("arbiter_output_not_changed")
        direction_review["reasons"] = reasons
        direction_review["escalate_to_human"] = True
        alerts = digest.get("alerts", [])
        if not isinstance(alerts, list):
            alerts = []
        alerts.append("role_output_not_changed")
        digest["alerts"] = alerts

    if isinstance(research_quality, dict) and research_quality.get("degraded"):
        if direction_review.get("verdict") != "uncertain_needs_human":
            direction_review["verdict"] = "direction_adjust"
        reasons = direction_review.get("reasons", [])
        if not isinstance(reasons, list):
            reasons = []
        reasons.extend([f"research_quality_degraded:{item}" for item in research_quality.get("reasons", [])])
        direction_review["reasons"] = reasons
        alerts = digest.get("alerts", [])
        if not isinstance(alerts, list):
            alerts = []
        alerts.append("research_quality_degraded")
        digest["alerts"] = alerts

    role_scores = _score_role_outputs(signals, role_artifacts)
    role_evolution_path, role_evolution_state = _update_role_evolution(root, config, role_scores)

    team_actions = _build_team_actions_summary(
        signals,
        proposals,
        direction_review,
        recursive_state=recursive_state,
        researcher_artifact_path=str(researcher_artifact_path),
    )
    if isinstance(role_evolution_state.get("suggestions"), list) and role_evolution_state.get("suggestions"):
        team_actions["conductor"] += " | evolution=" + ",".join(role_evolution_state.get("suggestions", []))
    digest["team_actions"] = team_actions

    proposal_paths = write_proposals(root, proposals)

    deep_cycle_failures = _evaluate_deep_cycle(signals, config, len(proposal_paths))
    if deep_cycle_failures:
        direction_review["verdict"] = "uncertain_needs_human"
        reasons = direction_review.get("reasons", [])
        if not isinstance(reasons, list):
            reasons = []
        reasons.extend([f"deep_cycle_gate_failed:{item}" for item in deep_cycle_failures])
        direction_review["reasons"] = reasons
        direction_review["escalate_to_human"] = True

        alerts = digest.get("alerts", [])
        if not isinstance(alerts, list):
            alerts = []
        alerts.append("deep_cycle_gate_failed")
        digest["alerts"] = alerts

    role_artifacts = _write_role_artifacts(root, signals, proposals, direction_review, recursive_state, proposal_paths=proposal_paths)
    role_artifacts["researcher"] = researcher_artifact_path
    primary_proposal = _primary_proposal_summary(proposal_paths)
    digest["primary_proposal"] = primary_proposal
    team_actions = _build_team_actions_summary(
        signals,
        proposals,
        direction_review,
        recursive_state=recursive_state,
        researcher_artifact_path=str(researcher_artifact_path),
        primary_proposal=primary_proposal,
    )
    digest["team_actions"] = team_actions
    digest_path = write_daily_digest(root, digest)
    evolution_artifacts = _write_evolution_artifacts(root, signals, operator_result=operator_result)
    changed_paths = _safe_git_changed_paths(root)
    delivery_decision = _build_delivery_decision(signals, direction_review, changed_paths, config)

    return {
        "team_name": config.get("team_name", "mustard-claw"),
        "digest_path": str(digest_path),
        "proposal_paths": [str(path) for path in proposal_paths],
        "proposal_count": len(proposal_paths),
        "failure_patterns_path": str(failure_patterns_path),
        "top_gap_path": str(top_gap_path),
        "research_brief_path": str(research_brief_path),
        "researcher_artifact_path": str(researcher_artifact_path),
        "carm_gap_map_path": str(carm_gap_map_path),
        "alerts": digest.get("alerts", []),
        "direction_review": direction_review,
        "signals": signals,
        "team_actions": team_actions,
        "role_artifact_paths": {role: str(path) for role, path in role_artifacts.items()},
        "role_output_status": role_output_status,
        "role_scores": role_scores,
        "role_evolution_path": str(role_evolution_path),
        "role_evolution_suggestions": role_evolution_state.get("suggestions", []),
        "research_quality_path": str(research_quality_path),
        "research_quality": research_quality,
        "research_recovery_report_path": str(research_recovery_report_path),
        "research_recovery": research_recovery,
        "role_content_history_path": str(divergence.get("history_path", "")),
        "researcher_changed_vs_last": bool(divergence.get("researcher_changed", False)),
        "arbiter_changed_vs_last": bool(divergence.get("arbiter_changed", False)),
        "cleanup_stats": cleanup_stats,
        "pipeline_stats": pipeline_stats,
        "eval_stability": eval_stability,
        "operator_result": operator_result,
        "quality_stabilization_report_path": str(quality_artifacts.get("report_path", "")),
        "quality_stabilization_json_path": str(quality_artifacts.get("json_path", "")),
        "quality_stabilization": quality_artifacts.get("payload", {}),
        "quality_operator_result": quality_operator_result,
        "evolution_track": str(evolution_artifacts.get("track", "")),
        "evolution_summary": evolution_artifacts.get("summary", {}),
        "evolution_candidate_path": str(evolution_artifacts.get("candidate_path", "")),
        "evolution_lineage_path": str(evolution_artifacts.get("lineage_path", "")),
        "evolution_run_path": str(evolution_artifacts.get("run_path", "")),
        "delivery_decision": delivery_decision,
    }


def main() -> int:
    result = run_cycle()
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

