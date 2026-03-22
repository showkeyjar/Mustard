from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path


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

    comparison = control_metrics.get("comparison", {})
    if not isinstance(comparison, dict):
        comparison = {}

    return {
        "episodes": _count_jsonl(root / "data/experience/episodes.jsonl"),
        "reviews": _count_jsonl(root / "data/review/reviews.jsonl"),
        "bridge_feedback": _count_jsonl(root / "data/desktop/bridge_feedback.jsonl"),
        "latest_train_run_id": train_report.get("run_id", ""),
        "dataset_sample_count": train_report.get("dataset", {}).get("sample_count", 0) if isinstance(train_report.get("dataset", {}), dict) else 0,
        "pretrain_eval": pretrain_eval,
        "real_prompt_eval": real_prompt_eval,
        "control_state": control_state,
        "control_comparison": comparison,
    }


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

    return {
        "timestamp_utc": utc_now().isoformat(),
        "team_name": config.get("team_name", "mustard-claw"),
        "focus_areas": focus_areas,
        "signals": signals,
        "alerts": alerts,
    }


def build_proposals(digest: dict[str, object], config: dict[str, object]) -> list[dict[str, object]]:
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

    max_new = config.get("heartbeat", {}).get("max_new_proposals_per_cycle", 3) if isinstance(config.get("heartbeat", {}), dict) else 3
    return proposals[: int(max_new)]


def write_daily_digest(root: Path, digest: dict[str, object]) -> Path:
    stamp = datetime.fromisoformat(str(digest["timestamp_utc"]))
    output_path = root / "memory" / "daily" / f"{stamp.strftime('%Y-%m-%d')}.md"
    lines = [
        f"# {stamp.strftime('%Y-%m-%d')} Daily Digest",
        "",
        f"- team: {digest.get('team_name', 'mustard-claw')}",
        f"- alerts: {', '.join(digest.get('alerts', [])) or 'none'}",
    ]

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

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output_path


def write_proposals(root: Path, proposals: list[dict[str, object]]) -> list[Path]:
    written: list[Path] = []
    for proposal in proposals:
        title = str(proposal.get("title", "proposal"))
        filename = f"{utc_now().strftime('%Y%m%dT%H%M%SZ')}_{_slug(title)}.md"
        path = root / "backlog" / "proposals" / filename
        lines = [
            f"# {title}",
            "",
            f"- problem: {proposal.get('problem', '')}",
            f"- change_type: {proposal.get('change_type', '')}",
            f"- risk_level: {proposal.get('risk_level', '')}",
            f"- needs_human_approval: {proposal.get('needs_human_approval', False)}",
            f"- proposed_change: {proposal.get('proposed_change', '')}",
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


def run_cycle(root: Path = Path("."), config_path: Path = DEFAULT_CONFIG_PATH) -> dict[str, object]:
    bootstrap_workspace(root)
    config = load_team_config(root / config_path if not config_path.is_absolute() else config_path)
    signals = collect_signals(root)
    digest = build_daily_digest(signals, config)
    proposals = build_proposals(digest, config)
    digest_path = write_daily_digest(root, digest)
    proposal_paths = write_proposals(root, proposals)
    return {
        "team_name": config.get("team_name", "mustard-claw"),
        "digest_path": str(digest_path),
        "proposal_paths": [str(path) for path in proposal_paths],
        "proposal_count": len(proposal_paths),
        "alerts": digest.get("alerts", []),
    }


def main() -> int:
    result = run_cycle()
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
