from __future__ import annotations

import json
import hashlib
import re
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
    Path("data/research"),
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
        "frontier_observation_count": _count_jsonl(root / "data/research/frontier_observations.jsonl"),
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

    verdict = "direction_correct" if not reasons else "uncertain_needs_human"
    return {
        "owner": str(decision_policy.get("owner", "arbiter")),
        "verdict": verdict,
        "reasons": reasons,
        "escalate_to_human": verdict == "uncertain_needs_human" and bool(decision_policy.get("escalate_on_uncertain_direction", True)),
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

    if not proposals:
        proposals.extend(_build_proactive_proposals(signals, gated_types))

    if recursive_mode == "pivot":
        proposals.insert(
            0,
            {
                "title": "Pivot proposal strategy to scenario-grounded changes",
                "problem": "连续多轮核心信号无变化，提案可能进入机械重复。",
                "evidence": [f"stagnation_rounds={stagnation_rounds}"],
                "change_type": "process_improvement",
                "proposed_change": "下一轮提案强制绑定真实使用场景（工作中断、误触发、高频命令），每条提案必须说明与上一轮差异点。",
                "risk_level": "low",
                "evaluation_plan": [
                    "在提案模板新增 previous_round_diff 字段",
                    "抽样检查最近 3 条提案是否场景化且不重复",
                ],
                "rollback_plan": "仅流程约束改动，删除新增字段即可回退。",
                "needs_human_approval": "process_improvement" in gated_types,
            },
        )

    if recursive_mode == "max_landing":
        proposals.insert(
            0,
            {
                "title": "Force maximum landing plan after stagnation",
                "problem": "连续多轮无显著变化，需要从探索转为最大可落地。",
                "evidence": [f"stagnation_rounds={stagnation_rounds}"],
                "change_type": "process_improvement",
                "proposed_change": "暂停新增探索项 1 轮，只保留可在 24 小时内验证落地的动作：扩充 real prompts、修复 Top1 failure pattern、补齐训练前后对比。",
                "risk_level": "low",
                "evaluation_plan": [
                    "执行 python -m scripts.evaluate_real_prompts",
                    "更新 memory/failure_patterns.md Top1 修复状态",
                    "补充一份训练前后指标对比",
                ],
                "rollback_plan": "恢复常规提案配额并移除强制落地限制。",
                "needs_human_approval": "process_improvement" in gated_types,
            },
        )

    max_new = config.get("heartbeat", {}).get("max_new_proposals_per_cycle", 3) if isinstance(config.get("heartbeat", {}), dict) else 3
    return proposals[: int(max_new)]


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


def _build_team_actions_summary(
    signals: dict[str, object],
    proposals: list[dict[str, object]],
    direction_review: dict[str, object],
    recursive_state: dict[str, object] | None = None,
    researcher_artifact_path: str = "",
) -> dict[str, str]:
    needs_human = sum(1 for proposal in proposals if bool(proposal.get("needs_human_approval", False)))
    proposal_brief = "; ".join(
        f"{proposal.get('title', '')}: {proposal.get('proposed_change', '')}" for proposal in proposals[:3]
    ) or "本轮无新增提案"

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
        "architect": f"将问题转为可执行提案（首条：{str(proposals[0].get('title', '无')) if proposals else '无'}）",
        "trainer": "执行训练/蒸馏流水线并产出可对比训练报告",
        "evaluator": "执行验证链路：unittest + evaluate_pretraining + evaluate_real_prompts + run_control_cycle",
        "guardian": f"风险审查完成，需 Human Gate 的提案={needs_human}",
        "arbiter": f"方向裁决={direction_review.get('verdict', 'direction_correct')} reasons={','.join(direction_review.get('reasons', [])) or 'none'}",
        "researcher": researcher_action + "（模板: team/RESEARCHER_OUTPUT_TEMPLATE.md）",
    }


def _write_failure_patterns(root: Path, signals: dict[str, object]) -> Path:
    incidents_path = root / "backlog" / "incidents" / "auto_failure_patterns.json"

    patterns: list[dict[str, object]] = []
    real_prompt_eval = signals.get("real_prompt_eval", {})
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
                    }
                )

    frontier_observation_count = int(signals.get("frontier_observation_count", 0) or 0)
    if frontier_observation_count < 2:
        patterns.append(
            {
                "id": "frontier_research_blindspot",
                "severity": "high",
                "evidence": {"frontier_observation_count": frontier_observation_count, "target_min": 2},
                "impact": "缺少前沿对标，容易重复走弯路。",
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
            }
        )

    payload = {
        "pattern_count": len(patterns),
        "patterns": patterns,
    }
    _write_if_changed(incidents_path, json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
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


def _write_top_gap_action_card(root: Path, signals: dict[str, object]) -> Path:
    top_gap_path = root / "backlog" / "opportunities" / "auto_top_gap.md"

    prompt_count = 0
    real_prompt_eval = signals.get("real_prompt_eval", {})
    if isinstance(real_prompt_eval, dict):
        summary = real_prompt_eval.get("summary", {})
        if isinstance(summary, dict):
            prompt_count = int(summary.get("prompt_count", 0) or 0)

    lines = [
        "# Top Gap Action Card",
        "",
        "- gap_id: eval_coverage_too_low",
        "- problem: 真实工具调用场景回归样本覆盖不足，当前无法证明可替代性提升。",
        f"- current: prompt_count={prompt_count}",
        "- target: prompt_count>=20（优先本地工具调用场景）",
        "- owner: benchmark_owner + failure_miner + trainer",
        "- action_plan:",
        "  - 1) 运行 python -m scripts.build_real_prompt_candidates 生成候选集",
        "  - 2) 合并候选集到 configs/real_prompt_eval.json（去重后保留高价值工具调用样本）",
        "  - 3) 运行 python -m scripts.evaluate_real_prompts 并记录前后对比",
        "- acceptance:",
        "  - real_prompt_eval.summary.prompt_count >= 20",
        "  - 产出一份前后指标对比摘要（match_rate / avg_steps）",
        "- rollback: 若样本质量下降，回退 real_prompt_eval.json 到上一版并重新评测",
    ]

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
    path = output_dir / f"research_{utc_now().strftime('%Y%m%dT%H%M%SZ')}.md"

    real_prompt_eval = signals.get("real_prompt_eval", {})
    rp_count = 0
    rp_match = 0.0
    if isinstance(real_prompt_eval, dict):
        summary = real_prompt_eval.get("summary", {})
        if isinstance(summary, dict):
            rp_count = int(summary.get("prompt_count", 0) or 0)
            rp_match = float(summary.get("pretrained_match_rate", 0.0) or 0.0)

    frontier_obs = int(signals.get("frontier_observation_count", 0) or 0)
    stagnation_rounds = int(recursive_state.get("stagnation_rounds", 0) or 0)

    lines = [
        "# Research Artifact",
        "",
        f"- from_top_gap: eval_coverage_too_low",
        f"- from_failure_pattern: frontier_research_blindspot",
        f"- relative_to_last_round: 强制从‘状态描述’改为‘带可证伪条件的实验卡’",
        f"- scenario_fit: 日常工作流中工具调用真实场景覆盖不足导致改进不可验证",
        "",
        "## Hypothesis",
        f"- 在 real_prompt_count 从 {rp_count} 扩充到 >=20 前，方向判断可靠性不足；优先扩充样本可提升改进决策质量。",
        "- falsifiable_condition: 若扩充后 match_rate 仍无改善且失败模式分布不变，则该路径失败。",
        "",
        "## Evidence",
        f"- real_prompt_count={rp_count}",
        f"- real_prompt_match_rate={rp_match:.4f}",
        f"- frontier_observation_count={frontier_obs}",
        f"- stagnation_rounds={stagnation_rounds}",
        "",
        "## Experiment Plan",
        "- python -m scripts.build_real_prompt_candidates",
        "- python -m scripts.evaluate_real_prompts",
        "- pass_criteria: prompt_count>=20 且可观察到失败模式重排",
        "- fail_criteria: prompt_count增长但关键指标/失败模式无变化",
        "",
        "## Landing",
        "- proposed_change: 生成并合并真实场景回归候选集，优先覆盖高频工具调用链路",
        "- rollback_plan: 回退 real_prompt_eval 配置到上一版",
    ]

    _write_if_changed(path, "\n".join(lines) + "\n")
    return path


def _write_role_artifacts(
    root: Path,
    signals: dict[str, object],
    proposals: list[dict[str, object]],
    direction_review: dict[str, object],
    recursive_state: dict[str, object],
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
    _write_if_changed(
        benchmark_path,
        "\n".join(
            [
                "# Benchmark Owner Output",
                "",
                "- top_gap: eval_coverage_too_low",
                f"- real_prompt_count: {rp_count}",
                f"- real_prompt_match_rate: {rp_match:.4f}",
                "- target_real_prompt_count: 20",
                f"- gap: {max(0, 20 - rp_count)}",
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
    _write_if_changed(
        architect_path,
        "\n".join(
            [
                "# Architect Output",
                "",
                f"- proposal_count_this_round: {len(proposals)}",
                "- note: no new proposal" if not proposals else "- note: proposals generated",
            ]
        )
        + "\n",
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


def run_cycle(root: Path = Path("."), config_path: Path = DEFAULT_CONFIG_PATH) -> dict[str, object]:
    bootstrap_workspace(root)
    config = load_team_config(root / config_path if not config_path.is_absolute() else config_path)
    signals = collect_signals(root)
    recursive_state = _update_recursive_state(root, signals, config)
    digest = build_daily_digest(signals, config)
    digest["recursive_state"] = recursive_state

    failure_patterns_path = _write_failure_patterns(root, signals)
    top_gap_path = _write_top_gap_action_card(root, signals)
    carm_gap_map_path = _write_carm_gap_map(root, signals)
    research_brief_path = _write_research_brief(root, signals, config)
    researcher_artifact_path = _run_researcher(root, signals, recursive_state)

    proposals = build_proposals(digest, config, recursive_state=recursive_state)
    existing_titles = _read_existing_proposal_titles(root)
    proposals = [proposal for proposal in proposals if str(proposal.get("title", "")).strip() not in existing_titles]

    direction_review = digest.get("direction_review", {})
    if not isinstance(direction_review, dict):
        direction_review = {}

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

    digest_path = write_daily_digest(root, digest)
    proposal_paths = write_proposals(root, proposals)

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
        "team_actions": team_actions,
        "role_artifact_paths": {role: str(path) for role, path in role_artifacts.items()},
        "role_output_status": role_output_status,
        "role_scores": role_scores,
        "role_evolution_path": str(role_evolution_path),
        "role_evolution_suggestions": role_evolution_state.get("suggestions", []),
        "role_content_history_path": str(divergence.get("history_path", "")),
        "researcher_changed_vs_last": bool(divergence.get("researcher_changed", False)),
        "arbiter_changed_vs_last": bool(divergence.get("arbiter_changed", False)),
    }


def main() -> int:
    result = run_cycle()
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
