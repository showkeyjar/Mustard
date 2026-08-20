# Review candidate runtime rollout

- problem: 当前存在候选运行时控制版本，需要继续观察或人工决策。
- from_failure_pattern: 
- from_top_gap: 
- change_type: runtime_control
- proposed_change: 保持候选版本隔离，继续运行控制周期并执行上线闸门判断。
- expected_metric_delta: 
- risk_level: high
- needs_human_approval: True
- relative_to_last_round: 
- scenario_fit: 
- architect_handoff: direct_execute_if_format_passes
- rollback_plan: python -m scripts.rollback_runtime_controls
- evidence:
  - candidate_version=20260819T021329064225Z
  - baseline_version=20260527T055114041414Z
- evaluation_plan:
  - python -m scripts.run_control_cycle
  - python -m scripts.judge_control_rollout
