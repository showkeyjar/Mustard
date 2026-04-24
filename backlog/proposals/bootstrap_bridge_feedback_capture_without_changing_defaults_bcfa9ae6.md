# Bootstrap bridge feedback capture without changing defaults

- problem: bridge feedback 持续为 0，系统无法得到真实用户纠偏信号。
- from_failure_pattern: no_tool_feedback_loop
- from_top_gap: research_quality_degraded
- change_type: desktop_behavior
- proposed_change: 先补反馈采样与整理入口，只记录 useful/misread，不修改桌面默认行为。
- expected_metric_delta: bridge_feedback > 0，且 failure miner 能获得真实纠偏样本
- risk_level: medium
- needs_human_approval: True
- relative_to_last_round: 从抽象提桥梁闭环，改为先把反馈入口打通。
- scenario_fit: 用户在真实桌面协作里纠偏系统误读/误触发的场景。
- architect_handoff: guardian reviews scope, then failure_miner defines feedback capture schema before any rollout
- rollback_plan: 仅停止采样与整理，不改动默认桌面策略。
- evidence:
  - failure_pattern=no_tool_feedback_loop
  - bridge_feedback=2
- evaluation_plan:
  - python -m scripts.desktop_agent_control snapshot
  - python -m scripts.desktop_bridge_chat
