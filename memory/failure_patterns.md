# Failure Patterns

记录稳定复现的失败模式、误判模式、采样盲区和回滚原因。

## Top patterns this round

### forced_action_conflict_resolution_gap
- pattern_id: forced_action_conflict_resolution_gap
- title: forced_action_conflict_resolution_gap
- frequency: current_round
- impact: 高压 conflict_detection 样本已能让 pretrained 失败，说明在被迫给动作建议时仍存在新的冲突化解弱点。
- repro_hint: 复用 quality focus eval 中 forced conflict rows，继续扩展 action-forcing / missing-evidence / authority-conflict 组合样本。
- owner_role: failure_miner
- representative_cases: {"logic_skill": "conflict_detection", "count": 2, "sample_ids": ["quality-real-conflict-03", "quality-repair-conflict_detection-017-03"], "expected_tool": ["search"]}
- likely_root_cause: current research loop still undersamples high-information weaknesses or lacks external feedback pressure
- recommended_fix_direction: tighten sampling + convert meta-failure into explicit repair tasks
- status: open

## Sampling insufficiency / blind spots

- blind_spot: high-information real prompts still underrepresented even when aggregate match stays high
- why_current_sample_is_insufficient: prompt_count=20, mismatch_count=2, bridge_feedback=2, frontier_observation_count=3
- next_sampling_action: add >=4 high-information prompts and collect non-zero frontier / bridge evidence
- owner_role: researcher

## Research degradation patterns

- repeated_low_information_candidates
- no_new_failure_pattern_across_rounds
- coverage_only_top_gap_repetition
- frontier_zero_signal_persistence
- bridge_zero_feedback_persistence
- report_style_research_without_diagnostic_novelty

## Repair leads for Architect

- 将 sampling_blind_spot / frontier_research_blindspot / no_tool_feedback_loop 直接转成下一轮 Architect 提案输入。
