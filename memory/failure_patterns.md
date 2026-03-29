# Failure Patterns

记录稳定复现的失败模式、误判模式、采样盲区和回滚原因。

## Top patterns this round

### sampling_blind_spot
- pattern_id: sampling_blind_spot
- title: sampling_blind_spot
- frequency: current_round
- impact: 样本表面全绿，但仍不足以证明隐藏弱点已被发现。
- repro_hint: 扩充高信息量 real prompts 后重跑 evaluate_real_prompts，看是否暴露新弱点簇
- owner_role: researcher
- representative_cases: {"prompt_count": 20, "mismatch_count": 0}
- likely_root_cause: current research loop still undersamples high-information weaknesses or lacks external feedback pressure
- recommended_fix_direction: tighten sampling + convert meta-failure into explicit repair tasks
- status: open

### repeated_conflict_detection_gap
- pattern_id: repeated_conflict_detection_gap
- title: repeated_conflict_detection_gap
- frequency: current_round
- impact: 高信息恢复采样持续指向同一类薄弱模式，说明 blind spot 已开始具体化。
- repro_hint: 复用 recovery variants / focus eval 继续压测同类 logic skill 与 mutation
- owner_role: failure_miner
- representative_cases: {"logic_skill": "conflict_detection", "count": 2, "avg_source_score": 6.0, "top_mutations": ["contradictory_authority", "missing_evidence"]}
- likely_root_cause: current research loop still undersamples high-information weaknesses or lacks external feedback pressure
- recommended_fix_direction: tighten sampling + convert meta-failure into explicit repair tasks
- status: open

### comparison_under_conflicting_sources
- pattern_id: comparison_under_conflicting_sources
- title: comparison_under_conflicting_sources
- frequency: current_round
- impact: 高信息恢复采样持续指向同一类薄弱模式，说明 blind spot 已开始具体化。
- repro_hint: 复用 recovery variants / focus eval 继续压测同类 logic skill 与 mutation
- owner_role: failure_miner
- representative_cases: {"logic_skill": "comparison", "count": 1, "avg_source_score": 6.0, "top_mutations": ["conflicting_sources"]}
- likely_root_cause: current research loop still undersamples high-information weaknesses or lacks external feedback pressure
- recommended_fix_direction: tighten sampling + convert meta-failure into explicit repair tasks
- status: open

### tool_boundary_sampling_gap
- pattern_id: tool_boundary_sampling_gap
- title: tool_boundary_sampling_gap
- frequency: current_round
- impact: 高信息恢复采样持续指向同一类薄弱模式，说明 blind spot 已开始具体化。
- repro_hint: 复用 recovery variants / focus eval 继续压测同类 logic skill 与 mutation
- owner_role: failure_miner
- representative_cases: {"logic_skill": "tool_selection", "count": 1, "avg_source_score": 3.0, "top_mutations": ["calculator_vs_search"]}
- likely_root_cause: current research loop still undersamples high-information weaknesses or lacks external feedback pressure
- recommended_fix_direction: tighten sampling + convert meta-failure into explicit repair tasks
- status: open

## Sampling insufficiency / blind spots

- blind_spot: high-information real prompts still underrepresented even when aggregate match stays high
- why_current_sample_is_insufficient: prompt_count=20, mismatch_count=0, bridge_feedback=12, frontier_observation_count=3
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
