# Failure Patterns

记录稳定复现的失败模式、误判模式、采样盲区和回滚原因。

## Top patterns this round

### attention_verification_handoff_gap
- pattern_id: attention_verification_handoff_gap
- title: attention_verification_handoff_gap
- frequency: current_round
- impact: 注意力流在 conflict/tool-boundary 残差尚未消解时过早释放，或没有自然转入 verification。
- repro_hint: 运行 attention_flow / attention_training_views 评估，并检查冲突残差是否在 release 前进入 VERIFY。
- owner_role: architect
- representative_cases: {"premature_release_count": 1, "conflict_to_verification_rate": 0.25, "view_count": 106}
- likely_root_cause: current research loop still undersamples high-information weaknesses or lacks external feedback pressure
- recommended_fix_direction: tighten sampling + convert meta-failure into explicit repair tasks
- status: open

### learning_focus_evidence_tool_routing_gap
- pattern_id: learning_focus_evidence_tool_routing_gap
- title: learning_focus_evidence_tool_routing_gap
- frequency: current_round
- impact: 系统在吸收公开设计思想与研究任务时，evidence_judgment 仍可能误路由到 calculator，说明新知识还没稳定映射到检索型证据流程。
- repro_hint: 运行 learning_focus_eval，重点检查 evidence_judgment 类任务是否仍把 search 型求证误判成 calculator。
- owner_role: trainer
- representative_cases: {"count": 3, "sample_ids": ["learning-focus-004", "learning-focus-005", "learning-focus-006"], "actual_tools": ["calculator"]}
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
- representative_cases: {"logic_skill": "comparison", "count": 1, "avg_source_score": 3.0, "top_mutations": ["ranking_flip"]}
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
- representative_cases: {"logic_skill": "conflict_detection", "count": 3, "avg_source_score": 1.0, "top_mutations": ["missing_evidence", "conflicting_sources", "contradictory_authority"]}
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
