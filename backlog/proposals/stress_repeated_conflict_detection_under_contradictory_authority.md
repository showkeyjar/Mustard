# Stress repeated conflict detection under contradictory authority

- problem: 高信息恢复采样连续指向 conflict_detection 弱点，说明该簇已足够具体，不能继续只按总括 blind spot 处理。
- from_failure_pattern: repeated_conflict_detection_gap
- from_top_gap: blind_spot_not_broken
- change_type: evaluation_or_dataset
- proposed_change: 围绕 contradictory_authority / missing_evidence 两类 mutation 新增定向 prompts，并把 conflict_detection 作为专项压测包维护。
- expected_metric_delta: 要么暴露新的 conflict_detection mismatch cluster，要么证明该专项簇在更强压力下仍稳定通过。
- risk_level: low
- needs_human_approval: False
- relative_to_last_round: 从总括 blind spot 施压升级为 conflict_detection 专项压测。
- scenario_fit: 多来源冲突、权威冲突、证据缺失下的搜索与判断场景。
- architect_handoff: researcher + benchmark_owner design conflict_detection stress pack
- rollback_plan: 若新增专项 prompts 只带来噪声，则回退 conflict_detection 专项样本包。
- evidence:
  - failure_pattern=repeated_conflict_detection_gap
  - stagnation_rounds=0
- evaluation_plan:
  - python -m scripts.build_real_prompt_candidates
  - python -m scripts.evaluate_real_prompts
