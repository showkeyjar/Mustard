# Research Artifact (Actionable)

## 1) Meta
- round_id: auto-20260428T131537Z
- date: 2026-04-28
- owner: researcher
- from_top_gap: attention_verification_handoff_gap
- from_failure_pattern: sampling_blind_spot
- relative_to_last_round: upgraded from generic coverage template to current top-gap / failure-pattern aligned research input
- scenario_fit: 真实复杂任务里隐藏弱点未被触发的场景。

## 2) New weakness discovered this round
- weakness_summary: pretrained mismatch cases surfaced under sampling_blind_spot
- weakness_cluster: sampling_blind_spot
- why_it_matters_now: 专项压力已经落到 sampling_blind_spot，当前需要判断这是稳定可复现的新弱点还是偶发样本噪声。
- why_previous_rounds_missed_it: previous researcher input was pinned to generic coverage logic instead of the active specialized cluster

## 3) Hypothesis（可证伪）
- hypothesis: tightening prompts around sampling_blind_spot will either expose a reproducible mismatch cluster or confirm robustness under this boundary.
- falsifiable_condition: targeted prompts increase, but no new mismatch cluster or new failure pattern is formed
- expected_gain: 在 sampling_blind_spot 专项场景下形成可复现的新 mismatch 或确认专项鲁棒性。
- risk: low-information candidates may still crowd out the prompts most likely to reveal hidden weaknesses

## 4) Evidence chain
- representative_case_1: pretrained_match_rate=0.9000, baseline_match_rate=0.9000, delta=+0.0000
- representative_case_2: mismatch_case_count=2
- representative_case_3: stagnation_rounds=0; frontier_observation_count=3
- evidence_quality_note: 当前已有 mismatch，但还需要更多同类专项样本验证 sampling_blind_spot 是否是稳定簇。
- blind_spot_if_no_failure_case: none
- candidate_pipeline_snapshot:
  - total_candidates: 9
  - filtered_candidates: 2
  - dropped_candidates: 7

## Concrete mismatch cases
- real-mixed: expected=calculator actual=code_executor logic_skill=tool_selection
- repair-comparison-005: expected=search actual=bigmodel_proxy logic_skill=comparison

## 5) Minimal next experiment（可执行）
- command_1: python -m scripts.evaluate_real_prompts
- command_2: python -m scripts.build_real_prompt_candidates
- metric_threshold: focused prompts around sampling_blind_spot increase while pretrained_match_rate stays >= 0.90
- pass_criteria: specialized prompts either expose a reproducible weakness cluster or prove sampling_blind_spot remains stable under stronger pressure
- fail_criteria: 围绕 sampling_blind_spot 的专项 prompts 仍无法形成可复现的新 pattern。

## 6) Landing Candidate（可直接进 Architect）
- proposed_change: 补充更高信息量的专项 prompts，并验证是否出现新的 mismatch cluster。
- change_scope: configs/real_prompt_eval.json + targeted prompt pack + research reporting
- rollback_plan: revert targeted prompts and restore previous prompt pack if evidence quality worsens
- handoff_to_architect: yes

## 7) Decision label
- tag: 待观察
- reason: current match is strong, but the system still lacks sufficient evidence that sampling_blind_spot has been truly broken open
