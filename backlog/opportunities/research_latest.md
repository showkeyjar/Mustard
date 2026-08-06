# Research Artifact (Actionable)

## 1) Meta
- round_id: manual-20260806T150000Z
- date: 2026-08-06
- owner: researcher (人工接续，修正被污染的 evidence 链)
- from_top_gap: eval_coverage_too_low
- from_failure_pattern: learning_focus_evidence_tool_routing_gap
- relative_to_last_round: 修复 real_prompt_eval_latest.json 被 recovery-variant 污染的问题后，重新基于 63 条真实评测证据整理研究输入。
- scenario_fit: 需要先查资料、判断证据、再下结论的真实求证型任务。

## 2) 上一轮证据污染说明（重要）
- 上一轮（auto-20260806T063804Z）的 mismatch 证据来自被 recovery-variant 覆盖的 `data/eval/real_prompt_eval_latest.json`（4 条探针、match_rate=0.0），并非真实评测结果。
- 根因：`scripts/team_conductor.py::_evaluate_recovery_variants` 以临时 config 调用 `python -m scripts.evaluate_real_prompts <temp_config>`，而 `scripts/evaluate_real_prompts.py::main()` 无条件写 latest 文件。
- 修复：main() 仅在无参数（官方 configs/real_prompt_eval.json）时写 latest；探针结果由各自调用方持久化。已随 2c7f692 提交。
- 同时修复了 policy.py Override 0d 缺 evidence_judgment 守卫导致的 evidence 任务误路由（learning_focus 004/005/006 等），以及 search_tool.py 被墙 DuckDuckGo connect 阻塞评测的问题。

## 3) 修正后的真实弱点（基于 63 条完整评测）
- weakness_summary: 63 条 real_prompt 评测 pretrained_match_rate=0.7302；17 条不匹配中，10 条被 multi_intent 拆分规则截获（期望 search），3 条被 synthesis 规则路由到 bigmodel_proxy（期望 search），2 条 consult 任务被过度路由到 search（期望 bigmodel_proxy），1 条 code_executor，1 条 calculator。
- weakness_cluster_1: multi_intent_over_capture —— “先查一下 X 还是 Y / 比较一下 A 和 B” 类提示词被拆分为 multi_intent 而不是 search-first 求证（constraint-plan-*, guard-ab-test-001, guard-rw-split-001, guard-crud-api-001, term-judgment-003/005, repair-missing-evidence-decision-001）。
- weakness_cluster_2: synthesis_over_evidence —— 缺证据的冲突/决策任务因包含“给结论/建议/总结”等写作信号被 Override 0b 路由到 bigmodel_proxy，而不是先 search（stress-conflict-missing-evidence-001/003, stress-conflict-authority-002, real-git-triage）。
- weakness_cluster_3: consult_over_synthesis —— 用户明确说“信息已足够/请直接给结论”时仍被路由到 search（term-judgment-001/004，期望 bigmodel_proxy）。
- why_previous_rounds_missed_it: latest 快照被 recovery 探针覆盖（4 条 0.0），团队基于污染数据误判为 sampling_blind_spot / 覆盖率问题。

## 4) Evidence chain
- representative_case_1: real_prompt_pretrained_match_rate=0.7302, baseline_match_rate=0.7302, prompt_count=63（修复后）
- representative_case_2: learning_focus_pretrained_match_rate=1.0（从 0.5714 提升，evidence_judgment 误路由已修复）
- representative_case_3: 修复前 A/B 对比旧 policy 匹配率 0.6984 → 修复后 0.7302，无回归
- evidence_quality_note: 数据来自官方 63 条评测（configs/real_prompt_eval.json），不含探针污染。
- candidate_pipeline_snapshot:
  - total_candidates: 待下一轮更新
  - filtered_candidates: 待下一轮更新

## Concrete mismatch cases（修复后 63 条评测，17 条）
- multi_intent_over_capture（期望 search → multi_intent）: repair-missing-evidence-decision-001, term-judgment-003, term-judgment-005, constraint-plan-001, constraint-plan-002, constraint-plan-003, guard-ab-test-001, guard-rw-split-001, guard-crud-api-001
- synthesis_over_evidence（期望 search → bigmodel_proxy）: stress-conflict-missing-evidence-001, stress-conflict-missing-evidence-003, stress-conflict-authority-002, real-git-triage
- consult_over_synthesis（期望 bigmodel_proxy → search）: term-judgment-001, term-judgment-004
- other: real-release-plan（期望 search → code_executor）, repair-missing-evidence-decision-003（期望 search → calculator）

## 5) Minimal next experiment（可执行）
- command_1: python -m scripts.evaluate_real_prompts（官方 63 条，验证 latest 恢复）
- command_2: 针对 multi_intent_over_capture 增加 evidence/求证场景下“A 还是 B”的定向样本并评估 multi_intent 规则是否需要 evidence 守卫
- command_3: 针对 synthesis_over_evidence 评估 Override 0b 是否需要“缺证据/冲突”守卫（与 0d 同族）
- metric_threshold: real_prompt_pretrained_match_rate 从 0.7302 提升且不引入 bigmodel_proxy 期望样本回归
- pass_criteria: 17 条不匹配中至少 8 条被正确路由且无新回归
- fail_criteria: 修复 multi_intent/synthesis 后 bigmodel_proxy 期望样本（term-judgment 类）被误路由

## 6) Landing Candidate（可直接进 Architect）
- proposed_change: 为 multi_intent 拆分规则与 Override 0b synthesis 规则补充 evidence/conflict 守卫（与 Override 0d 同族），并扩充 term-judgment 类合成意图样本。
- change_scope: carm/policy.py + configs/real_prompt_eval.json + 定向评测
- risk: low（与已落地第四轮修复同一家族，先离线验证）
