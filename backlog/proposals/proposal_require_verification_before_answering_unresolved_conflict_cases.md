# Proposal: Require verification before answering unresolved conflict cases

- problem: `real-conflict` 在工具命中为真的情况下仍未通过 hard eval，因为当前轨迹包含 `conflict_unresolved` 但缺少 `VERIFY`，说明现有指标的全绿掩盖了冲突任务的关键推理风险。
- evidence:
  - `artifacts/reasoning_pattern_codec_latest.json` 中 `hard_eval.pass_rate = 0.8333`。
  - `hard_eval.failed_case_ids = ["real-conflict"]`。
  - `real-conflict` 的 residual 为 `conflict_unresolved + missing_verify`。
  - `artifacts/current_best.json` 已降为 `status = needs_attention`，但 `real_prompt_match_rate` 仍为 `1.0`，证明工具命中不足以代表推理质量。
- from_failure_pattern: repeated_conflict_detection_gap
- from_top_gap: CARM hard logic pass rate
- change_type: runtime_control
- proposed_change: 在默认运行时接入前，先做候选实验：当 prompt 或中间表示显示 unresolved conflict，且尚未执行 `VERIFY` 时，不允许直接 `ANSWER`；候选策略应优先写入/保留冲突假设并触发验证步骤。
- expected_metric_delta:
  - `hard_eval_pass_rate`: 0.8333 -> 1.0 on current hard pack
  - `verify_when_residual_risky_rate`: should increase on conflict cases
  - `real_prompt_match_rate`: must stay >= 0.9
- risk_level: medium
- evaluation_plan:
  - 先在隔离候选路径中实现，不直接替换默认运行时。
  - 运行 `python -m scripts.evaluate_real_prompts`。
  - 运行 `python -m scripts.analyze_reasoning_patterns`。
  - 运行 `python -m scripts.current_best`。
  - 运行 `python -m unittest discover -s tests -p "test_runner.py" -v` 与相关 codec/current_best 测试。
- rollback_plan: 移除候选 verify gating 逻辑，恢复当前默认动作选择；保留 hard eval 与 codec 报告作为诊断工具。
- needs_human_approval: true
- relative_to_last_round: 上一步只建立了 VQR-inspired pattern/residual 诊断，本提案把诊断结果转为一个具体的候选行为修复目标。
- scenario_fit: 用户希望从自动研发切回人工指挥 Codex，优先修复能证明核心推理能力提升的硬逻辑缺口。

