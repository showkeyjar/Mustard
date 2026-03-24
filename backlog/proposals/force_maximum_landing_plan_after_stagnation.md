# Force maximum landing plan after stagnation

- problem: 连续多轮无显著变化，需要从探索转为最大可落地。
- change_type: process_improvement
- risk_level: low
- needs_human_approval: False
- proposed_change: 暂停新增探索项 1 轮，只保留可在 24 小时内验证落地的动作：扩充 real prompts、修复 Top1 failure pattern、补齐训练前后对比。
- rollback_plan: 恢复常规提案配额并移除强制落地限制。
- evidence:
  - stagnation_rounds=3
- evaluation_plan:
  - 执行 python -m scripts.evaluate_real_prompts
  - 更新 memory/failure_patterns.md Top1 修复状态
  - 补充一份训练前后指标对比
