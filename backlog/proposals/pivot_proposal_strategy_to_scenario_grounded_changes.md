# Pivot proposal strategy to scenario-grounded changes

- problem: 连续多轮核心信号无变化，提案可能进入机械重复。
- change_type: process_improvement
- risk_level: low
- needs_human_approval: False
- proposed_change: 下一轮提案强制绑定真实使用场景（工作中断、误触发、高频命令），每条提案必须说明与上一轮差异点。
- rollback_plan: 仅流程约束改动，删除新增字段即可回退。
- evidence:
  - stagnation_rounds=2
- evaluation_plan:
  - 在提案模板新增 previous_round_diff 字段
  - 抽样检查最近 3 条提案是否场景化且不重复
