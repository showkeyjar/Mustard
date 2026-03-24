# Scale pretraining corpus from real episodes

- problem: 当前预训练样本规模偏小，可能限制两段式学习上限。
- change_type: evaluation_or_dataset
- risk_level: low
- needs_human_approval: False
- proposed_change: 执行自动训练流水线并将高价值真实 episode 回流到预训练集，提升创新链路覆盖。
- rollback_plan: 仅新增训练产物与评测报告，不修改默认运行时策略。
- evidence:
  - dataset_sample_count=186
- evaluation_plan:
  - python -m scripts.auto_train
  - python -m scripts.evaluate_pretraining
  - python -m scripts.evaluate_real_prompts
