# Expand real-prompt regression coverage

- problem: 真实回归样本覆盖偏窄，难以持续发现创新能力退化。
- change_type: evaluation_or_dataset
- risk_level: low
- needs_human_approval: False
- proposed_change: 从近期高价值 episode 构建候选真实回归集，并合并到 real_prompt_eval 基准。
- rollback_plan: 仅修改评测集配置，可回退到上一版配置文件。
- evidence:
  - real_prompt_prompt_count=6
- evaluation_plan:
  - python -m scripts.build_real_prompt_candidates
  - python -m scripts.evaluate_real_prompts
