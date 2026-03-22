# Evaluator

负责对候选改动做统一验证。

## Standard Checks

- 单元测试
- `evaluate_pretraining`
- `evaluate_real_prompts`
- `run_control_cycle`
- `judge_control_rollout`

## Verdict

- `pass`
- `soft_pass`
- `fail`
- `needs_human_review`
