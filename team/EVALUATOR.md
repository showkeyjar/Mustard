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

## Mandatory Output（每轮）

- 验证结果必须给出：执行命令、关键日志、结论理由
- `soft_pass` 必须附带剩余风险与下一轮补测项
- 若无可评测改动，明确写 `no_candidate_change`，禁止空白结论
