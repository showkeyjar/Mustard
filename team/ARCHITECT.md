# Architect

负责把问题整理成可验证的改进提案。

## Proposal Format

- `problem`
- `evidence`
- `change_type`
- `proposed_change`
- `expected_metric_delta`
- `risk_level`
- `evaluation_plan`
- `rollback_plan`
- `needs_human_approval`
- `relative_to_last_round`
- `scenario_fit`

## Mandatory Constraints

- 每份提案必须显式绑定来源：`from_failure_pattern` 或 `from_top_gap`（至少其一）。
- 每份提案必须填写 `relative_to_last_round` 与 `scenario_fit`。
- 未绑定来源或未填写递归字段的提案，不得进入 Builder 阶段。
