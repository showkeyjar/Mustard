# Researcher Output Template (Value Gate)

> 用途：Researcher 每轮必须按此模板提交；缺字段即视为无效产出。
>
> 禁止只写“指标汇总/结果摘要”。若没有新增弱点，也必须明确写出采样盲区、证据不足点和下一步补证动作，否则视为未通过 Value Gate。

## 1) Meta

- round_id:
- date:
- owner:
- from_top_gap:
- from_failure_pattern:
- relative_to_last_round:
- scenario_fit:

## 2) New weakness discovered this round

- weakness_summary:
- weakness_cluster:  （若无明确模式，写 none，并说明为什么仍值得跟踪）
- why_it_matters_now:
- why_previous_rounds_missed_it:

## 3) Hypothesis（可证伪）

- hypothesis:
- falsifiable_condition: （什么结果出现时判定假设失败）
- expected_gain:
- risk:

## 4) Evidence chain

- representative_case_1:
- representative_case_2:
- representative_case_3:
- evidence_quality_note:
- blind_spot_if_no_failure_case: （若当前没有失败样本，必须写采样不足/覆盖盲区）

## 5) Minimal next experiment（可执行）

- command_1:
- command_2:
- metric_threshold:
- pass_criteria:
- fail_criteria:

## 6) Landing Candidate（可直接进 Architect）

- proposed_change:
- change_scope:
- rollback_plan:
- handoff_to_architect: yes/no

## 7) Decision label

- tag: 可借鉴 / 不建议跟进 / 待观察
- reason:

---

## Output validity rules

以下任一情况成立，则本轮 Researcher 产出无效：

- 只重复 `match_rate / avg_steps / prompt_count`，但没有给出新增弱点或采样盲区诊断
- 没有可执行实验命令
- 没有明确失败条件（只能写成功路径）
- 没有 `relative_to_last_round`
- 没有 `scenario_fit`

## Value Gate Scorecard（总分 10）

- 绑定 Top Gap / Failure Pattern（0-2）
- 可证伪性（0-2）
- 证据质量（0-2）
- 可落地性（0-2）
- 与上一轮差异 + 场景拟合（0-2）

### 判定

- 8-10: 有价值（可进入 Architect）
- 6-7: 边缘（需补证据后复审）
- <=5: 无价值（记为未通过 Value Gate）
