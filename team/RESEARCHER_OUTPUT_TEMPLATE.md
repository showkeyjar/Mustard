# Researcher Output Template (Value Gate)

> 用途：Researcher 每轮必须按此模板提交；缺字段即视为无效产出。

## 1) Meta

- round_id:
- date:
- owner:
- from_top_gap:
- from_failure_pattern:
- relative_to_last_round:
- scenario_fit:

## 2) Hypothesis（可证伪）

- hypothesis:
- falsifiable_condition:  （什么结果出现时判定假设失败）
- expected_gain:
- risk:

## 3) Evidence Pack

- evidence_1:
- evidence_2:
- evidence_3:
- evidence_quality_note:

## 4) Experiment Plan（可执行）

- command_1:
- command_2:
- metric_threshold:
- pass_criteria:
- fail_criteria:

## 5) Landing Candidate（可直接进 Architect）

- proposed_change:
- change_scope:
- rollback_plan:
- handoff_to_architect: yes/no

## 6) Verdict Tag

- tag: 可借鉴 / 不建议跟进 / 待观察
- reason:

---

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
