# 提案：以 Self-Harness 评估协议替代 0.9 单分数门槛

- 提案编号：v25（接续 v24）
- 日期：2026-08-19
- 关联文档：`docs/Mustard_修改与优化建议_2026-08-18.md` §8
- 变更类型：**Human Gate**（改 `configs/team_cycle.json` 的 `decision_policy`，属 `runtime_control` 治理变更）
- 状态：待用户审批

## 1. 问题（已实测确认）

- Arbiter 当前因 `configs/team_cycle.json:18` 的 `min_real_prompt_match_rate: 0.9` 将团队卡在 `uncertain_needs_human`：`real_prompt_match = 0.7302 < 0.90`。
- 2026-08-19 实测结论：
  - `0.7302`（46/63）是**外部工具可达环境**下的可达上限；其中 17 条失败是 v24 已结论的**模型上限**，任何 controls 都修不了。
  - 本沙箱外部工具（search / bigmodel_proxy）不可达时，真实得分塌到 ~0.11（7/63）——证明分数由外部可达性决定，不在 flag 调节范围内。
  - 因此绝对 0.9 门槛是**不可达目标**，会永久卡死 Arbiter，与"可达上限"事实矛盾。
- 项目内部已存在 ΔP 思想雏形：`deep_cycle_policy.require_positive_real_prompt_delta: true`（L62）与 `require_non_negative_control_success_delta`（L64）已要求正向增益，但 `decision_policy` 仍用绝对 0.9 硬卡——**两策略自相矛盾**。

## 2. 核心主张（docs/2026-08-18 §8）

以 **Self-Harness 评估协议**替代单分数门槛，真正衡量"Agent 会成长且成长可迁移"：

| 指标 | 含义 | 在本项目的落点 |
|---|---|---|
| `P_before` | 初始/可达上限能力 | 当前即 `0.7302`，作为基线，**不再要求绝对 0.9** |
| `ΔP` | 适应增益 = `P_after − P_before` | 引入候选 Harness 后的增益，为正即通过主线 |
| Transfer Gain | 迁移到未见任务 | 用 full 63 条 coverage 集观察，不卡 gate |
| Regression | 回归检测 | 锁定 `configs/real_prompt_regression.json` 的 46 条稳定子集，**零回归为硬闸** |
| Cost | token/latency/tool-call 代价 | 记录，不卡 |

## 3. 具体落地建议（需 Human Gate 批准）

1. `decision_policy.min_real_prompt_match_rate` **语义降级为"可达上限参考线"**：记录当前 `0.7302`，取消"绝对 <0.9 即 uncertain"的硬卡逻辑。
2. Arbiter 决策主线改为 `deep_cycle_policy` 已有的 ΔP 逻辑：候选须同时满足
   - `require_positive_real_prompt_delta = True`（P_after ≥ P_before）
   - `require_non_negative_control_success_delta = True`
   - 回归子集（`real_prompt_regression.json` 46 条）**零回归**
3. 若坚持保留数值门槛，建议设为 `P_before × 1.0`（不降）而非固定 `0.9`。

## 4. 影响与风险

- **解卡**：团队不再因 `0.7302 < 0.9` 永久卡死；Arbiter 转为关注"是否在可达上限上正向改进 + 无回归"，与 Self-Harness 方向一致。
- **风险**：
  - 改 `decision_policy` 属 `runtime_control` 治理变更，须 Human Gate 批准。
  - 调整前应在**外部可达 CI** 重测真实 `P_before`，确认 `0.7302` 是稳定上限而非沙箱假象（本沙箱不可达会失真）。
- **配套已完成**：`build_runner_from_state_dir` 已支持 `override_controls` 参数，`evaluate_combined_tool_policy_candidate` 已通过它真正注入候选 HarnessPolicy（原 `CONTROLS` 死代码陷阱已根除），候选闸现在能如实测候选 ΔP 与回归。
- **不改默认运行时行为**：`data/control/runtime_controls.json` 的 flag 状态不受影响（已确认 flag 全 1 无害）。

## 5. 验证计划

- 外部可达环境重跑 `evaluate_real_prompts`，确认 `P_before = 0.7302` 稳定。
- 跑 `evaluate_combined_tool_policy_candidate`（`override_controls` 已生效），输出候选 `ΔP` 与回归零回归判定。
- 模拟新 `decision_policy` 逻辑：`candidate_pass` 条件 = `ΔP ≥ 0` 且回归子集零回归。
- 回归：`python -m unittest discover -s tests`。

## 6. 待办

- [ ] 用户审批 Human Gate（改 `team_cycle.json` `decision_policy`）
- [ ] 外部可达 CI 重测 `P_before`
- [ ] 实施 `decision_policy` 语义调整（降级 0.9 为可达上限参考线 + 主线走 ΔP）
- [ ] 回归：`unittest discover -s tests`
