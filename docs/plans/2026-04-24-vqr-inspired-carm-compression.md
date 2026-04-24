# VQR 启发的 CARM 压缩式推理路线

## 背景判断

Mustard 的核心目标应回到 CARM 本体：构建信息密度高、推理稳定、可验证的小模型/智能体原型。现有 `Claw Team` 更适合做维护与观察，不适合作为当前阶段的默认自动研发驾驶员。

本轮依据：

- `docs/plans/2026-04-12-codex-human-loop-reset.md` 已指出自动团队流程复杂度高于能力研发收益。
- `memory/failure_patterns.md` 的 top pattern 仍是 `sampling_blind_spot` 与冲突/多源整合盲区。
- `artifacts/current_best.json` 虽显示真实 prompt match rate 为 `1.0`，但主要指标仍集中在工具命中与步数，尚未证明中间表示真的更强。
- `D:\codes\vqr` 已有一条可借鉴的压缩思想：主模式码本、残差编码、质量门控、背景场扣除、局部重建。

## 可迁移思想

VQR 不应被照搬为“把文本压缩成文件”，而应迁移为 CARM 的推理表示原则：

1. 主模式码本
   - VQR 用 codebook 表示高频时空模式。
   - CARM 可用 reasoning pattern codebook 表示常见任务结构，例如 comparison、conflict、tool_boundary、termination、integration。

2. 残差表示
   - VQR 不只存码字，还保存码字无法解释的 residual。
   - CARM 应显式保存当前任务相对典型模式的 residual，例如新增约束、冲突点、缺失证据、异常工具边界。

3. 质量门控
   - VQR 用压缩率与重建质量共同决定是否启用二级码本。
   - CARM 应用 `fit_score / residual_risk / answer_ready` 决定是否直接回答、验证、继续查证或升级到人工判断。

4. 背景场扣除
   - VQR 先扣除时间均值或静态背景，再压缩剩余变化。
   - CARM 可把任务类型、用户偏好、项目记忆视为 background，核心只处理偏离背景的高信息残差。

5. 局部重建
   - VQR bundle 支持只恢复一个区域。
   - CARM 的记忆与推理轨迹应支持按 claim/source/conflict/tool_decision 局部恢复，而不是每次展开整段上下文。

## 建议目标

短期北极星不再写成“超越 GPT”，而是：

> 在冲突判断、多源整合、工具边界与终止判断四类高信息任务上，用更紧凑的中间表示得到更稳定的可验证推理行为。

## 最小实验

### MVI-1: reasoning pattern codec

新增一个离线实验模块，不接默认运行时：

- 输入：现有 real prompt eval rows、pretrain samples、成功 episode 摘要。
- 输出：`pattern_id + residual_features + fit_score + reconstruction_notes`。
- 评估：
  - 同一类任务是否被压到稳定 pattern。
  - residual 是否能解释失败模式。
  - residual 特征是否能预测需要 `VERIFY`、`search` 或 `bigmodel_proxy`。

### MVI-2: hard logic eval pack

冻结一组高信息任务，不追求数量，追求诊断密度：

- conflict_detection
- result_integration
- termination_judgment
- tool_boundary

每条样本必须标注：

- expected_pattern
- required_residuals
- expected_decision
- unacceptable_failure

### MVI-3: current_best 扩展

在 `artifacts/current_best.json` 的后续生成链路中追加能力指标：

- `hard_logic_pass_rate`
- `residual_explanation_rate`
- `critical_failure_count`
- `avg_steps_or_cost`

这一步只改报告链路，不改默认推理行为。

## 不做事项

- 不让 `Claw Team` 自动修改默认运行时策略。
- 不自动扩大桌面采样、截图或主动追问。
- 不自动改变训练数据准入规则。
- 不把 VQR 的数值压缩代码直接复制进 CARM 核心。

## 建议执行顺序

1. 先实现离线 `reasoning pattern codec` 原型。
2. 再建立 hard logic eval pack。
3. 用 hard eval 对比现有 latent/slot 表示与 pattern/residual 表示。
4. 只有当离线指标正向，再考虑接入 `AdaptiveReasoningCore`。

