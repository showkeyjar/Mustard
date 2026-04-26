# AttentionFlow: 把智能体工作流转成模型可学习的注意力轨迹

## 背景判断

当前 CARM 已经有结构化工作记忆、在线策略、经验回放、reasoning pattern codec 和真实 prompt 评测。但这些机制仍主要是智能体工程层的状态机：它们能记录做了什么，却还不能稳定表达“为什么此刻应该看这里、忽略那里、等待什么证据、何时收束”。

这造成一个核心断裂：

- 大模型的智能主要体现在注意力分配、上下文压缩、证据整合和状态转移。
- 智能体的设计主要体现在动作协议、工具调用、记忆槽位和评测闸门。
- 如果两者之间没有可学习的中间层，智能体经验只能变成离散动作日志，大模型也无法吸收智能体在真实任务中形成的流程知识。

因此，下一阶段不应只继续堆工具策略或样本数量，而应引入一个中间表示：`AttentionFlow`。它把智能体每一步工作流投影成“焦点、证据、残差、转移动机、收束条件”的轨迹，让模型能学习心流式推理，而不是只模仿动作序列。

## 核心定义

`AttentionFlow` 是 episode 的可学习投影，不替代现有 `StepRecord`，也不直接改变运行时行为。它从现有轨迹派生：

- `focus_target`: 当前注意力聚焦对象，例如 goal、constraint、evidence、conflict、tool_boundary、draft。
- `focus_reason`: 为什么当前应聚焦该对象。
- `evidence_need`: 当前还缺什么证据或外部支持。
- `residual_pressure`: 当前任务相对典型模式仍未解释的残差，例如冲突未消解、边界模糊、证据不足。
- `transition`: 本步从上一焦点迁移到下一焦点的原因，例如 plan_to_evidence、evidence_to_draft、conflict_to_verify。
- `release_condition`: 什么时候可以停止关注当前问题并进入下一阶段。
- `model_view`: 可训练文本视图，把上述结构压成模型能学习的简洁样本。

这使 CARM 的学习对象从：

> prompt -> action/tool/slot

升级为：

> prompt + context -> attention state -> transition -> action/tool/slot -> residual outcome

## 与现有系统的关系

现有 `reasoning_codec` 是宏观压缩：它把整条评测轨迹编码为 `pattern_id + residual_features + fit_score`。`AttentionFlow` 是微观轨迹：它解释每一步的焦点如何移动，以及哪些残差正在驱动下一步。

现有 `StepRecord` 已经包含足够的第一版信号：

- `action`
- `target_slot`
- `selected_tool`
- `feature_snapshot`
- `state_signature`
- `memory_signature`
- `reward`
- `reward_reason`
- `glance_used`

第一版不需要新增采样、不需要外部模型、不需要改变默认控制。只要新增一个离线 projector，就能把历史 episode 和真实 prompt eval 的 step traces 转成 AttentionFlow 数据集。

现有 `glance` 可以被重新解释为低成本内部注意力脉冲：它不是最终方案，但可以作为 AttentionFlow 的一个弱监督来源，例如 `prefer_tool`、`delay_answer`、`mark_conflict` 分别映射到证据聚焦、收束延迟、冲突聚焦。

## 架构草图

```text
EpisodeRecord / Eval Trace
        |
        v
AttentionFlow Projector
        |
        +--> attention_flow.jsonl
        |
        +--> Attention Metrics
        |       - focus_continuity
        |       - evidence_grounding
        |       - residual_resolution
        |       - transition_validity
        |       - premature_release_rate
        |
        +--> Training Views
                - next_focus prediction
                - transition rationale
                - residual-aware tool/slot decision
                - release/verify decision
```

这个架构的关键是双投影：

1. 面向智能体：保留动作、槽位、工具和奖励，继续服务评测与审计。
2. 面向模型：压成注意力状态与焦点转移，让模型学到流程，而不是只记住动作。

## 心流的工程定义

这里的“心流”不作为玄学概念，而定义为可测量的推理连续性：

- 每一步焦点都有明确来源，而不是随机跳转。
- 注意力会优先流向最大残差，而不是平均处理所有信息。
- 外部证据进入后，焦点能从 evidence_need 转到 integration 或 draft。
- 冲突未消解时不会释放到 final answer。
- 当残差压力下降且 release_condition 满足时，系统能自然收束。

因此，心流不是“想得更多”，而是：

> 注意力压力沿着任务结构稳定下降，直到答案可以被释放。

## 最小可落地实验

### MVI-1: AttentionFlow Schema + Projector

新增离线模块，从 `EpisodeRecord.steps` 派生 attention nodes。

输入：

- `data/experience/episodes.jsonl`
- 测试中可构造的 `RunTrace`

输出：

- `data/attention/attention_flow.jsonl`
- 每条 episode 一组 ordered nodes

验收：

- 不改变 `AgentRunner.run` 默认决策。
- 不改变 `configs/real_prompt_eval.json`。
- projector 能从现有 StepRecord 派生非空 `focus_target / transition / residual_pressure`。

### MVI-2: Attention Metrics

新增报告脚本，计算注意力质量。

建议指标：

- `focus_continuity`: 相邻焦点迁移是否有可解释 transition。
- `evidence_grounding`: 调用工具前后是否从 evidence_need 转到 result/integration。
- `residual_resolution`: 高风险 residual 是否在 answer 前被 verify 或降压。
- `premature_release_rate`: 冲突、证据不足或工具边界不清时是否过早 ANSWER。
- `attention_compression_ratio`: 原始 step trace 到 attention nodes 的压缩比。

验收：

- 能解释当前 `repeated_conflict_detection_gap`。
- 能给出比 tool_match_rate 更细的失败原因。

### MVI-3: Training View Export

把 AttentionFlow 转成离线训练样本，但不直接并入默认训练集。

样本形态：

```json
{
  "user_input": "...",
  "current_focus": "conflict",
  "residual_pressure": ["conflict_unresolved", "missing_evidence"],
  "next_focus": "evidence",
  "recommended_transition": "conflict_to_evidence",
  "recommended_action": "CALL_TOOL",
  "release_allowed": false
}
```

验收：

- 只生成 `data/attention/training_views.jsonl`。
- 不修改 `data/pretrain/pretrain_corpus.jsonl`。
- 后续如要进入训练数据准入，必须 Human Gate。

### MVI-4: Candidate Runtime Guidance

只有当前三步离线指标证明有效后，才做隔离候选：

- 在临时 runner 中读取 attention guidance。
- 只影响候选路径，不改默认 runtime controls。
- 用 hard logic eval 和 real prompt eval 验证是否降低 premature answer、missing verify、tool boundary shift。

## 风险与边界

必须避免三类误区：

1. 把 AttentionFlow 做成另一套复杂状态机。第一版只做投影和评估，不接管决策。
2. 过早把生成样本并入训练集。训练数据准入属于 Human Gate。
3. 把“注意力”误解为更多上下文。真正目标是更好地选择、压缩和释放上下文。

高风险项：

- 默认运行时策略变更：Human Gate。
- 训练数据准入变更：Human Gate。
- 桌面采样或主动追问策略变更：Human Gate。
- 默认模型或工具供应商变更：Human Gate。

## 建议执行顺序

1. 写 `carm/attention_flow.py`，只定义 schema 和 projector。
2. 为 projector 增加单元测试，覆盖 comparison、conflict_detection、tool_boundary。
3. 写 `scripts/export_attention_flow.py`，从 episodes 导出 attention flow。
4. 写 `scripts/evaluate_attention_flow.py`，输出 attention metrics。
5. 在 `current_best` 或独立 artifact 中展示 attention 指标。
6. 若指标稳定，再讨论是否导出训练视图。
7. 若训练视图经人工批准，再进入离线训练实验。

## 成功标准

短期成功不是“模型突然更聪明”，而是系统能回答三个问题：

- 这次失败是工具选错，还是注意力焦点过早释放？
- 这次成功是靠工具命中，还是靠 residual 被正确消解？
- 哪些智能体经验可以被转成模型可学习的注意力转移？

当这三点能被稳定记录、评测和回放时，芥子系统才真正开始把智能体经验转成模型可吸收的心流。
