# Mustard Claw Team Design

## Goal

参考 OpenClaw 的核心思路，为 Mustard 设计一支“持续改进团队”：

- 由多个专职智能体围绕同一项目长期协作
- 通过共享记忆、固定节奏和结构化工件形成持续改进闭环
- 自动完成发现问题、提出改进、验证效果、准备变更的大部分工作
- 把高风险决策、产品方向和最终发布保留给人类把关

这个设计刻意不追求“所有 agent 都能直接改主分支”。更适合 Mustard 当前状态的，是围绕现有训练、评测、桌面桥梁和控制周期脚本，建立一个保守、可审计、可暂停的改进组织。

## What To Borrow From OpenClaw

OpenClaw 值得借鉴的不是“多几个 agent 名字”，而是下面四件事：

1. 有长期连续性的共享工作区，而不是每次从零开始
2. 有固定职责的角色分工，而不是所有 agent 都做所有事
3. 有周期性 heartbeat / review 机制，而不是只在用户提需求时动作
4. 有把记忆、任务、提案、结果都落盘的习惯，而不是只留在上下文里

对应到 Mustard，这意味着我们要补齐三类基础工件：

- `team/`：团队总规程、角色说明、审批规则
- `memory/`：日报、长期记忆、失败模式、用户偏好
- `backlog/`：候选问题、实验提案、已批准任务、回滚记录

## Design Principles

1. 人类只管关键闸门，不介入每次小循环
2. 自动化只对“低风险、可回滚、可评估”的变更放行
3. 一切提案都要绑定证据，而不是凭感觉优化
4. 默认小步实验，先离线验证，再灰度上线
5. 任何 agent 都不能单独决定训练数据准入、策略发布和桌面主动打扰策略

## Recommended Team Shape

推荐采用 `1 个编排者 + 6 个专职 agent + 1 个人类审批者` 的结构。

### 1. Conductor

团队编排者，负责触发周期、分派任务、收集结果、准备审批包。

职责：

- 按定时器或事件触发每日 / 每周改进循环
- 读取共享记忆、未完成 backlog 和最新评测结果
- 把任务派给各专职 agent
- 汇总为结构化 `change proposal`
- 把可自动执行和需人工审批的任务分流

它不直接决定产品方向，只做组织与状态推进。

### 2. Observer

问题发现者，负责持续扫描项目信号。

信号来源：

- `data/experience/episodes.jsonl`
- `data/desktop/bridge_*.jsonl`
- `data/train_runs/auto_train_latest.json`
- `configs/real_prompt_eval.json` 的回归结果
- `scripts/system_status.py`、`scripts.run_control_cycle`
- 测试失败、波动指标、误判反馈、追问预算收缩信号

输出：

- `backlog/incidents/*.md`
- `backlog/opportunities/*.md`
- `memory/failure_patterns.md`

它回答的问题是：哪里最值得改，为什么现在该改。

### 3. Curator

数据与记忆管理员，负责把原始反馈整理成可学习、可审计、可复用的工件。

职责：

- 聚合高价值 episode 和桥梁反馈
- 清洗误判样本与噪声样本
- 生成待审核的数据包
- 维护长期记忆与“最近改变了什么”
- 更新失败模式、用户偏好和已知禁区

Mustard 现在已经有 `build_real_prompt_candidates`、`consolidate_reviews`、`apply_pretrain_review_feedback` 这些能力，Curator 就是把这些脚本纳入固定团队职责。

### 4. Architect

方案设计师，负责把“问题”变成“可实施、可验证”的改进提案。

职责：

- 给出 1 到 3 个候选方案
- 说明影响范围、风险和回滚路径
- 明确是数据改进、策略改进、提示改进、桥梁改进还是 UI/交互改进
- 为 Builder 和 Evaluator 生成任务说明

输出格式建议固定为：

- `problem`
- `hypothesis`
- `proposed_change`
- `expected_metric_delta`
- `risk_level`
- `rollback_plan`
- `human_decision_needed`

### 5. Builder

执行者，负责在隔离分支或隔离工作区内实现候选改动。

允许它自动处理的范围：

- 新增或调整评测集
- 补测试
- 补数据整理脚本
- 调整低风险配置
- 生成实验分支上的代码改动

默认不允许它自动直推主分支，也不允许它绕过 Evaluator 直接发布。

### 6. Evaluator

验证者，负责对 Builder 的候选改动做统一评估。

Mustard 现有能力已经很适合这个角色直接接管：

- `python -m unittest discover -s tests -v`
- `python -m scripts.evaluate_pretraining`
- `python -m scripts.evaluate_real_prompts`
- `python -m scripts.run_control_cycle`
- `python -m scripts.judge_control_rollout`

Evaluator 必须输出结构化 verdict：

- `pass`
- `soft_pass`
- `fail`
- `needs_human_review`

并给出至少三类证据：

- 回归是否通过
- 哪个指标提升或退化
- 是否触发风险规则

### 7. Guardian

安全与边界守卫，专门盯住“自动化过头”的风险。

职责：

- 检查是否触碰桌面主动打扰策略
- 检查是否扩大了数据采集范围
- 检查是否修改了学习信号准入规则
- 检查是否涉及 secrets、外部 API、删除或覆盖历史数据
- 检查是否影响用户可感知行为

以下改动必须经过 Guardian + 人类双批准：

- 默认主动追问策略变化
- 训练数据准入标准变化
- 自动学习开关变化
- 桌面采样频率、截图策略、多模态入口变化
- 模型或外部工具供应商切换

### 8. Human Gate

人类不是日常执行者，而是关键闸门。

建议只保留四类人工审批：

1. 是否接纳高影响提案进入实验
2. 是否合并影响运行时行为的改动
3. 是否采纳训练数据审核包
4. 是否执行回滚、冻结或策略收缩

## Operating Model

建议把团队拆成三条循环，而不是一个大而全的循环。

### Loop A: Daily Observe

每天自动运行 2 到 4 次，只做发现与整理，不做发布。

流程：

1. Conductor 拉取最新状态
2. Observer 扫描失败、误判、测试、桥梁反馈和训练报告
3. Curator 归档高价值样本与异常模式
4. 形成 `daily digest`
5. 只在命中阈值时通知人类

产物：

- `memory/daily/YYYY-MM-DD.md`
- `backlog/opportunities/*.md`
- `backlog/incidents/*.md`

### Loop B: Weekly Improve

每周运行 1 次，自动挑选 1 到 3 个最高价值问题推进为实验。

流程：

1. Observer 提名问题
2. Architect 生成提案
3. Guardian 做前置风控分级
4. Human Gate 审批要不要开始实验
5. Builder 在隔离环境实现
6. Evaluator 统一跑验证
7. Conductor 生成周报和合并建议

这是团队的主增量循环。

### Loop C: Release / Rollback

只在有通过验证的候选变更时触发。

流程：

1. Evaluator 提交发布候选
2. Guardian 检查是否满足自动放行规则
3. 低风险改动可自动合并到实验分支
4. 运行时行为相关改动必须人工批准
5. 发布后进入短观察窗口
6. 如指标恶化，自动触发回滚建议

## Human Gate Policy

为了让“关键环节由人类把控”真正落地，建议用明确闸门，而不是笼统说“重要的交给人”。

### 可自动放行

- 新增测试
- 补充文档
- 新增评测样本但不改变准入标准
- 训练报告生成与摘要归档
- 低风险脚本修复
- 非默认路径下的实验配置

### 必须人工审批

- 修改默认模型、默认工具或默认学习行为
- 修改桌面采样、截图、主动追问预算和触发逻辑
- 修改训练数据过滤和 review 准入标准
- 会影响用户可见输出风格或决策偏好的变更
- 删除、覆盖、迁移历史经验数据
- 任何上线到默认运行路径的策略改动

## Shared Artifacts

建议新增一个最小团队工作区：

```text
team/
  AGENTS.md
  CONDUCTOR.md
  OBSERVER.md
  CURATOR.md
  ARCHITECT.md
  BUILDER.md
  EVALUATOR.md
  GUARDIAN.md
  HUMAN_GATE.md
memory/
  MEMORY.md
  failure_patterns.md
  user_preferences.md
  daily/
backlog/
  opportunities/
  incidents/
  proposals/
  approved/
  rejected/
  shipped/
```

每个 agent 启动时先读：

1. `team/AGENTS.md`
2. 对应角色文件
3. `memory/MEMORY.md`
4. 最近两天 `memory/daily/`
5. 自己负责的 backlog 文件夹

这就是 OpenClaw 风格里最重要的“长期连续性”。

## How This Maps To Current Mustard

这套设计不需要推翻现有脚本，反而应该把它们角色化。

### 当前可直接复用

- `scripts.auto_train`：作为 Curator / Evaluator 的训练闭环入口
- `scripts.evaluate_pretraining`：验证离线能力变化
- `scripts.evaluate_real_prompts`：验证真实 prompt 回归
- `scripts.run_control_cycle`：运行时控制实验采样
- `scripts.judge_control_rollout`：控制策略上线前闸门
- `scripts.consolidate_reviews`：聚合人工 / 半人工反馈
- `scripts.build_real_prompt_candidates`：从真实使用里沉淀回归集
- `scripts.apply_slow_path_actions`：作为批准后的慢速执行通道
- `scripts.rollback_runtime_controls`：作为回滚动作

### 还缺的薄层

1. `scripts/team_conductor.py`
2. `scripts/team_observer.py`
3. `scripts/team_curator.py`
4. `scripts/team_architect.py`
5. `scripts/team_evaluator.py`
6. `scripts/team_guardian.py`
7. `configs/team_cycle.json`
8. `team/`、`memory/`、`backlog/` 目录与模板文件

注意这里新增的是“组织与工件层”，不是重写 CARM 内核。

## Decision Protocol

每一次候选改进都应使用统一提案协议，例如：

```json
{
  "proposal_id": "prop_2026_03_22_001",
  "problem": "desktop bridge 在数据库选型场景下频繁误触发主动追问",
  "evidence": [
    "bridge feedback useful rate 下降",
    "misread 标签升高",
    "同类 prompt 在 real prompt eval 中 avg_step_count 增加"
  ],
  "change_type": "runtime_control",
  "risk_level": "high",
  "proposed_change": "收紧 proactive relevance gate，并加入 tag-level 抑制",
  "evaluation_plan": [
    "run_control_cycle",
    "judge_control_rollout",
    "real_prompt_eval subset"
  ],
  "rollback_plan": "rollback_runtime_controls",
  "needs_human_approval": true
}
```

这样人类审批时看的不是一堆日志，而是结构化决策包。

## Automation Rules

为了避免团队失控，建议固定几条规则：

1. 一个周期最多推进 3 个候选改进
2. 高风险提案不能并行上线
3. 连续两次验证失败的问题自动降级回 backlog
4. 同一问题没有新证据时，不允许无限重复实验
5. 任何自动改动都必须附带测试或评测依据
6. 自动学习只能写入候选区，不能直接覆盖正式基线

## Recommended First Version

第一版不要做真多 agent 并发。先做一个“单执行器 + 多角色协议”的假想团队最稳。

### V1

- 一个 `team_conductor.py` 顺序扮演多个角色
- 通过不同 prompt / 模板生成各角色输出
- 所有结果落到 `team/ memory/ backlog/`
- 只自动做观察、整理、提案和验证
- 不自动改主分支，不自动改默认运行时策略

### V2

- 把 Builder、Evaluator 分离成独立执行单元
- 支持并行实验分支
- 增加提案评分与优先级排序

### V3

- 接入真正的后台 heartbeat
- 支持按风险级别自动合并低风险改动
- 支持面向桌面代理、训练闭环、评测闭环的多 lane 编排

## Recommended Metrics

团队不是为了“更忙”，而是为了提高改进效率。建议追这些指标：

- `proposal_accept_rate`
- `experiment_success_rate`
- `regression_fail_rate`
- `time_to_validated_improvement`
- `bridge_useful_rate`
- `misread_rate`
- `tool_match_rate`
- `structured_write_rate`
- `avg_step_count`
- `rollback_frequency`

如果这些指标没有改善，就说明团队只是增加了流程噪音。

## Risks

### Risk 1: 团队过早复杂化

如果一开始就上真并行多 agent，会先把调度和状态同步搞复杂，反而掩盖真正的产品问题。

应对：

- 先做单执行器、多角色协议
- 先把工件和审批链跑顺

### Risk 2: 自动化替代了用户判断

Mustard 有桌面观察和主动追问链路，这类行为直接影响用户体验，不能让系统自己越改越激进。

应对：

- 主动行为相关变更一律人工审批
- Guardian 独立审查

### Risk 3: 数据污染

如果把误判 episode、噪声桥梁反馈或低质量 distill 数据直接喂回训练，会形成自我强化的坏循环。

应对：

- Curator 负责准入前整理
- Human Gate 审批关键训练包
- Evaluator 做回归隔离验证

## Final Recommendation

对 Mustard，最合适的不是“照搬 OpenClaw 的外形”，而是借用它的组织原则：

- 共享长期记忆
- 固定角色分工
- 周期性 heartbeat
- 一切落盘
- 自动做低风险部分
- 人类控制高风险闸门

最推荐的落地顺序是：

1. 先补 `team/ memory/ backlog/` 三类工件
2. 先实现 `Conductor + Observer + Curator + Architect + Evaluator + Guardian`
3. 暂时把 Builder 限制在实验分支和候选补丁
4. 只让人类审批高风险提案、训练包准入和默认行为上线
5. 跑 2 周后，再决定是否升级到真并行多 agent

这会让 Mustard 从“有训练和评测脚手架的单体项目”，升级成“有持续改进能力的半自动研究团队”。
