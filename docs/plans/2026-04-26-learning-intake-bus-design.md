# Learning Intake Bus Design

## Goal

让 Mustard 开始系统性利用一切低风险可用信号进化，但不直接改默认运行时，也不绕过训练数据准入。

## Core Idea

在 `experience -> feedback -> frontier/public ideas -> attention gaps -> training` 之间增加一层统一的 `Learning Intake Bus`：

1. 收集多源信号
2. 转成结构化候选学习样本
3. 打分、去重、导出 review pack
4. 仅在人工或后续 gate 同意时进入离线训练构建

## Sources

- `data/experience/episodes.jsonl`
- `data/desktop/bridge_events.jsonl`
- `data/desktop/bridge_feedback.jsonl`
- `data/research/frontier_observations.jsonl`
- `data/research/public_agent_ideas.jsonl`
- `artifacts/attention_flow_latest.json`
- `artifacts/attention_training_views_latest.json`

## Outputs

- `data/learning/learning_intake_samples.jsonl`
- `data/learning/learning_intake_import.jsonl`
- `data/learning/learning_intake_review_pack.jsonl`
- `backlog/opportunities/learning_intake_report.md`

## Why This Matters

这层机制把“用户怎么纠偏系统”“公开 agent 设计思想有哪些可借鉴点”“AttentionFlow 现在哪里断了”都变成统一的、可审阅的监督候选，而不是分散在各自的日志和报告里。

## Safety

- 不修改默认运行时
- 不自动并入默认训练集
- 不覆盖历史数据
- 所有学习候选都保留来源与 review 入口

## Next Step

把 `learning_intake_report` 接到 `team_conductor` 的研究恢复与 curator 流程里，让系统按周期自动沉淀候选学习样本。 
