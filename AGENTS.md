# Mustard OpenClaw Workspace

这个工作区用于让 OpenClaw 在 `Mustard` 仓库内直接按项目既有的 `Claw Team` 规则持续工作。

## Session Start

每次进入本工作区时，按下面顺序建立上下文：

1. 读取 [IDENTITY.md](/d:/codes/Mustard/IDENTITY.md)
2. 读取 [SOUL.md](/d:/codes/Mustard/SOUL.md)
3. 读取 [USER.md](/d:/codes/Mustard/USER.md)
4. 读取 [team/AGENTS.md](/d:/codes/Mustard/team/AGENTS.md)
5. 读取自己涉及的角色文件，例如 [team/CONDUCTOR.md](/d:/codes/Mustard/team/CONDUCTOR.md)、[team/ARCHITECT.md](/d:/codes/Mustard/team/ARCHITECT.md)、[team/BUILDER.md](/d:/codes/Mustard/team/BUILDER.md)、[team/GUARDIAN.md](/d:/codes/Mustard/team/GUARDIAN.md)
6. 读取 [configs/team_cycle.json](/d:/codes/Mustard/configs/team_cycle.json) 与 [configs/team_github.json](/d:/codes/Mustard/configs/team_github.json)
7. 读取 [memory/MEMORY.md](/d:/codes/Mustard/memory/MEMORY.md)
8. 读取最近两天的 [memory/daily/](/d:/codes/Mustard/memory/daily)

## Mission

- 持续发现、验证并推动 Mustard 的低风险改进
- 把高风险改动明确升级到 Human Gate
- 优先使用仓库内现有脚本和配置，而不是临时发明并行流程

## Non-Negotiables

- 先找证据，再提改动
- 默认小步迭代，不直接修改默认运行时行为
- 不绕过评测、Guardian 审查和 GitHub 交付闸门
- 以下变更必须交给人类审批：
  - 默认运行时策略
  - 桌面采样、截图和主动追问策略
  - 训练数据准入规则
  - 默认模型或工具供应商
  - 历史数据迁移、删除、覆盖

## Execution Path

- 团队周期入口：`python -m scripts.claw_team_control run`
- 体检入口：`python -m scripts.claw_team_control doctor`
- 状态入口：`python -m scripts.claw_team_control status`
- GitHub 交付入口：`python -m scripts.claw_team_control deliver --title "..."`
- GitHub 诊断入口：`python -m scripts.claw_team_control github-doctor`

## Memory Rules

- 长期记忆的项目版本以 [memory/MEMORY.md](/d:/codes/Mustard/memory/MEMORY.md) 为准
- 每日日志写入 [memory/daily/](/d:/codes/Mustard/memory/daily)
- 候选提案写入 [backlog/proposals/](/d:/codes/Mustard/backlog/proposals)

## Workspace Safety

- 只在仓库工作区内读写
- 未经明确授权，不向外部发送消息或公开发布内容
- 未经明确授权，不执行破坏性 Git 操作
