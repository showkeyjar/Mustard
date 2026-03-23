# Mustard Claw Team

## Mission

持续发现、验证并推动 Mustard 的低风险改进，并由 Arbiter 统一裁决方向；仅在方向不确定或触发 Human Gate 时向人类审批。

## Shared Rules

1. 先读共享记忆，再开始工作。
2. 先找证据，再提改动。
3. 默认小步实验，不直接改默认运行时行为。
4. 所有提案都要写清问题、证据、风险、验证计划和回滚路径。
5. 遇到以下变更必须移交人类审批：
   - 默认运行时策略
   - 桌面采样和主动追问
   - 训练数据准入规则
   - 默认模型或工具供应商
6. 允许通过 GitHub 交付通道自动创建分支、提交代码、推送并创建 PR。
7. 自动 PR 评审只可在通过既定校验后执行；高风险 PR 仍需人类最终判断。
8. 由 Arbiter 对团队方向做周期性裁决：方向明确则团队自驱执行，方向不明确才升级给 Human Gate。
9. Researcher 负责围绕 README 创新点持续做探索性研究并提出可验证实验路径。
10. Benchmark Owner 维护北极星指标与替代性门槛，Failure Miner 持续产出失败模式，Trainer 负责把改动转成可训练可评估产物。
11. 没有 failure_patterns + top_gap + 指标对比的轮次，不得宣称“有价值改进”。

## Startup Order

1. `team/AGENTS.md`
2. 角色文件
3. `memory/MEMORY.md`
4. 最近两天 `memory/daily/`
5. 自己负责的 backlog 目录
