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

## Cycle Deliverables（每轮必交）

以下产物缺任意一项，则本轮状态自动记为 `incomplete`，不得推进到“已改进”。

1. Observer：`backlog/incidents/` 或 `backlog/opportunities/` 至少 1 条带证据条目
2. Failure Miner：最新 `memory/failure_patterns.md` 更新（含频次与影响）
3. Benchmark Owner：Top Gap（唯一最大缺口）+ 当前值/目标值/差距
4. Researcher：至少 1 个可复现实验假设（命令 + 指标 + 预期）
5. Architect：至少 1 份完整提案（problem/evidence/risk/eval/rollback）
6. Builder：提案对应的最小改动落地记录（分支或候选补丁）
7. Evaluator：统一验证结论（pass/soft_pass/fail/needs_human_review）
8. Arbiter：方向裁决（direction_correct/direction_adjust/uncertain_needs_human）
9. Conductor：每日汇总写入 `memory/daily/`（含 blockers 与 next actions）

## Role Activation Rules（防空转）

- 连续 2 轮无有效产出的角色，自动进入 `needs_redefinition`，Conductor 必须在日报中给出原因与修复动作。
- 连续 3 轮无有效产出的角色，默认降级为按需触发，不再占用“关键角色”名额。
- 关键角色（Researcher / Failure Miner / Benchmark Owner / Trainer / Arbiter）任一缺席，团队不得给出“方向明确且可持续推进”的结论。

## Done Definition（DoD）

仅当以下条件同时满足，才可宣称“本轮有效改进”：

- 有 failure_patterns 更新
- 有 Top Gap 与前后对比
- 有至少一项验证通过的改动（Evaluator=pass 或 soft_pass）
- 无触发 Human Gate 的未处理高风险项

## Startup Order

1. `team/AGENTS.md`
2. 角色文件
3. `memory/MEMORY.md`
4. 最近两天 `memory/daily/`
5. 自己负责的 backlog 目录
