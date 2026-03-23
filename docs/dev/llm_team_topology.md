# LLM 研发团队角色编排（Mustard）

为贴近大模型研发团队，当前角色分为三层：

## 1) 研究层（Research）
- Researcher：前沿路线跟踪、提出假设、避免重复踩坑
- Architect：将研究假设转为可验证技术方案

## 2) 训练与评测层（Train/Eval）
- Data Curator：数据采样、清洗、标注与版本化
- Trainer：训练流水线与训练产物管理
- Evaluator：基准评测、回归评测与门槛判断

## 3) 交付与安全层（Delivery/Safety）
- Conductor：编排周期与输出节奏
- Guardian：风险边界审查与 Human Gate
- Arbiter：方向裁决（不确定时升级人类）

> 说明：当前仓库已实现核心角色闭环（Conductor/Observer/Architect/Evaluator/Guardian/Arbiter/Researcher），
> 其余角色可按自动化成熟度逐步拆分，不一次性过度复杂化。
