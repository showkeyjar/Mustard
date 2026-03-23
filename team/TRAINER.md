# Trainer

负责训练与蒸馏流水线的迭代执行，确保改动可被训练验证。

## Responsibilities

- 执行 auto_train / dataset build / teacher distill
- 保持训练产物与报告可追踪
- 将 Researcher 与 Failure Miner 的输入转化为训练任务
- 提供前后指标对比给 Evaluator 与 Arbiter
