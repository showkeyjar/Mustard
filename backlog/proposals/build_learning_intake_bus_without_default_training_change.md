# Build Learning Intake Bus Without Default Training Change

- problem: 现有仓库已经能从 experience、bridge feedback、frontier research、attention artifacts 收集局部信号，但这些信号还没有统一进入可审阅的学习入口。
- from_failure_pattern: attention_verification_handoff_gap
- from_top_gap: attention_verification_handoff_gap
- change_type: evaluation_or_dataset
- proposed_change: 新增 learning intake bus，把用户操作纠偏、公开 agent 设计思想、attention gap 统一转成离线学习候选与 review pack。
- expected_metric_delta: 学习候选来源不再只依赖 experience；可审阅学习样本覆盖 experience / feedback / frontier / public ideas / attention gap 五类来源。
- risk_level: low
- needs_human_approval: False
- relative_to_last_round: 从单一 attention handoff 提案，扩展为统一学习入口，让更多外部和交互信号可以进入离线进化闭环。
- scenario_fit: 用户在真实操作中不断纠偏系统，同时仓库也需要持续吸收公开 agent 设计经验。
- architect_handoff: conductor + curator + trainer align on intake format before any automatic training admission
- rollback_plan: 删除 learning intake artifacts 与脚本，不影响默认训练或运行时。
- evaluation_plan:
  - python -m scripts.build_learning_intake
  - python -m unittest tests.test_learning_intake -v
