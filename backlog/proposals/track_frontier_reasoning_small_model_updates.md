# Track frontier reasoning-small-model updates

- problem: 前沿小模型研究观察样本不足，容易重复踩坑。
- change_type: research_tracking
- risk_level: low
- needs_human_approval: False
- proposed_change: Researcher 本周补齐 DeepSeek / MiniMax 等前沿观察，并形成可借鉴结论标签。
- rollback_plan: 仅新增研究记录，不影响运行时。
- evidence:
  - frontier_observation_count=0
- evaluation_plan:
  - 更新 docs/dev/research_frontier_watchlist.md
  - 沉淀观察到 data/research/frontier_observations.jsonl
