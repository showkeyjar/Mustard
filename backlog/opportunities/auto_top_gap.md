# Top Gap Action Card

- gap_id: quality_stabilization
- problem: 当前主要任务是维持已识别高信号质量差异，并防止退化。
- current: high_signal_count=0
- target: 保持高信号样本稳定可复现，并避免质量回退
- gap: 0
- owner: benchmark_owner + evaluator
- why_this_is_top_gap_now: coverage 已达标，且未检测到更高优先级缺口，当前以质量维稳为主。
- action_plan:
  - 复核高信号样本集的稳定性
  - 继续监控 baseline/pretrained 分离度
- acceptance:
  - 高信号样本集持续可复现
- rollback: 若质量维稳样本失真，则回退到上一轮有效集合
