# Top Gap Action Card

- gap_id: blind_spot_not_broken
- problem: 研究已识别 blind spot，但仍未证明该盲区被真正打穿。
- current: high_signal_count=1; blind_spot_persistence_rounds=13
- target: 通过更强压力样本证明 blind spot 已消除，或显式暴露出新的 mismatch cluster
- gap: 13
- owner: researcher + benchmark_owner + architect
- why_this_is_top_gap_now: 当前主要问题不是样本数量，而是高信息盲区迟迟没有被打穿。
- action_plan:
  - 围绕高信号行构建更尖锐的 follow-up prompts
  - 减少低信息重复样本，提升单条样本的揭弱能力
  - 要求 architect 输出直接针对 blind spot 的最小落地变更
- acceptance:
  - blind_spot_persistence_rounds 下降或归零
  - 新一轮研究产物不再声明 blind spot remains
- rollback: 若新样本无法提供更高信息增量，则回退到上一轮高信号集合
