# Top Gap Action Card

- gap_id: new_failure_pattern_stalled
- problem: 长期停滞但没有新增 failure pattern，说明新增弱点发现环节失灵。
- current: stagnation_rounds=55; new_failure_pattern_count=0
- target: 在后续 1~2 个周期内形成至少 1 个新增 failure pattern 或新弱点簇
- gap: 55
- owner: researcher + failure_miner + arbiter
- why_this_is_top_gap_now: 覆盖数已达标，但系统仍未形成新增发现，当前瓶颈已从 coverage 转向 discovery。
- action_plan:
  - 围绕高信号样本簇生成更具区分度的压测 prompt
  - 优先验证 comparison / conflict_detection / tool_boundary 的新弱点簇
  - 要求 researcher 明确给出可证伪的新 failure hypothesis
- acceptance:
  - 出现至少 1 个新增 failure pattern 或新 recovery cluster
  - research_quality 不再出现 no_new_failure_pattern
- rollback: 若新增样本只制造噪声，则回退本轮高压样本并保留有效簇
