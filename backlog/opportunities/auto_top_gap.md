# Top Gap Action Card

- gap_id: eval_coverage_too_low
- problem: 真实工具调用场景回归样本覆盖不足，当前无法证明可替代性提升。
- current: prompt_count=4
- target: prompt_count>=20（优先本地工具调用场景）
- gap: 16
- owner: benchmark_owner + failure_miner + trainer
- why_this_is_top_gap_now: 当前真实 prompt 覆盖数未达最低目标，基础验证样本仍不足。
- action_plan:
  - 运行 python -m scripts.build_real_prompt_candidates 生成候选集
  - 合并候选集到 configs/real_prompt_eval.json（去重后保留高价值工具调用样本）
  - 运行 python -m scripts.evaluate_real_prompts 并记录前后对比
- acceptance:
  - real_prompt_eval.summary.prompt_count >= 20
  - 产出一份前后指标对比摘要（match_rate / avg_steps）
- rollback: 若样本质量下降，回退 real_prompt_eval.json 到上一版并重新评测
