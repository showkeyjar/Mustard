# Top Gap Action Card

- gap_id: attention_verification_handoff_gap
- problem: 智能体工作流尚未稳定投影成 attention handoff，冲突残差没有可靠流向 verification 或 release gate。
- current: premature_release_count=1; conflict_to_verification_rate=0.2500
- target: premature_release_count=0 且 conflict_to_verification_rate>=0.50
- gap: 25
- owner: architect + benchmark_owner + trainer
- why_this_is_top_gap_now: 覆盖已不再是主矛盾，当前更关键的是让冲突/边界残差真正顺流到 verification，而不是过早释放答案。
- action_plan:
  - 围绕 premature release case 生成 follow-up attention supervision 样本
  - 把 conflict/tool-boundary residual 显式映射到 recommended_transition=VERIFY
  - 复跑 attention_flow 与 attention_training_views 评估并核对 handoff 改善
- acceptance:
  - attention_flow.premature_release_count == 0
  - attention_training_views.conflict_to_verification_rate >= 0.50
- rollback: 若 attention 监督视图只带来噪声，则回退新增 projector / evaluator 样本而不影响默认运行时
