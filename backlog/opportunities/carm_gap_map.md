# CARM Gap Map

## Top Gaps

### Gap 1: 真实场景覆盖不足
- current: real_prompt_count=20
- target: >=20
- blocker: 候选样本构建与筛选未形成稳定流水线
- owner: CARM Owner + Benchmark Owner
- next_action: build_real_prompt_candidates + evaluate_real_prompts
- acceptance: prompt_count>=20 且报告可复现

### Gap 2: 用户纠偏闭环缺失
- current: bridge_feedback=0
- target: >=30 条高价值反馈
- blocker: bridge 反馈采样与标注流程缺位
- owner: CARM Owner + Failure Miner
- next_action: 启动 bridge 反馈采集并结构化标注
- acceptance: 形成反馈样本包并进入评测

### Gap 3: 前沿对标不足
- current: frontier_observation_count=0
- target: >=10 条可比较观察
- blocker: 外部路线跟踪未形成固定节奏
- owner: Arbiter(CARM Track) + Researcher
- next_action: 每轮补充 3 条前沿观察并打标签
- acceptance: 形成可借鉴/不建议/待观察三类结论

## CARM MVI（本轮）
- carm_mvi: 先完成 Gap1 的样本覆盖增量（6 -> 20）并输出前后对比
- window: 24-72h
