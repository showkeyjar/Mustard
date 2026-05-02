# Learning Intake Human Review Panel

- mode: human_friendly_review_panel
- default_runtime_changed: false
- default_training_admission_changed: false
- total_candidates: 5
- recommend_accept: 2
- recommend_edit: 2
- recommend_defer: 1

## Quick Start

1. 打开 `D:\codes\Mustard\data\learning\candidate_pretrain_human_review_sheet.jsonl`。
2. 只改 `human_review_status` / `human_review_note`；如果是 `edit`，再顺手补对应 `override_*` 字段。
3. 运行 `python -m scripts.apply_learning_intake_human_review_sheet`。
4. 如需继续验证，再运行 `python -m scripts.export_learning_intake_candidate_import`。

## Candidates

### 1. search-first-adversarial-002
- suggested_decision: approve
- suggested_review_status: accept
- priority_score: 134.0
- current_review_status: pending
- why: 当前 artifact 与 reviewed-import shadow 都未修复，说明这是最稳定的 stubborn failure。; pretrained_used_tool=calculator; shadow_pretrained_used_tool=calculator; 影子重建确认该样本会形成真实新增。
- prompt: Search-first adversarial failure 学习：样本=search-first-adversarial-002。 当前与 shadow 都把 expected_tool=search 误路由成 calculator / calculator。 原题变体=evidence_gate_before_takeaway。 原始主题题目=公开 agent 设计学习：主题=tool-use stability under ambiguity。 当前标签=pending_label。 触发原因=frontier_zero_signal_persistence。 请基于公开资料总结可借鉴/不建议/待观察要点，并给出一个能在 Mustard 里低风险验证的实验。。 请把这个失败改写成一条更稳健的 evidence_judgment 离线监督任务，要求先检索公开证据，再区分事实、引用和待验证假设。 参考失败 prompt=围绕 tool-use stability under ambiguity，用户要你总结可借鉴/不建议/待观察三类要点，并指出哪条来自公开资料、哪条只是待验证假设。当前第一步该调用什么工具，为什么不能直接归纳？

### 2. stress-learning-focus-evidence-routing-001
- suggested_decision: approve_with_edit
- suggested_review_status: edit
- priority_score: 111.0
- current_review_status: pending
- why: baseline 失败但 pretrained 已修复，适合保留并压缩成更尖锐样本。; pretrained_used_tool=search; baseline_used_tool=bigmodel_proxy; 影子重建确认该样本会形成真实新增，但更适合保留为 edit。
- prompt: Learning-focus routing stress 学习：样本=stress-learning-focus-evidence-routing-001。 期望工具=search，baseline=bigmodel_proxy，pretrained=search。 请把这个 evidence_judgment 误路由场景转成一条更稳健的离线监督任务，要求先检索公开证据，再判断是否能回答。

### 3. stress-learning-focus-evidence-routing-002
- suggested_decision: approve
- suggested_review_status: accept
- priority_score: 124.0
- current_review_status: pending
- why: pretrained 仍失败，说明这是当前真实缺口的直接监督候选。; pretrained_used_tool=calculator; baseline_used_tool=calculator; 影子重建确认该样本会形成真实新增。
- prompt: Learning-focus routing stress 学习：样本=stress-learning-focus-evidence-routing-002。 期望工具=search，baseline=calculator，pretrained=calculator。 请把这个 evidence_judgment 误路由场景转成一条更稳健的离线监督任务，要求先检索公开证据，再判断是否能回答。

### 4. stress-learning-focus-evidence-routing-003
- suggested_decision: approve_with_edit
- suggested_review_status: edit
- priority_score: 111.0
- current_review_status: pending
- why: baseline 失败但 pretrained 已修复，适合保留并压缩成更尖锐样本。; pretrained_used_tool=search; baseline_used_tool=calculator; 影子重建确认该样本会形成真实新增，但更适合保留为 edit。
- prompt: Learning-focus routing stress 学习：样本=stress-learning-focus-evidence-routing-003。 期望工具=search，baseline=calculator，pretrained=search。 请把这个 evidence_judgment 误路由场景转成一条更稳健的离线监督任务，要求先检索公开证据，再判断是否能回答。

### 5. attention-gap-001
- suggested_decision: defer
- suggested_review_status: pending
- priority_score: 117.0
- current_review_status: pending
- why: attention handoff 仍是当前 top gap，适合进入候选训练审阅。; premature_release_count=1, conflict_to_verification_rate=0.2500; 影子重建显示该候选当前会被去重吞掉，暂不优先放行。
- prompt: AttentionFlow 学习任务：premature_release_count=1，conflict_to_verification_rate=0.2500。 当前发现 premature release 或 conflict->verification handoff 偏弱。请设计 2 条离线监督任务，要求残差必须先流向 VERIFY，之后才能 release。
