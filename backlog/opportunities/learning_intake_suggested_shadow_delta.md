# Learning Intake Suggested Shadow Delta

- base_sample_count: 184
- shadow_sample_count: 188
- added_count: 4
- removed_count: 0
- suggested_import_count: 5
- surviving_import_count: 4
- deduped_import_count: 1
- added_source_counts: {"learning_intake:search_first_adversarial_failure": 1, "learning_intake:learning_focus_stress": 3}

## Surviving Additions

### evidence_judgment
- source_type: human_review_patch
- candidate_source_type: learning_intake:search_first_adversarial_failure
- review_status: accept
- prompt: Search-first adversarial failure 学习：样本=search-first-adversarial-002。 当前与 shadow 都把 expected_tool=search 误路由成 calculator / calculator。 原题变体=evidence_gate_before_takeaway。 原始主题题目=公开 agent 设计学习：主题=tool-use stability under ambiguity。 当前标签=pending_label。 触发原因=frontier_zero_signal_persistence。 请基于公开资料总结可借鉴/不建议/待观察要点，并给出一个能在 Mustard 里低风险验证的实验。。 请把这个失败改写成一条更稳健的 evidence_judgment 离线监督任务，要求先检索公开证据，再区分事实、引用和待验证假设。 参考失败 prompt=围绕 tool-use stability under ambiguity，用户要你总结可借鉴/不建议/待观察三类要点，并指出哪条来自公开资料、哪条只是待验证假设。当前第一步该调用什么工具，为什么不能直接归纳？

### evidence_judgment
- source_type: human_review_patch
- candidate_source_type: learning_intake:learning_focus_stress
- review_status: edit
- prompt: Learning-focus routing stress 学习：样本=stress-learning-focus-evidence-routing-001。 期望工具=search，baseline=bigmodel_proxy，pretrained=search。 请把这个 evidence_judgment 误路由场景转成一条更稳健的离线监督任务，要求先检索公开证据，再判断是否能回答。

### evidence_judgment
- source_type: human_review_patch
- candidate_source_type: learning_intake:learning_focus_stress
- review_status: accept
- prompt: Learning-focus routing stress 学习：样本=stress-learning-focus-evidence-routing-002。 期望工具=search，baseline=calculator，pretrained=calculator。 请把这个 evidence_judgment 误路由场景转成一条更稳健的离线监督任务，要求先检索公开证据，再判断是否能回答。

### evidence_judgment
- source_type: human_review_patch
- candidate_source_type: learning_intake:learning_focus_stress
- review_status: edit
- prompt: Learning-focus routing stress 学习：样本=stress-learning-focus-evidence-routing-003。 期望工具=search，baseline=calculator，pretrained=search。 请把这个 evidence_judgment 误路由场景转成一条更稳健的离线监督任务，要求先检索公开证据，再判断是否能回答。

## Deduped Imports

### conflict_detection
- candidate_source_type: learning_intake:attention_gap
- review_status: accept
- prompt: AttentionFlow 学习任务：premature_release_count=1，conflict_to_verification_rate=0.2500。 当前发现 premature release 或 conflict->verification handoff 偏弱。请设计 2 条离线监督任务，要求残差必须先流向 VERIFY，之后才能 release。
