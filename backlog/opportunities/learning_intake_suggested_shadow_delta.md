# Learning Intake Suggested Shadow Delta

- base_sample_count: 184
- shadow_sample_count: 187
- added_count: 3
- removed_count: 0
- suggested_import_count: 4
- surviving_import_count: 3
- deduped_import_count: 1
- added_source_counts: {"learning_intake:learning_focus_stress": 3}

## Surviving Additions

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
