# Learning Intake Review Queue

- queue_count: 4
- recommend_accept: 2
- recommend_edit: 2
- recommend_hold: 0
- default_training_admission_changed: false

## learning_intake:learning_focus_stress
- recommended_status: accept
- priority_score: 124.0
- sample_id: stress-learning-focus-evidence-routing-002
- reasons: pretrained 仍失败，说明这是当前真实缺口的直接监督候选。; pretrained_used_tool=calculator; baseline_used_tool=calculator
- prompt: Learning-focus routing stress 学习：样本=stress-learning-focus-evidence-routing-002。 期望工具=search，baseline=calculator，pretrained=calculator。 请把这个 evidence_judgment 误路由场景转成一条更稳健的离线监督任务，要求先检索公开证据，再判断是否能回答。

## learning_intake:attention_gap
- recommended_status: accept
- priority_score: 117.0
- sample_id: 
- reasons: attention handoff 仍是当前 top gap，适合进入候选训练审阅。; premature_release_count=1, conflict_to_verification_rate=0.2500
- prompt: AttentionFlow 学习任务：premature_release_count=1，conflict_to_verification_rate=0.2500。 当前发现 premature release 或 conflict->verification handoff 偏弱。请设计 2 条离线监督任务，要求残差必须先流向 VERIFY，之后才能 release。

## learning_intake:learning_focus_stress
- recommended_status: edit
- priority_score: 111.0
- sample_id: stress-learning-focus-evidence-routing-001
- reasons: baseline 失败但 pretrained 已修复，适合保留并压缩成更尖锐样本。; pretrained_used_tool=search; baseline_used_tool=bigmodel_proxy
- prompt: Learning-focus routing stress 学习：样本=stress-learning-focus-evidence-routing-001。 期望工具=search，baseline=bigmodel_proxy，pretrained=search。 请把这个 evidence_judgment 误路由场景转成一条更稳健的离线监督任务，要求先检索公开证据，再判断是否能回答。

## learning_intake:learning_focus_stress
- recommended_status: edit
- priority_score: 111.0
- sample_id: stress-learning-focus-evidence-routing-003
- reasons: baseline 失败但 pretrained 已修复，适合保留并压缩成更尖锐样本。; pretrained_used_tool=search; baseline_used_tool=calculator
- prompt: Learning-focus routing stress 学习：样本=stress-learning-focus-evidence-routing-003。 期望工具=search，baseline=calculator，pretrained=search。 请把这个 evidence_judgment 误路由场景转成一条更稳健的离线监督任务，要求先检索公开证据，再判断是否能回答。
