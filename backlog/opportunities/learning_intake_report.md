# Learning Intake Report

- candidate_count: 10
- source_breakdown: {"learning_intake:public_idea": 3, "learning_intake:learning_focus_stress": 3, "learning_intake:attention_gap": 1, "learning_intake:frontier": 3}
- default_runtime_changed: false
- default_training_admission_changed: false

## Recommended Use

- Step 1: 审阅 data/learning/learning_intake_review_pack.jsonl
- Step 2: 如需将提示并入离线构建，使用环境变量 CARM_PRETRAIN_IMPORT_PATHS=data/learning/learning_intake_import.jsonl
- Step 3: 运行 python -m scripts.build_pretrain_dataset 或 python -m scripts.auto_train

## Top Candidates

### learning_intake:public_idea
- logic_skill: result_integration
- expected_tool: search
- quality_score: 0.9900
- prompt: 公开 agent 设计思想内化：主题=keep agent systems simple and composable。 外部观点=优先用清晰可检查的工具接口、路由和人工监督来构造 agent，而不是一开始就堆很复杂的自主链路。。 来源=Anthropic - Building Effective Agents。 请把它转写成一条适合 Mustard 离线评测或监督学习的任务，并说明验证通过阈值。

### learning_intake:learning_focus_stress
- logic_skill: evidence_judgment
- expected_tool: search
- quality_score: 0.9900
- prompt: Learning-focus routing stress 学习：样本=stress-learning-focus-evidence-routing-001。 期望工具=search，baseline=bigmodel_proxy，pretrained=search。 请把这个 evidence_judgment 误路由场景转成一条更稳健的离线监督任务，要求先检索公开证据，再判断是否能回答。

### learning_intake:learning_focus_stress
- logic_skill: evidence_judgment
- expected_tool: search
- quality_score: 0.9900
- prompt: Learning-focus routing stress 学习：样本=stress-learning-focus-evidence-routing-002。 期望工具=search，baseline=calculator，pretrained=calculator。 请把这个 evidence_judgment 误路由场景转成一条更稳健的离线监督任务，要求先检索公开证据，再判断是否能回答。

### learning_intake:learning_focus_stress
- logic_skill: evidence_judgment
- expected_tool: search
- quality_score: 0.9900
- prompt: Learning-focus routing stress 学习：样本=stress-learning-focus-evidence-routing-003。 期望工具=search，baseline=calculator，pretrained=search。 请把这个 evidence_judgment 误路由场景转成一条更稳健的离线监督任务，要求先检索公开证据，再判断是否能回答。

### learning_intake:attention_gap
- logic_skill: conflict_detection
- expected_tool: search
- quality_score: 0.9900
- prompt: AttentionFlow 学习任务：premature_release_count=1，conflict_to_verification_rate=0.2500。 当前发现 premature release 或 conflict->verification handoff 偏弱。请设计 2 条离线监督任务，要求残差必须先流向 VERIFY，之后才能 release。

### learning_intake:public_idea
- logic_skill: result_integration
- expected_tool: search
- quality_score: 0.9900
- prompt: 公开 agent 设计思想内化：主题=verbal self-feedback before policy update。 外部观点=先把失败经验总结成可复用的语言反思，再让后续回合利用这些反思，能提高自我改进的稳定性。。 来源=Reflexion。 请把它转写成一条适合 Mustard 离线评测或监督学习的任务，并说明验证通过阈值。

### learning_intake:frontier
- logic_skill: evidence_judgment
- expected_tool: search
- quality_score: 0.9900
- prompt: 公开 agent 设计学习：主题=tool-use stability under ambiguity。 当前标签=pending_label。 触发原因=frontier_zero_signal_persistence。 请基于公开资料总结可借鉴/不建议/待观察要点，并给出一个能在 Mustard 里低风险验证的实验。

### learning_intake:frontier
- logic_skill: evidence_judgment
- expected_tool: search
- quality_score: 0.9900
- prompt: 公开 agent 设计学习：主题=conflict-aware answer suppression。 当前标签=pending_label。 触发原因=frontier_zero_signal_persistence。 请基于公开资料总结可借鉴/不建议/待观察要点，并给出一个能在 Mustard 里低风险验证的实验。

### learning_intake:frontier
- logic_skill: evidence_judgment
- expected_tool: search
- quality_score: 0.9900
- prompt: 公开 agent 设计学习：主题=small reasoning model routing。 当前标签=pending_label。 触发原因=frontier_zero_signal_persistence。 请基于公开资料总结可借鉴/不建议/待观察要点，并给出一个能在 Mustard 里低风险验证的实验。

### learning_intake:public_idea
- logic_skill: result_integration
- expected_tool: search
- quality_score: 0.9900
- prompt: 公开 agent 设计思想内化：主题=reasoning and acting loop。 外部观点=把推理轨迹与工具行动交替展开，比只拟合最终答案更容易暴露和修复中间失误。。 来源=ReAct。 请把它转写成一条适合 Mustard 离线评测或监督学习的任务，并说明验证通过阈值。
