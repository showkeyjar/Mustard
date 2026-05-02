# Learning Intake Candidate Dataset Report

- mode: preview_only
- default_runtime_changed: false
- default_training_admission_changed: false
- base_sample_count: 184
- selected_candidate_count: 5
- merged_sample_count: 188
- selected_source_counts: {"learning_intake:search_first_adversarial_failure": 1, "learning_intake:learning_focus_stress": 3, "learning_intake:attention_gap": 1}

## Paths

- candidate_dataset: data\learning\candidate_pretrain_corpus.jsonl
- candidate_review_pack: data\learning\candidate_pretrain_review_pack.jsonl

## Recommended Next Step

- Step 1: 审阅 candidate review pack，确认 learning_focus_stress / attention_gap / search_first_adversarial_failure 样本是否值得进入正式离线构建。
- Step 2: 若通过人工审阅，再将对应 import path 用于 build_pretrain_dataset 或 auto_train。