# Decide Top Human Review

- mode: decide_top_human_review
- default_runtime_changed: false
- default_training_admission_changed: false
- selected_sample_id: search-first-adversarial-002
- selected_status: accept
- selected_source_type: learning_intake:search_first_adversarial_failure
- priority_score: 134.0

## Notes

- 该命令会读取当前 human review sheet 中还未填写的候选，优先选择最高 priority_score 的一条。
- 若 suggested_review_status=pending，则会按 defer 处理，实际写回 pending。