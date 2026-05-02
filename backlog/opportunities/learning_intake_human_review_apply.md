# Learning Intake Human Review Apply Report

- mode: apply_human_review_sheet
- default_runtime_changed: false
- default_training_admission_changed: false
- review_total: 4
- applied_count: 4
- accepted_count: 1
- edited_count: 2
- rejected_count: 0
- pending_count: 1

## Paths

- review_pack_path: D:\codes\Mustard\data\learning\candidate_pretrain_review_pack.jsonl
- backup_path: D:\codes\Mustard\data\learning\candidate_pretrain_review_pack.backup.jsonl

## Notes

- 只有 `human_review_status` 非空的行会覆盖原 review pack。
- helper 字段不会写回正式 review pack。