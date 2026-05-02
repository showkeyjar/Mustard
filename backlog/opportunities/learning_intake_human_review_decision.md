# Learning Intake Human Review Decision

- mode: set_single_human_review_decision
- default_runtime_changed: false
- default_training_admission_changed: false
- sample_id: search-first-adversarial-002
- requested_status: accept
- applied_status: accept
- updated: true

## Paths

- review_sheet_path: D:\codes\Mustard\data\learning\candidate_pretrain_human_review_sheet.jsonl
- backup_path: D:\codes\Mustard\data\learning\candidate_pretrain_human_review_sheet.backup.jsonl

## Notes

- `defer` 会被落成 `pending`，用于保留候选但暂不放行。
- 修改的是 human review sheet，不会直接改正式 review pack。