# Learning Intake Human Review Draft Sync

- mode: sync_human_review_draft_to_sheet
- default_runtime_changed: false
- default_training_admission_changed: false
- source_count: 4
- synced_count: 4
- nonempty_human_status_count: 3
- status_counts: {"edit": 2, "accept": 1, "blank": 1}
- preview_ready_to_apply_count: 3
- preview_blank_decision_count: 1

## Paths

- draft_sheet_path: D:\codes\Mustard\data\learning\candidate_pretrain_human_review_sheet.draft.jsonl
- review_sheet_path: D:\codes\Mustard\data\learning\candidate_pretrain_human_review_sheet.jsonl
- backup_path: D:\codes\Mustard\data\learning\candidate_pretrain_human_review_sheet.backup.jsonl
- preview_report_path: D:\codes\Mustard\backlog\opportunities\learning_intake_human_review_preview.md

## Notes

- 这一步只同步草稿表到正式 human review sheet，不会改 review pack。
- 同步完成后会自动刷新 preview，方便马上看到还剩几条待人工决定。