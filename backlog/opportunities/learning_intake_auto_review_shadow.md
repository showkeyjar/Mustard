# Learning Intake Auto Review Shadow

- mode: auto_review_shadow
- default_runtime_changed: false
- default_training_admission_changed: false
- review_total: 5
- auto_accept_count: 2
- auto_edit_count: 2
- auto_pending_count: 1
- auto_import_count: 4
- shadow_sample_count: 188
- top_priority_sample_id: search-first-adversarial-002

## Paths

- auto_review_pack_path: D:\codes\Mustard\data\learning\candidate_pretrain_auto_review_shadow.jsonl
- auto_import_path: D:\codes\Mustard\data\learning\candidate_pretrain_auto_import_shadow.jsonl
- auto_shadow_corpus_path: D:\codes\Mustard\data\learning\candidate_pretrain_auto_shadow_corpus.jsonl

## Notes

- 这是自动审阅的影子通道，不会改正式 candidate_pretrain_review_pack.jsonl。
- 只有 packet 里建议为 accept/edit 的候选会进入 auto import 和 auto shadow corpus。