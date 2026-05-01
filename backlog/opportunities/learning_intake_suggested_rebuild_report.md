# Learning Intake Suggested Rebuild Report

- mode: simulated_review_rebuild
- default_runtime_changed: false
- default_training_admission_changed: false
- source_review_count: 4
- suggested_accept_count: 2
- suggested_edit_count: 2
- suggested_import_count: 4
- base_sample_count: 184
- shadow_sample_count: 187

## Paths

- suggested_review_pack: D:\codes\Mustard\data\learning\candidate_pretrain_suggested_review_pack.jsonl
- suggested_import: D:\codes\Mustard\data\learning\candidate_pretrain_suggested_import.jsonl
- suggested_shadow_corpus: D:\codes\Mustard\data\learning\candidate_pretrain_suggested_corpus.jsonl

## Notes

- 这是按系统推荐状态生成的影子结果，不会改动原始 review pack。
- 只有 suggested accept/edit 会进入 suggested import 和 shadow corpus。