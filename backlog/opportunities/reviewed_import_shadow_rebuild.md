# Reviewed Import Shadow Rebuild

- mode: controlled_shadow_rebuild
- default_runtime_changed: false
- default_training_admission_changed: false
- base_sample_count: 184
- approved_review_count: 3
- import_prompt_count: 3
- shadow_sample_count: 187
- added_count: 3
- surviving_import_count: 3
- deduped_import_count: 0
- approved_source_counts: {"learning_intake:learning_focus_stress": 3}

## Paths

- import_path: D:\codes\Mustard\data\learning\candidate_pretrain_import.jsonl
- shadow_corpus_path: D:\codes\Mustard\data\learning\reviewed_import_shadow_corpus.jsonl
- shadow_review_pack_path: D:\codes\Mustard\data\learning\reviewed_import_shadow_review_pack.jsonl

## Notes

- 这是 Human Gate 通过后的受控影子重建，不会覆盖正式 pretrain corpus。
- 只有 review_status=accept/edit 且进入 import 的样本会参与这次重建。