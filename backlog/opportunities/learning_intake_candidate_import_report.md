# Learning Intake Candidate Import Report

- mode: reviewed_export_only
- default_runtime_changed: false
- default_training_admission_changed: false
- review_total: 4
- approved_count: 3
- pending_count: 1
- rejected_count: 0
- status_counts: {"edit": 2, "accept": 1, "pending": 1}

## Paths

- import_path: D:\codes\Mustard\data\learning\candidate_pretrain_import.jsonl

## Gate

- 只有 review_status=accept/edit 的样本会进入 import 候选。
- pending 样本保持在 review pack，不进入离线构建。