# Reviewed Import Shadow Learning Focus Eval

- mode: controlled_shadow_eval
- default_runtime_changed: false
- default_training_admission_changed: false
- prompt_count: 7
- current_pretrained_match_rate: 0.5714
- shadow_pretrained_match_rate: 0.5714
- delta_pretrained_match_rate: +0.0000
- delta_pretrained_avg_steps: -5.0000

## Paths

- shadow_dataset_path: D:\codes\Mustard\data\learning\reviewed_import_shadow_corpus.jsonl
- eval_path: D:\codes\Mustard\data\eval\learning_focus_eval.json

## Notes

- 这是在临时 artifact 目录里重放 shadow corpus 后得到的专项对比，不会覆盖正式 pretrain artifact。
- 对比对象是当前正式 artifact 与 shadow artifact 在同一份 learning_focus_eval 上的表现。