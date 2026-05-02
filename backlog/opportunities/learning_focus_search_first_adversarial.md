# Learning Focus Search-First Adversarial Eval

- mode: controlled_search_first_adversarial_eval
- default_runtime_changed: false
- default_training_admission_changed: false
- target_failure_count: 3
- prompt_count: 6
- current_pretrained_match_rate: 0.8333
- shadow_pretrained_match_rate: 0.8333
- delta_pretrained_match_rate: +0.0000

## Target IDs

- learning-focus-004, learning-focus-005, learning-focus-006

## Notes

- 这是一组专门围绕 search-first evidence_judgment 误路由构造的对抗评测。
- 对比对象是当前正式 artifact 与 reviewed-import shadow artifact 在同一批题上的表现。