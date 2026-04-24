# Quality Stabilization Report

- row_count: 20
- high_signal_count: 1
- bigmodel_proxy_mismatch_count: 1

## Top Separation Rows
- repair-comparison-005 | logic_skill=comparison | baseline=bigmodel_proxy | pretrained=bigmodel_proxy | expected=search | score=3
- candidate-379152 | logic_skill=comparison | baseline=search | pretrained=search | expected=search | score=1
- real-conflict | logic_skill=conflict_detection | baseline=search | pretrained=search | expected=search | score=1
- real-exec-summary | logic_skill=result_integration | baseline=bigmodel_proxy | pretrained=bigmodel_proxy | expected=bigmodel_proxy | score=1
- real-multi-source-summary | logic_skill=result_integration | baseline=bigmodel_proxy | pretrained=bigmodel_proxy | expected=bigmodel_proxy | score=1
- repair-comparison-001 | logic_skill=comparison | baseline=search | pretrained=search | expected=search | score=1
- repair-comparison-002 | logic_skill=comparison | baseline=search | pretrained=search | expected=search | score=1
- repair-comparison-003 | logic_skill=comparison | baseline=search | pretrained=search | expected=search | score=1
- repair-comparison-004 | logic_skill=comparison | baseline=search | pretrained=search | expected=search | score=1
- repair-conflict_detection-017 | logic_skill=conflict_detection | baseline=search | pretrained=search | expected=search | score=1
