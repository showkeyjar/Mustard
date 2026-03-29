# Quality Stabilization Report

- row_count: 20
- high_signal_count: 4
- bigmodel_proxy_mismatch_count: 3

## Top Separation Rows
- repair-comparison-005 | logic_skill=comparison | baseline=bigmodel_proxy | pretrained=search | expected=search | score=6
- repair-conflict_detection-019 | logic_skill=conflict_detection | baseline=bigmodel_proxy | pretrained=search | expected=search | score=6
- repair-conflict_detection-020 | logic_skill=conflict_detection | baseline=bigmodel_proxy | pretrained=search | expected=search | score=6
- real-mixed | logic_skill=tool_selection | baseline=code_executor | pretrained=calculator | expected=calculator | score=3
- candidate-379152 | logic_skill=comparison | baseline=search | pretrained=search | expected=search | score=1
- real-conflict | logic_skill=conflict_detection | baseline=search | pretrained=search | expected=search | score=1
- real-exec-summary | logic_skill=result_integration | baseline=bigmodel_proxy | pretrained=bigmodel_proxy | expected=bigmodel_proxy | score=1
- real-multi-source-summary | logic_skill=result_integration | baseline=bigmodel_proxy | pretrained=bigmodel_proxy | expected=bigmodel_proxy | score=1
- repair-comparison-001 | logic_skill=comparison | baseline=search | pretrained=search | expected=search | score=1
- repair-comparison-002 | logic_skill=comparison | baseline=search | pretrained=search | expected=search | score=1
