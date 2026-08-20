# Quality Stabilization Report

- row_count: 63
- high_signal_count: 5
- bigmodel_proxy_mismatch_count: 6

## Top Separation Rows
- stress-conflict-missing-evidence-004 | logic_skill=conflict_detection | baseline=bigmodel_proxy | pretrained=search | expected=search | score=6
- repair-missing-evidence-decision-003 | logic_skill=conflict_detection | baseline=bigmodel_proxy | pretrained=calculator | expected=search | score=3
- stress-conflict-authority-002 | logic_skill=conflict_detection | baseline=bigmodel_proxy | pretrained=bigmodel_proxy | expected=search | score=3
- stress-conflict-missing-evidence-001 | logic_skill=conflict_detection | baseline=bigmodel_proxy | pretrained=bigmodel_proxy | expected=search | score=3
- stress-conflict-missing-evidence-003 | logic_skill=conflict_detection | baseline=bigmodel_proxy | pretrained=bigmodel_proxy | expected=search | score=3
- real-git-triage | logic_skill=evidence_judgment | baseline=bigmodel_proxy | pretrained=bigmodel_proxy | expected=search | score=2
- candidate-379152 | logic_skill=comparison | baseline=search | pretrained=search | expected=search | score=1
- guard-conflict-fenqi-001 | logic_skill=conflict_detection | baseline=search | pretrained=search | expected=search | score=1
- guard-formal-synthesis-conflict-001 | logic_skill=result_integration | baseline=bigmodel_proxy | pretrained=bigmodel_proxy | expected=bigmodel_proxy | score=1
- guard-or-mapping-001 | logic_skill=comparison | baseline=search | pretrained=search | expected=search | score=1
