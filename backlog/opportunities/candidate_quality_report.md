# Candidate Quality Report

- total_candidates: 9
- filtered_candidates: 2
- dropped_candidates: 7
- existing_prompt_count: 12
- target_prompt_count: 20
- total_gap: 8

## Drop Reason Counts
- observer_learning_noise: 7
- tool_not_allowed:code_executor: 2

## Existing Coverage By Logic Skill
- comparison: 1
- conflict_detection: 1
- evidence_judgment: 2
- result_integration: 2
- step_planning: 2
- tool_selection: 4

## Filtered Candidate Coverage By Logic Skill
- comparison: 1
- tool_selection: 1

## Gap By Logic Skill
- comparison: gap=1
- conflict_detection: gap=1
- evidence_judgment: gap=0
- result_integration: gap=0
- step_planning: gap=0
- tool_selection: gap=0

## Drop Details
- candidate-682361 | tool=code_executor | logic_skill=tool_selection | reasons=observer_learning_noise,tool_not_allowed:code_executor
- candidate-957657 | tool=search | logic_skill=tool_selection | reasons=observer_learning_noise
- candidate-396298 | tool=code_executor | logic_skill=tool_selection | reasons=observer_learning_noise,tool_not_allowed:code_executor
- candidate-668772 | tool=search | logic_skill=tool_selection | reasons=observer_learning_noise
- candidate-787023 | tool=search | logic_skill=tool_selection | reasons=observer_learning_noise
- candidate-445372 | tool=search | logic_skill=tool_selection | reasons=observer_learning_noise
- candidate-859782 | tool=search | logic_skill=tool_selection | reasons=observer_learning_noise
