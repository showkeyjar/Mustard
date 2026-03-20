# CARM MVP

Compact Agentic Reasoning Model (CARM) online-learning prototype.

This repository contains a runnable scaffold for a small "reasoning core + working memory + tool bus" agent. The current version is hybrid:

- heuristic priors provide initial behavior
- a lightweight online action head updates after each dialogue
- dialogue traces are persisted as reusable experience memory

## What is included

- Structured agent state and action protocol
- Fixed-slot working memory board
- Online policy/controller loop with persistent weights
- Adaptive concept learner that infers action/tool preferences from prior dialogues
- Adaptive reasoning core that learns when to form `PLAN`, `HYP`, or `DRAFT`
- Structured intermediate schemas with explicit action items, unknowns, evidence targets, and confidence bands
- Mock tools for search, calculator, code execution, and large-model proxy
- Experience replay store for dialogue summaries and action rewards
- Runnable local entrypoint
- Focused tests for memory flow and agent loop

## Quick start

```powershell
python -m scripts.run_local_agent "比较 PostgreSQL 和 MySQL 在中小团队里的适用性"
python -m scripts.interactive_agent
python -m scripts.migrate_experience_schema
python -m scripts.consolidate_reviews
python -m scripts.apply_slow_path_actions
python -m scripts.rollback_runtime_controls
python -m scripts.evaluate_control_versions
python -m scripts.judge_control_rollout
python -m scripts.run_control_cycle
python -m scripts.system_status
python -m unittest discover -s tests -v
```

`python -m scripts.run_control_cycle` 默认会从 [control_cycle.json](d:/codes/Mustard/configs/control_cycle.json) 读取采样任务集。任务现在支持结构化字段：`id / prompt / tag / expected_tool`。`judge_control_rollout` 会把这些采样结果里的工具命中率作为辅助闸门，且支持在配置里按 `tag` 设置不同阈值，避免总体分数看起来正常但某类任务已经退化。也可以用环境变量 `CARM_CONTROL_PROMPT_SET` 选择其他任务集，或直接在命令行后面传 prompt 覆盖配置。

## Status

This is an online-learning scaffold. The current step updates the action head during inference-time interaction, learns concept-level action/tool affinities from successful dialogues, trains a lightweight adaptive reasoning core that learns slot preferences for `PLAN`, `HYP`, and `DRAFT`, stores those states in tighter training-oriented schemas, and versions runtime-control changes so slow-path tuning can be audited and rolled back. The next step is to replace the lightweight core with a recurrent or state-space module and to add trajectory-supervised training.
