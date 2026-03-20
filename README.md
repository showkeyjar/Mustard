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
- Mock tools for search, calculator, code execution, and large-model proxy
- Experience replay store for dialogue summaries and action rewards
- Runnable local entrypoint
- Focused tests for memory flow and agent loop

## Quick start

```powershell
python -m scripts.run_local_agent "比较 PostgreSQL 和 MySQL 在中小团队里的适用性"
python -m scripts.interactive_agent
python -m unittest discover -s tests -v
```

## Status

This is an online-learning scaffold. The current step updates the action head during inference-time interaction. The next step is to replace the heuristic core with a trainable recurrent or state-space module and to add trajectory-supervised training.
