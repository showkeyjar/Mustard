# Two-Stage CARM Design

## Summary

The project is being restructured into two explicit loops:

1. A pretraining loop that replays successful episodes, review signals, and prior evolution signals into stable bootstrap artifacts.
2. An online evolution loop that accepts structured user signals and nudges the live policy/core with bounded updates.

## Architecture

`AgentRunner` remains the runtime entrypoint, but it no longer treats all learning as one undifferentiated stream. The new `OfflinePretrainer` builds an artifact bundle under `data/pretrain/`, while `OnlineEvolutionManager` owns user-facing adaptation state under `data/evolution/`. Runtime inference consults evolution guidance before choosing slots or tools, which means user-confirmed goals and preferences can shape behavior immediately without waiting for many natural-language episodes to accumulate.

## Signal Protocol

The online loop now supports explicit fields for:

- `goal`
- `preferred_tool`
- `preferred_slot`
- `reward`
- `learn`
- `correction`

This makes feedback directional. A goal confirmation can bias planning. A tool preference can sharpen tool routing. A negative or non-learning signal can damp updates and block accidental absorption of noisy traces.

## Rollout Plan

- Keep current MVP behavior available through existing runner and bridge scripts.
- Add pretraining script and config first.
- Route bridge feedback and interactive commands through structured evolution signals.
- Preserve auditability by logging every signal and storing evolution state separately from policy/core weights.

## Low-Cost Data Plan

For generic task understanding and planning, the cheapest sustainable source is not manual annotation. The system now supports a data factory based on:

- prompt templates for common task types
- programmatic annotation into `expected_tool`, `target_slot`, `action_items`, `unknowns`, and `evidence_targets`
- quality scoring for filtering
- replay of those synthetic-but-structured samples during pretraining

This keeps marginal data cost low while still teaching the model to recognize task shape and planning structure.
