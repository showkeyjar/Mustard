# CARM v1 Spec

## Goal

Provide a runnable architecture prototype for a compact reasoning-centric agent that prioritizes:

- structured working memory
- online action selection
- verification before answer
- external tool orchestration
- inference-time learning from dialogue

## Current MVP choices

- Heuristic encoder and core, plus a lightweight online-trainable policy head
- Text-backed memory slots for interpretability
- Deterministic tools with predictable outputs for testability
- Single-process agent loop for simplicity, but with explicit separation between pretraining and online evolution
- Persistent experience store and local policy state
- Structured user evolution signals for goals, corrections, and tool preferences

## Next steps

1. Upgrade offline replay pretraining into a reproducible batch-training pipeline.
2. Replace heuristic core with a recurrent or state-space module.
3. Expand user evolution signals into richer reward shaping and safety constraints.
4. Add persistent multi-session working memory and evaluation harness.
