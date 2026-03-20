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
- Single-process agent loop for simplicity
- Persistent experience store and local policy state

## Next steps

1. Add explicit trajectory dataset export and offline replay training script.
2. Replace heuristic core with a recurrent or state-space module.
3. Add reward shaping based on user feedback instead of internal heuristics only.
4. Add persistent multi-session working memory and evaluation harness.
