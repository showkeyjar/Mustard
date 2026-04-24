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

1. Build an offline hard-logic eval pack for conflict detection, result integration, tool boundaries, and termination judgment.
2. Prototype a VQR-inspired reasoning pattern codec: `pattern_id + residual_features + fit_score + reconstruction_notes`.
3. Compare the current latent/slot representation against the pattern/residual representation before changing runtime behavior.
4. Upgrade offline replay pretraining only after the new representation proves useful on hard-logic evals.
5. Keep desktop bridge and proactive behavior as secondary tracks until core reasoning metrics improve.
