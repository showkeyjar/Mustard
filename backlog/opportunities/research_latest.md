# Research Artifact (Actionable)

## Why this round matters
- Top gap remains eval_coverage_too_low; current_count=12, target=20, gap=8
- stagnation_rounds=90; frontier_observation_count=0

## New findings (not template text)
- pretrained_match_rate=1.0000, baseline_match_rate=0.9167, delta=+0.0833
- mismatch_case_count=0
- candidate_pipeline_snapshot:
  - total_candidates: 9
  - filtered_candidates: 2
  - dropped_candidates: 7

## Concrete mismatch cases
- none (all current prompts matched for pretrained runner)

## Root-cause hypothesis
- Low information value came from repetitive observer-learning candidates entering the pool.
- Current remaining gap is mainly coverage (count) and tool-label stability for added candidates.

## Next 24h execution plan
- Step1: add >=4 high-quality non-observer prompts (manual curation) into configs/real_prompt_eval.json
- Step2: rerun python -m scripts.evaluate_real_prompts and compare mismatch_case_count
- Step3: if mismatch_case_count > 0, patch tool-label mapping rules before next merge

## Acceptance / Failure
- acceptance: prompt_count>=12 this iteration AND mismatch_case_count not worse
- failure: prompt_count increased but mismatch_case_count rises or match_rate drops below 0.90

## relative_to_last_round
- switched from static template to concrete mismatch + quality snapshot + executable next-24h plan
