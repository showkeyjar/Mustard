# BFCL v4 Evaluation Report — LLM Fallback + Disambiguation

**Date**: 2026-07-17
**Model**: CARM Router v4 (Prompt mode)
**Backend LLM**: qwen3-coder:latest (Ollama, 192.168.31.8:11434)
**BFCL Version**: bfcl-eval 2026.3.23 (BFCL V4 dataset)

## Architecture (v4)

v4 introduces two key architectural improvements over v3:

1. **LLM function selection fallback** — when CARM signal matching fails (score < threshold), use LLM to select the correct function from the full function list
2. **LLM disambiguation** — when top-2 signal scores are close (within 0.15 margin), use LLM to pick the correct one

```
User Query → CARM Signal Scoring
  ├─ score ≥ 0.4 → High confidence: use top-1 (or parallel if multiple high)
  ├─ 0.2 ≤ score < 0.4, multi-func → LLM disambiguation (top-3 candidates)
  ├─ 0.15 ≤ score < 0.4, single-func → LLM verification (accept or reject)
  └─ score < threshold → LLM fallback (select from all functions)
       └─ If LLM-selected func has signal=0.0 and no action words → reject as irrelevance
```

## v4 Changes from v3

| Change | Purpose |
|--------|---------|
| `select_function_via_llm()` | LLM selects function when signal matching fails — fixes 273 empty `[]` responses in live_multiple |
| `disambiguate_via_llm()` | LLM picks correct function when top-2 scores are close — fixes wrong_func_name errors |
| Adaptive threshold (0.15 single, 0.2 multi) | Balances false positive vs false negative |
| Irrelevance guard in LLM fallback | Rejects LLM selections with zero signal + no action words |
| Enum/pattern in param extraction prompt | Improves parameter value accuracy |
| Removed semantic verification (v3) | Proven ineffective (100% false negative in borderline range) |

## Full Results: v2 vs v3 vs v4

| Subset | n | v2 | v3 | v4 | v3→v4 |
|--------|---|----|----|----|--------|
| simple_python | 400 | 97.0% | 85.0% | 86.0% | +1.0% |
| simple_java | 100 | 92.0% | 67.0% | 53.0% | -14.0% |
| simple_javascript | 50 | 94.0% | 78.0% | 66.0% | -12.0% |
| multiple | 200 | 96.0% | 76.0% | 81.5% | +5.5% |
| parallel | 200 | 13.5% | 82.5% | 83.5% | +1.0% |
| parallel_multiple | 200 | 2.5% | 40.0% | 40.0% | +0.0% |
| irrelevance | 240 | 64.6% | 60.0% | 71.7% | +11.7% |
| live_simple | 258 | 85.7% | 58.5% | 76.0% | +17.5% |
| live_multiple | 1053 | 77.8% | 35.6% | 52.6% | +17.0% |
| live_parallel | 140 | 0.0% | 62.5% | 43.8% | -18.8% |
| live_parallel_multiple | 120 | 4.2% | 20.8% | 29.2% | +8.4% |
| live_relevance | 16 | 100.0% | 68.8% | 100.0% | +31.2% |
| live_irrelevance | 884 | 38.2% | 68.1% | 42.5% | -25.6% |
| **Weighted Average** | **3861** | **59.7%** | **58.0%** | **58.3%** | **+0.3%** |

## Key Improvements

### live_multiple: 35.6% → 52.6% (+17.0%)

**Root cause of v3 failure**: 273/1053 (26%) queries returned empty `[]` because natural language queries (e.g. "how can i cook steak Indian style") had zero token overlap with function names, falling below the 0.2 threshold.

**v4 fix**: LLM fallback selects the correct function when signal matching fails. Verified: "cook steak Indian style" → signal score 0.00 → LLM fallback → `cookbook.search_recipe(keyword="steak", cuisine="Indian")`.

### live_simple: 58.5% → 76.0% (+17.5%)

Same mechanism — LLM fallback handles NL queries that don't mention function names.

### irrelevance: 60.0% → 71.7% (+11.7%)

**v4 fix**: Single-function medium-score (0.15-0.4) queries now go through LLM verification. BFCL irrelevance tests provide a single function that seems related but isn't (e.g. `calculate_compound_interest` for "Calculate prime factors of 100"). LLM verification correctly rejects these.

### live_relevance: 68.8% → 100.0% (+31.2%)

LLM fallback correctly identifies relevant functions for all 16 live_relevance queries.

## Remaining Issues

### live_irrelevance: 68.1% → 42.5% (-25.6%)

510/884 queries returned a function call when they should return `[]`. Live_irrelevance uses generic functions like `requests.get` or `get_current_weather` that LLM tends to select for any weather/API-related query.

The LLM cannot reliably distinguish "I want to see weather data for coordinates X" (irrelevant — user wants an API, not a function call) from "What's the weather in Boston?" (relevant — should call get_current_weather).

### live_parallel: 62.5% → 43.8% (-18.8%)

Parallel detection heuristics (separator-based) are less reliable with LLM fallback in the pipeline. The LLM may select functions without properly detecting parallel intent.

### simple_java/javascript: -14%/-12%

Parameter type formatting issues with Java/Javascript syntax. The threshold change from 0.05 to 0.15 may also cause some borderline cases to go through unnecessary LLM verification.

## Error Analysis (live_multiple, 100-sample)

| Error Type | Count | % of Errors |
|-----------|-------|-------------|
| value_error:string | 24 | 45% |
| multiple_function_checker:wrong_count | 16 | 30% |
| simple_function_checker:wrong_func_name | 12 | 23% |
| value_error:dict_key | 1 | 2% |

- **value_error:string (45%)**: Function selected correctly, but parameter values don't match BFCL's expected values (e.g. "Family Therapist" vs expected "Family Counselor")
- **wrong_count (30%)**: Returned wrong number of function calls (over/under-detected parallel calls)
- **wrong_func_name (23%)**: LLM disambiguation selected the wrong function among similar candidates

## Next Steps

1. **Improve live_irrelevance**: Add domain-specific irrelevance detection for generic functions (`requests.get`, `get_current_weather`)
2. **Improve parameter extraction**: Use function schema constraints (enum, pattern) more aggressively
3. **Parallel detection**: Replace separator heuristics with LLM-based parallel detection
4. **simple_java/javascript**: Investigate parameter type formatting issues
5. **multi_turn evaluation**: Still untested (requires multi-turn conversation support)
