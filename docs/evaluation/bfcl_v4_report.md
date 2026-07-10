# CARM Router — BFCL V4 Official Evaluation Report

**Date**: 2026-07-10
**Model**: CARM Router (Prompt mode)
**Backend LLM**: qwen3-coder:latest (Ollama, 192.168.31.8:11434)
**BFCL Version**: bfcl-eval 2026.3.23 (BFCL V4)
**Temperature**: 0.001

## 1. Evaluation Setup

### Architecture

```
BFCL CLI → OpenAI Chat Completions API → CARM BFCL Server (:11400) → Ollama (qwen3-coder)
```

CARM is registered as a prompting-mode model in BFCL's `model_config.py` using the `OpenAICompletionsHandler`. BFCL injects function documentation into the system prompt and expects the model to return function calls in Python syntax: `[func_name(param=value)]`.

### Reproducibility

- **CARM BFCL Server**: `scripts/carm_bfcl_server.py` — OpenAI-compatible API wrapper
- **Model registration**: `carm-router` in BFCL's `MODEL_CONFIG_MAPPING` (is_fc_model=False)
- **Evaluation script**: `scripts/run_bfcl_eval.py`
- **Confidence intervals**: `scripts/compute_ci.py` (Wilson score interval, 95% CI)

## 2. BFCL V4 Official Results

### Single-Turn Categories

| Category | Accuracy | 95% CI | n |
|----------|----------|--------|---|
| simple_python | 96.50% | [94.2%, 97.9%] | 400 |
| multiple | 95.00% | [91.0%, 97.3%] | 200 |
| parallel | 13.50% | [9.4%, 18.9%] | 200 |
| parallel_multiple | 3.50% | [1.7%, 7.0%] | 200 |
| irrelevance | 66.25% | [60.1%, 71.9%] | 240 |
| live_simple | 86.05% | [81.3%, 89.7%] | 258 |
| live_multiple | 77.97% | [75.4%, 80.4%] | 1053 |
| live_parallel | 0.00% | [0.0%, 19.4%] | 16 |
| live_parallel_multiple | 4.17% | [0.7%, 20.2%] | 24 |
| live_irrelevance | 38.24% | [35.1%, 41.5%] | 882 |
| live_relevance | 93.75% | [67.2%, 96.9%] | 18 |

### Multi-Turn Categories

| Category | Accuracy | 95% CI | n |
|----------|----------|--------|---|
| multi_turn_base | 0.00% | [0.0%, 1.9%] | 200 |
| multi_turn_miss_func | N/A (timeout) | — | 200 |
| multi_turn_miss_param | N/A (timeout) | — | 200 |
| multi_turn_long_context | N/A (timeout) | — | 200 |

### Summary Metrics

| Metric | Value |
|--------|-------|
| Overall Accuracy | 16.56% |
| Non-Live Accuracy | 36.04% |
| Live Accuracy | 77.28% |
| Relevance Detection | 93.75% |
| Irrelevance Detection | 52.24% |
| Latency Mean | 3.45s |
| Latency 95th Percentile | 5.54s |
| Cost | $0.00 (open-source model) |

## 3. Analysis

### Strengths

1. **Simple function selection (96.5%)**: Near-perfect on single-function calls — CARM's signal-based routing accurately identifies which tool to invoke.
2. **Multiple function selection (95.0%)**: Strong on choosing the correct single function from a set when multiple are available.
3. **Live relevance detection (93.75%)**: Correctly identifies when a function should be called vs. answering directly.

### Weaknesses

1. **Parallel function calls (0-13.5%)**: The model struggles to generate `[func1(), func2()]` multi-call format. This is a prompting-mode limitation — qwen3-coder tends to explain in natural language rather than returning structured function calls.
2. **Irrelevance detection (38-66%)**: The model often attempts to call a function even when the query is irrelevant. CARM's signal detection could be improved with stricter irrelevance patterns.
3. **Multi-turn (0%)**: The model returns natural language instead of function calls in multi-turn conversations. The BFCL multi-turn format requires strict `[func()]` syntax, which prompting-mode models find difficult.
4. **Live irrelevance (38.2%)**: Lower than non-live irrelevance (66.25%), suggesting the model is more likely to force function calls on real-world queries.

### Root Causes

- **Parallel calls**: BFCL's prompting format expects `[func1(param=val), func2(param=val)]` in a single response. qwen3-coder often returns only the first function or describes both in prose.
- **Multi-turn**: Each turn expects `[func()]` format response. The model instead returns explanatory text, which BFCL's parser cannot decode as function calls.
- **Irrelevance**: CARM's signal detection triggers on broad keyword matches (e.g., "查" in irrelevant contexts routes to search).

## 4. Comparison with Internal Evaluation

| Dimension | BFCL Official | Internal (D1-D5) |
|-----------|--------------|------------------|
| D1 Function Selection | 96.5% (simple) | 92.8% (BFCL simple mapped to 4 tools) |
| D2 Parameter Extraction | N/A | 77.9% |
| D3 End-to-End Execution | N/A | 88.9% |
| D4 Multi-Turn Dialogue | 0% (BFCL format) | 88.2% (CARM native) |
| D5 Robustness | N/A | 100% |

The BFCL multi-turn 0% vs. internal D4 88.2% gap is explained by format mismatch: CARM's internal multi-turn evaluation uses CARM's own session memory and tool routing, while BFCL expects strict `[func()]` format responses.

## 5. How to Reproduce

### Prerequisites

```bash
conda create -n BFCL python=3.10 -y
conda activate BFCL
pip install bfcl-eval soundfile httpx
```

### Register CARM in BFCL

Add to `model_config.py` in the `local_inference_model_map`:
```python
"carm-router": ModelConfig(
    model_name="carm-router",
    display_name="CARM Router (Prompt)",
    url="https://github.com/anthropics/Mustard",
    org="Mustard",
    license="mit",
    model_handler=OpenAICompletionsHandler,
    is_fc_model=False,
    underscore_to_dot=False,
),
```

### Run Evaluation

```bash
# 1. Start CARM API server
python scripts/carm_bfcl_server.py --port 11400

# 2. Run BFCL generate + evaluate
set OPENAI_API_KEY=dummy
set OPENAI_BASE_URL=http://localhost:11400/v1
bfcl generate --model carm-router --test-category simple_python
bfcl evaluate --model carm-router --test-category simple_python

# 3. Compute confidence intervals
python scripts/compute_ci.py --bfcl
```

### Confidence Intervals

All results include 95% Wilson score confidence intervals. Key observations:
- **simple_python** (n=400): CI [94.2%, 97.9%] — statistically significant, tight interval
- **live_multiple** (n=1053): CI [75.4%, 80.4%] — largest sample, most precise
- **live_parallel** (n=16): CI [0.0%, 19.4%] — too few samples for meaningful CI
- **live_relevance** (n=18): CI [67.2%, 96.9%] — very wide CI, needs more samples

## 6. Next Steps

1. **Improve parallel call accuracy**: Post-process model output to detect multiple function intents and format as `[func1(), func2()]`
2. **Improve irrelevance detection**: Tighten CARM's signal matching to reduce false positives
3. **Run full multi-turn evaluation**: Implement parallel API calls to reduce wall time (current: ~1hr per category)
4. **Expand D2-D5 test sets**: Generate ≥500 programmatic test cases per dimension
5. **Docker packaging**: Create reproducible Docker environment with all dependencies
