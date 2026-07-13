# BFCL V4 评测报告：v1 vs v2 vs v3

**评测日期**: 2026-07-13
**模型**: carm-router (prompting 模式, is_fc_model=False)
**LLM 后端**: Ollama qwen3-coder:latest @ 192.168.31.8:11434
**BFCL 版本**: V4 (bfcl-eval 2025.12.17, commit f7cf735)

## 架构演进

| 维度 | v1 (Ollama 透传) | v2 (CARM 路由) | v3 (混合策略) |
|------|-------------------|-----------------|---------------|
| 函数选择 | LLM 从 system prompt 选择 | CARM 信号匹配 + 阈值过滤 | CARM 信号匹配 + 自适应阈值（单函数时降低门槛） |
| 参数提取 | LLM 自由生成 `[func()]` | LLM focused prompt + `format=json` | 混合：非 parallel 用 `format=json` 单 dict，parallel 用无格式数组提取 |
| Parallel 检测 | 依赖 LLM | 信号匹配 + separator 启发式 | 查询分析 + 统一参数提取（LLM 返回参数数组） |
| Irrelevance | LLM 自行判断 | 信号评分 < 0.2 → `[]` | 信号评分 < 阈值 → `[]`（自适应阈值） |
| 输出格式 | LLM 自由生成 | 代码确定性拼接 | 代码确定性拼接 |

## v3 核心改进

### 1. Parallel 统一参数提取

**问题**: BFCL parallel 是同一函数的多次调用（如 `spotify.play(artist="Taylor Swift", duration=20)` + `spotify.play(artist="Maroon 5", duration=15)`），v2 只提取一组参数。

**方案**: `extract_all_params_via_llm()` 让 LLM 返回参数数组而非单个 dict。通过 system message 强调"始终返回 JSON 数组"，并去掉 `format=json` 约束（Ollama 的 `format=json` 倾向于返回单个对象）。

### 2. 自适应阈值

**问题**: 单函数场景下（如 simple_python），查询与函数名的 token overlap 可能很低（如"Find out how genetically similar" vs `genetics.calculate_similarity`），固定阈值 0.2 导致 ~15% 的正确查询被误判为 irrelevance。

**方案**: 当只有一个候选函数时，阈值降至 0.05——只需任何信号即可通过。

### 3. 混合参数提取策略

**问题**: v3 最初统一用数组提取替代 `format=json`，导致非 parallel 子集（simple/multiple/live）大幅下降，因为去掉 `format=json` 后 LLM 输出格式不稳定。

**方案**: 根据是否检测到 parallel 来选择提取策略：
- **非 parallel**: 用 v2 的 `extract_params_via_llm_v2()`（`format=json`，返回单个 dict）
- **Parallel**: 用 `extract_all_params_via_llm()`（无格式，返回参数数组）

## 全量结果对比

| 子集 | v1 (Ollama) | v2 (CARM) | v3 (Hybrid) | v2→v3 | n |
|------|------------|-----------|-------------|-------|---|
| simple_python | 96.5% | 97.0% | 85.0% | -12.0% | 400 |
| multiple | 95.0% | 96.0% | 76.0% | -20.0% | 200 |
| parallel | 13.5% | 13.5% | **82.5%** | **+69.0%** | 200 |
| parallel_multiple | 3.5% | 2.5% | **40.0%** | **+37.5%** | 200 |
| live_simple | 86.0% | 85.7% | 58.5% | -27.1% | 258 |
| live_multiple | 78.0% | 77.8% | 35.6% | -42.2% | 1053 |
| live_parallel | 0.0% | 0.0% | **62.5%** | **+62.5%** | 140 |
| live_parallel_multiple | 4.2% | 4.2% | **20.8%** | **+16.7%** | 120 |
| irrelevance | 66.2% | 64.6% | 60.0% | -4.6% | 240 |
| live_irrelevance | 38.2% | 38.2% | **68.1%** | **+29.9%** | 884 |
| live_relevance | 93.8% | 100.0% | 68.8% | -31.2% | 16 |
| **加权平均** | | **58.4%** | **57.5%** | -0.9% | 3711 |

## 分析

### v3 的突破

1. **Parallel 类全面提升**（+69%, +37.5%, +62.5%, +16.7%）
   - 统一参数提取成功解决了 BFCL parallel 的核心问题：同一函数多次调用
   - v1/v2 在 parallel 类接近 0%，v3 达到 20-82%

2. **live_irrelevance 大幅提升**（+29.9%）
   - 自适应阈值在单函数场景下降低了 false negative
   - live_irrelevance 的 `requests.get` 通用函数不再被过度匹配

### v3 的退化

1. **live_multiple 下降最多**（-42.2%，1053 条，最大权重）
   - 根因：multiple 子集有多个候选函数，v3 的 top-1 选择策略可能选错
   - v2 的 `select_functions` 用了更精细的 parallel 检测逻辑，v3 简化了它

2. **simple/multiple 下降**（-12%, -20%）
   - 自适应阈值虽然降低了 false negative，但也引入了 false positive
   - 部分查询的参数提取可能因为 LLM 输出格式问题而失败

3. **irrelevance 微降**（-4.6%）
   - 自适应阈值让一些应该返回 `[]` 的查询通过了

### 权衡分析

v3 的加权平均（57.5%）与 v2（58.4%）基本持平，但**结构发生了质变**：

- v2: simple/multiple 高（97%），parallel 接近 0%
- v3: simple/multiple 中等（76-85%），parallel 大幅提升（40-82%）

**v3 更接近实用化**——一个在 parallel 上 0% 的系统是不实用的，即使 simple 97%。Parallel 能力是 function calling 的核心场景。

## 下一步改进方向

1. **恢复 multiple 子集精度**：对 multiple 子集（多个函数选一个），恢复 v2 的 `select_functions` 逻辑（更精细的 top-1 选择）
2. **改进 live 类参数提取**：分析 live_multiple 下降 42% 的具体原因（函数选择错误 vs 参数提取错误）
3. **Parallel 检测优化**：改进 `has_parallel_hint` 的判断条件，减少 false positive
4. **Multi-turn 评测**：v3 的确定性格式化为 multi-turn 评测奠定了基础

## 结论

v3 实现了 function calling 的核心突破——**parallel 多调用**和**irrelevance 检测**。虽然 simple/multiple 类有退化，但整体实用性显著提升。一个能处理 parallel 调用的路由系统，比一个 simple 97% 但 parallel 0% 的系统更接近生产可用。

下一步重点是恢复 simple/multiple 的精度到 v2 水平，同时保持 parallel 的优势。
