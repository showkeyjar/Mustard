# BFCL V4 评测报告：v1 (Ollama 透传) vs v2 (CARM 路由)

**评测日期**: 2026-07-10
**模型**: carm-router (prompting 模式, is_fc_model=False)
**LLM 后端**: Ollama qwen3-coder:latest @ 192.168.31.8:11434
**BFCL 版本**: V4 (bfcl-eval 2025.12.17, commit f7cf735)

## 架构对比

| 维度 | v1 (Ollama 透传) | v2 (CARM 路由) |
|------|-------------------|-----------------|
| 函数选择 | LLM 直接从 system prompt 中选择 | CARM 信号匹配评分（函数名/描述/参数名 token overlap + action verbs） |
| 参数提取 | LLM 生成完整 `[func()]` 格式 | LLM 只做参数提取（focused prompt + format=json） |
| 输出格式 | LLM 自由生成，经常不遵守格式 | 代码确定性拼接 `[func_name(param=value)]` |
| Irrelevance 处理 | LLM 自行判断，false positive 高 | 信号评分低于阈值(0.2) → 返回 `[]` |
| Parallel 检测 | 依赖 LLM 输出多个 `[func()]` | 信号匹配检测多个高分函数 |

## 结果对比

| 子集 | v1 准确率 | v2 准确率 | 变化 | n |
|------|----------|----------|------|---|
| simple_python | 96.50% | **97.00%** | +0.50% | 400 |
| multiple | 95.00% | **96.00%** | +1.00% | 200 |
| live_simple | 86.05% | **85.66%** | -0.39% | 258 |
| live_multiple | 77.97% | **77.78%** | -0.19% | 1053 |
| live_relevance | 93.75% | **100.00%** | +6.25% | 16 |
| live_irrelevance | 38.24% | **38.24%** | 0.00% | 884 |
| irrelevance | 66.25% | **64.58%** | -1.67% | 240 |
| parallel | 13.50% | **13.50%** | 0.00% | 200 |
| parallel_multiple | 3.50% | **2.50%** | -1.00% | 200 |
| live_parallel | 0.00% | **0.00%** | 0.00% | 140 |
| live_parallel_multiple | 4.17% | **4.17%** | 0.00% | 120 |

## 分析

### 提升的子集

1. **live_relevance: +6.25%** (93.75% → 100%)
   - v2 的信号匹配对 "有意图且有匹配函数" 的场景识别更准确
   - 16 条全部正确，CARM 路由没有漏选

2. **multiple: +1.00%** (95.00% → 96.00%)
   - 参数提取用 focused prompt + format=json 比 LLM 自由生成更稳定

3. **simple_python: +0.50%** (96.50% → 97.00%)
   - 同上，确定性格式化消除了格式错误

### 持平的子集

4. **live_irrelevance: 38.24%** — 不变
   - 这是 BFCL 的设计问题：irrelevance 查询看起来 actionable 但 GT 是 `[]`
   - v2 的信号评分仍然会给这些查询较高分数（因为查询中的词汇与函数描述匹配）
   - 根因：BFCL irrelevance 是 "有意图但不在函数列表中"，不是 "无意图"

5. **parallel / live_parallel / live_parallel_multiple** — 基本不变
   - v2 的 parallel 检测逻辑虽然能选中多个函数，但 BFCL 的 parallel 用例
     要求精确匹配所有函数的参数，且函数间可能有依赖关系
   - 信号匹配的 parallel 检测对 "and also" 类分隔符有效，但对隐式 parallel（无明确分隔符）无效
   - v2 在 parallel_multiple 上甚至略降 (-1%)，因为误检测了 parallel

### 下降的子集

6. **irrelevance: -1.67%** (66.25% → 64.58%)
   - v2 在某些 irrelevance 用例中 false positive 略增
   - 可能是 action verb hints 匹配了更多无关查询

7. **live_simple/live_multiple: -0.3%** — 微降
   - 在统计噪声范围内（258/1053 条，<1% 变化）

## 关键发现

### v2 解决了什么

- **格式问题完全消除**：v2 的确定性格式化确保 100% 的输出符合 `[func()]` 格式
- **live_relevance 显著提升**：CARM 信号路由对 "有匹配函数" 的场景非常有效
- **参数提取更稳定**：focused prompt + format=json 比自由生成更可靠

### v2 没解决什么

- **Parallel 类仍是短板** (0-13.5%)：
  - 根因不是格式问题，而是 **函数选择 + 参数提取的精度不够**
  - BFCL parallel 用例需要精确选择多个函数并分别提取参数，信号匹配的精度不足
  - 需要 LLM 参与 parallel 检测（而非纯信号匹配），或改用更精细的语义匹配

- **Irrelevance 类仍偏低** (38-65%)：
  - 根因是 BFCL 的 irrelevance 定义与 CARM 的信号匹配逻辑不兼容
  - BFCL irrelevance = "有意图但函数列表中没有匹配的"
  - CARM 信号匹配 = "查询与函数描述有 token overlap"
  - 真正解决需要语义级别的函数-查询匹配（不是 token 级别）

### 延迟对比

- v1 平均延迟：~1-2s/请求（单次 LLM 调用）
- v2 平均延迟：~1-1.5s/请求（信号匹配 <1ms + LLM 参数提取 ~1s）
- v2 延迟与 v1 相当，因为信号匹配是 O(n) 级别，可忽略

## 下一步改进方向

1. **Parallel 改进**：让 LLM 参与 parallel 检测（"这个查询需要调用多个函数吗？"），而非纯信号匹配
2. **Irrelevance 改进**：用 embedding 语义匹配替代 token overlap，或增加 "函数列表上下文" 判断
3. **Multi-turn**：v2 的确定性格式化为 multi-turn 评测奠定了基础（格式不再出错），但需要实现 multi-turn 的状态管理
4. **测试集扩展**：当前 BFCL 已覆盖 ~3500 条，但自建 D2-D5 测试集仍需扩展到 500+

## 结论

v2 的 CARM 路由架构在 **格式正确性** 和 **relevance 检测** 上显著优于 v1 纯透传，但在 **parallel 检测精度** 和 **irrelevance 语义判断** 上仍有瓶颈。整体准确率与 v1 相当（非 parallel 子集），验证了 CARM 信号路由 + LLM 参数提取的架构可行性。

Parallel 和 irrelevance 的根本解决需要从 token 级匹配升级到语义级匹配，这是下一阶段的重点。
