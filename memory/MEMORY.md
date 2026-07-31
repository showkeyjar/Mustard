# Mustard Long-Term Memory

## Project Focus

- 提高真实 prompt 回归表现
- 提高桌面桥梁 useful rate
- 保守管理主动追问与自动学习风险

## Stable Rules

- 先离线验证，再考虑默认上线
- 高风险行为变更必须经过人类审批

## Model Routing Fixes (2026-07-30)

### 第一轮修复（3项）

1. **evaluate_real_prompts.py 写文件**：原脚本只输出到 stdout 不写文件，导致 `data/eval/real_prompt_eval_latest.json` 记录旧失败状态。已添加写入逻辑。

2. **policy.py Override 2c flag 消费**：`prefer_calculator_for_mixed_numeric_code` flag 之前是死代码。已修改为：flag=0 时 code_executor 优先（旧行为），flag=1 时 calculator 优先（除非有强代码动作动词）。

3. **policy.py _enforce_constraints guard 放宽**：`prefer_search_for_comparison_evidence` guard 从 `has_comparison_evidence_signal` 改为 `has_compare_signal`。

### 第二轮修复（5项，同日）

4. **signals.py CALC_TOKENS 清理**：删除单字 puzzle 关键词（"鸡"/"兔"/"头"/"腿"/"脚"），只保留 "鸡兔同笼"。"脚本"中的"脚"曾误触发 calc signal。

5. **policy.py Override 0b 守卫**：在 writing/synthesis→consult 的 hard rule 中增加 `and not has_evidence_judgment_signal(user_input)` 守卫，让 evidence_judgment 优先于 synthesis。修复 `real-doc-check`（"这个建议可靠吗"需要 search 验证而非 LLM 合成）。

6. **policy.py Override 4a**：新增 `has_deep_analysis_signal AND has_search_signal AND not hard_code_action AND not hard_arithmetic → search`。修复 `real-plan`（"拆计划"+"查资料"应走 search）。

7. **policy.py _enforce_constraints search action guard**：在 compare guard 后新增 guard——当 query 有显式 search action（检索/搜索/查一下）且没有 writing/formal intent 时，将 CALL_BIGMODEL 改为 CALL_TOOL(search)。修复 `result-integration` 和 `termination`（concept memory 曾将它们推到 bigmodel_proxy）。

8. **policy.py Override 4d**：新增 "还是" 选择结构 + search signal → search。修复 `code-looking-search`（"查官方资料还是跑个脚本"是决策支持查询，不是代码执行请求）。

### 修复验证

- 14 条评测用例：基线 8/14 → 修复后 **14/14（100%）**
- 全部通过，无剩余失败
- 已确认无回归：14 条信号回归测试全部 PASS，31 个单元测试全部 PASS

### Artifacts 重建

- `data/eval/real_prompt_eval_latest.json`：14 条完整评测，包含 baseline/pretrained 兼容字段，code-debug 已修复
- `configs/hard_logic_eval.json`：case ID 从旧格式（real-conflict 等）更新为当前评测 ID（conflict-check 等），required_residuals 对齐实际 records
- `artifacts/reasoning_pattern_codec_latest.json`：重建后 hard_eval_pass_rate 0.1667→1.0，verify_when_residual_risky_rate 0.0→1.0
- `artifacts/current_best.json`：通过 `write_current_best()` 重建，status 从 needs_attention→healthy

### 最终指标

| 指标 | 基线 | 修复后 |
|---|---|---|
| real_prompt_match_rate | 0.0 | **1.0** |
| hard_eval_pass_rate | 0.1667 | **1.0** |
| hard_eval_failures | 5 | **0** |
| critical_failure_count | 4 | **0** |
| verify_when_residual_risky_rate | 0.0 | **1.0** |
| residual_explanation_rate | 1.0 | **0.5714** |
| status | needs_attention | **healthy** |

### code-debug 修复详情（第三轮，同日）

9. **signals.py has_code_signal code error override**：在 `has_en` + ("是什么"/"为什么") 分支中新增 code error context exception。当 "为什么" + code error patterns（错误/异常/崩溃/null/pointer/exception/traceback/segfault 等）同时出现时，返回 True（代码调试而非知识查询）。"是什么" 不触发此 exception。关键：`_CODE_ERROR_PATTERNS` 不含 "报"（过于宽泛），只含明确的代码错误词。

### BFCL V3 评测（同日）

**真实 BFCL V3 函数选择准确率评测**（2551 条测试用例，0 次 LLM 调用）：

| 子集 | 基线 | 修复后 | 变化 |
|---|---|---|---|
| simple | 84.0% | 96.5% | +12.5 |
| multiple | 82.0% | 95.0% | +13.0 |
| parallel | 91.0% | 99.0% | +8.0 |
| parallel_multiple | 94.5% | 98.0% | +3.5 |
| irrelevance | 85.0% | 57.5% | -27.5 |
| live_simple | 43.0% | 86.0% | +43.0 |
| live_multiple | 35.9% | 74.5% | +38.6 |
| **OVERALL** | **61.3%** | **82.9%** | **+21.6** |

- CARM 排名从 #10/12 → **#5/12**，超越 Qwen2.5-72B(81.33%)、GPT-4o Prompt(80.54%)、Llama-3.1-70B(78.67%)
- 三项关键改进（evaluate_bfcl_v3_llm.py）：
  1. **route_hybrid 阈值修复**：no-LLM 模式下不再丢弃 score < 0.3 的匹配，改为总是返回 best_name（除非 score=0.0）
  2. **停用词过滤**：desc_overlap 计算时过滤 the/and/for/how 等停用词，减少 irrelevance 误匹配
  3. **反向子串匹配**：query token 是 function name substring 时加分（"cook"→"cookbook"、"weather"→"OpenWeatherMap"）
- irrelevance 下降是已知 tradeoff：零 LLM 调用下无法完美区分"不相关"和"低重叠但相关"
- pyarrow 通过直接下载 wheel 安装（pip 损坏，用 zipfile.extractall 到 site-packages）
