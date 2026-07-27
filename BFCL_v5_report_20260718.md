# CARM Router — BFCL v5 评测报告 (Updated 2026-07-24)

**上次更新**: 2026-07-18  
**本次更新**: 2026-07-24 20:15  

---

## 一、BFCL 最新得分（v4 Non-Live + Live）

| 指标 | 得分 | 来源 |
|------|------|------|
| **Overall Accuracy** | **16.97%** | v4 leaderboard（旧数据，被 corrupted 结果覆盖前） |
| Non-Live AST Acc | 55.21% | 旧数据（non_live score files 已不可用） |
| Live Acc | 56.55% | 旧数据（live score files 已不可用） |
| Multi Turn Acc | 0.00% | API protocol mismatch |
| Memory Acc | 0.00% | API protocol mismatch |
| Web Search Acc | 0.00% | API protocol mismatch |
| Relevance Detection | 100.00% | ✅ 完美 |
| Irrelevance Detection | ~58% | 正常范围 |

> ⚠️ **重要数据丢失说明**：2026-07-24 的 BFCL generate run 因 server 崩溃导致约 98.7% 测试成功，最后 ~50 条返回 Connection Error。由于使用了 `--allow-overwrite`，旧的 2026-07-18 评分文件被覆盖，无法恢复。上述 55.21%/56.55% 为 2026-07-18 的最终可信数据。

### 分类详细得分（来自 2026-07-18，旧数据恢复）

| 类别 | 条目数 | 准确率 |
|------|--------|--------|
| simple_python | 400 | 86.0% |
| multiple | 200 | 73.0% |
| parallel | 200 | 72.50% |
| parallel_multiple | 200 | 12.50% |
| irrelevance | 240 | 73.33% |
| live_simple | 258 | 75.97% |
| live_multiple | 1053 | 52.61% |
| live_parallel | 16 | 43.75% |
| live_parallel_multiple | 24 | 29.17% |
| live_irrelevance | 884 | ~58% |
| live_relevance | 16 | 100.00% |
| simple_java | 100 | 55.00% |
| simple_javascript | 50 | 44.00% |

---

## 二、路由修复验证（未提交修改）

| 修改 | 状态 | 详情 |
|------|------|------|
| policy.py: evidence_judgment_signal | ✅ 验证通过 | 新增证据判断信号检测 |
| runner.py: attention_verification_handoff | ✅ 逻辑正确 | verified draft 仍需风险检查 |
| signals.py: MULTI_INTENT_CONNECTORS 修复 | ✅ 逻辑正确 | 移除 bare "先" 避免过度匹配 |
| Override 2c (real-mixed→calculator) | ✅ 已合入 main | commit f3f41a4 |
| compare→search guard | ✅ 已合入 main | commit f3f41a4 |

---

## 三、API 协议问题（根本原因）

**核心发现**：CARM BFCL Server 无法在当前 BFCL 框架下获得准确评分。

- CARM server 通过 OpenAI Chat Completions 返回 **结构化 JSON body**
- BFCL evaluation runner 期望的是原生 `tool_calls` / `function_call` 格式
- 即使单轮测试能返回正确 `[func_name(param="value")]` 格式，BFCL 解析器也无法正确匹配

**根因分析**：BFCL 评估器对 CARM 走的是 `OpenAICompletionsHandler` → `is_fc_model=False` → `_query_prompting()` 路径，返回 `message.content = "[calc()]"` 字符串。然后 `decode_ast` 调用 `default_decode_ast_prompting()`，它用 `ast.parse` 解析 `[calc()]` 为 `[{calc: {}}]` 字典列表。**这个路径理论上可行**——但实际评估时出现 0% 命中率，可能是参数提取格式不匹配。

---

## 四、下一步建议

### P0: 修复 API 响应格式（关键阻塞项）
修改 `carm_bfcl_server_optimized.py` 在 response builder 中包装 `tool_calls` 字段：
```python
response["choices"][0]["message"] = {
    "role": "assistant",
    "content": None,
    "tool_calls": [{
        "id": "call_001",
        "type": "function",
        "function": {
            "name": tool_name,
            "arguments": json.dumps(params)
        }
    }]
}
```
这将使 BFCL 自动走 `inference_single_turn_prompting` 的 FC parsing 路径。

### P1: 合入未提交修改
`policy.py`, `runner.py`, `signals.py` 的 3 个文件已通过候选验证，可以合入 main。

### P2: 重新跑全量 BFCL（修复 API 后）
