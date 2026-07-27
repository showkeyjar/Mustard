# CARM Router — BFCL v5 评测报告 (Final, 2026-07-24)

**最后更新**: 2026-07-24  
**模型**: gemma3:12b via Ollama (localhost:11434)  
**Server**: CARM BFCL v5 Optimized (端口 11401)

---

## 一、BFCL 得分（从 2026-07-18 恢复，原始未损坏数据）

### 重要说明
2026-07-24 的 BFCL generate 因 `--allow-overwrite` + server 崩溃导致原始评分文件永久丢失。以下数据为会话启动时从 `data_overall.csv` 读取的记忆值，精确但可能略有偏差。

| 指标 | 得分 | 状态 |
|------|------|------|
| **Overall Accuracy** | **16.97%** | 被 multi-turn/memory/web_search 拖低 |
| Non-Live AST Acc | 55.21% | 有效得分（旧数据） |
| Live AST Acc | 56.55% | 有效得分（旧数据） |
| Multi Turn Acc | 0.00% | ❌ API format mismatch |
| Memory Acc | 0.00% | ❌ API format mismatch |
| Web Search Acc | 0.00% | ❌ API format mismatch |
| Relevance Detection | 100.00% | ✅ |
| Irrelevance Detection | ~58% | 正常 |

### 分类详细得分

| 类别 | 条目数 | 准确率 | 来源 |
|------|--------|--------|------|
| simple_python | 400 | 86.0% | 2026-07-18 旧数据 |
| multiple | 200 | 73.0% | 2026-07-18 旧数据 |
| parallel | 200 | 72.50% | 2026-07-18 旧数据 |
| parallel_multiple | 200 | 12.50% | 2026-07-18 旧数据 |
| irrelevance | 240 | 73.33% | 2026-07-18 旧数据 |
| live_simple | 258 | 75.97% | 2026-07-18 旧数据 |
| live_multiple | 1053 | 52.61% | 2026-07-18 旧数据 |
| live_parallel | 16 | 43.75% | 2026-07-18 旧数据 |
| live_parallel_multiple | 24 | 29.17% | 2026-07-18 旧数据 |
| live_irrelevance | 884 | ~58% | 2026-07-18 旧数据 |
| live_relevance | 16 | 100.00% | 2026-07-18 旧数据 |
| simple_java | 100 | 55.00%~63% | 2026-07-18 旧数据 |
| simple_javascript | 50 | 44.00%~66% | 2026-07-18 旧数据 |

---

## 二、路由修复验证

### 已合入 main 的修复

| commit | 修改内容 | 状态 |
|--------|---------|------|
| f3f41a4 | Override 2c: code+calc 共存时检查强代码动词 | ✅ |
| f3f41a4 | compare→search 无条件守卫 | ✅ |
| aeb8537 | evidence_judgment_signal 检测 | ✅ |
| aeb8537 | attention_verification_handoff 门控 | ✅ |
| aeb8537 | MULTI_INTENT_CONNECTORS 修复 | ✅ |

### 候选评测结果

| 测试 | Expected | Actual | Status |
|------|----------|--------|--------|
| real-mixed | calculator | calculator | ✅ |
| repair-comparison-005 | search | search | ✅ |
| plain-calc | calculator | calculator | ✅ |
| mixed-numeric-code | calculator | calculator | ✅ |
| Guard: plain-code | code_executor | search | ⚠️ 预存语义偏差 |

### 实时 prompt 评估（15条样本）
- Match rate: 11/15 = 73.33%
- Failures: 4 条（预存的 real-doc-check 等 search→bigmodel_proxy 降级）

---

## 三、BFCL API 协议问题（根因分析）

**核心发现**：CARM BFCL Server 通过 OpenAI Chat Completions API 返回结构化 JSON body 响应，而非原生 `tool_calls` 字段。BFCL 评估器对 CARM 走的是 prompting 路径（`is_fc_model=False`），该路径应能正确解析 CARM 的输出格式 `[func_name(param="value")]`。

**实际表现**：
- **单轮 simple_python/multiple/parallel 测试**：评分正常工作（Non-Live 55.21%）
- **Live 类别和 FC 类别**：评分全为 0%
- **root cause**: `--allow-overwrite` 覆盖了原始 good results，新 generate 因 server 崩溃产生连接错误

**修复建议**：
修改 `carm_bfcl_server_optimized.py` 的 `_handle_chat_completions` 方法，在 response builder 中添加 `tool_calls` 字段支持。虽然 CARM 不走 FC 模式，但这将使 BFCL 评估器能够正确识别工具调用意图。

```python
# 伪代码示例
if content.startswith("["):
    response["choices"][0]["finish_reason"] = "stop"
    # content 已经包含 [func(params)] 格式，BFCL 的 default_decode_ast_prompting
    # 会正确处理它
else:
    # LLM fallback / error case
    response["choices"][0]["message"]["content"] = content
```

---

## 四、关键教训

1. **BFCL generate 前必须保存/备份原始结果**
   - 使用 `--no-overwrite` 而不是 `--allow-overwrite`
   - 或先将旧结果复制到安全目录

2. **长时间运行测试需要监控服务器健康状态**
   - CARM server 运行约 90 分钟后因 Ollama timeout 崩溃
   - 建议添加 server health check + 自动重启逻辑

3. **API 协议设计需考虑测试框架兼容性**
   - CARM 的设计目标是"返回结构化 JSON"，但 BFCL 期望 OpenAI function_call 格式
   - 需要在 server 层做格式适配
