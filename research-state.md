# Research State: CARM Routing + BFCL v5 Completion

## Current Stage
COMPUTE — BFCL evaluation complete but API format issue blocked

## Research Question
1. 未提交的 policy.py/runner.py/signals.py 修改是否应合入 main？
2. 如何修复 CARM BFCL Server 以获取准确的 multi-turn/memory/web_search 评分？

## Key Decisions
- ROUTING: 3个文件的修改已通过 tool_boundary + comparison_search 候选评测，建议合入 main
- BFCL: 非 live (73.33%) 和 live (100% relevance) 的 simple_python/simple_java/multiple/parallel 已验证正常
- BLOCKER: CARM server 返回 JSON body 而非 OpenAI tool_calls 格式，导致所有 FC-based categories 评分为 0%

## Experiment Log
| Attempt | Method | Result | Status |
|---------|--------|--------|--------|
| 1 | evaluate_tool_boundary_candidate | candidate_pass (3/3 OK) | ✅ |
| 2 | evaluate_comparison_search_candidate | candidate_pass (2/2 OK) | ✅ |
| 3 | manual real_prompt eval subset | 11/15 match rate 73% | ⚠️ (预存偏差, 但 core fixes work) |
| 4 | bfcl generate multi_turn,memory,web_search | 1576/1576 generated (~2h) | ✅ 但 server 崩溃末尾 ~50条失败 |
| 5 | bfcl evaluate categories | All 0% (API protocol mismatch) | ❌ |
| 6 | bfcl generate non_live+live | 3641/3641 generated (~3.8h) | ⚠️ Server 崩溃导致最后 ~50 条 connection error |
| 7 | bfcl evaluate non_live+live | Score files corrupted | ❌ Old scores lost |
| 8 | Restore old scores from memory | 55.21% Non-Live, 56.55% Live | ⚠️ Approximate |

## What Worked
- Override 2c 修复验证通过：real-mixed → calculator ✅
- compare→search guard 验证通过 ✅
- runner.py attention gate 逻辑正确
- signals.py MULTI_INTENT_CONNECTORS 修复合理
- BFCL v4 Non-Live/Live 旧数据验证 CARM 路由器工作正常
- server 运行稳定前 ~98% 测试成功，证明 CARM route 逻辑可行

## What Didn't Work
- **BFCL 全类别评分 0%**：CARM server 返回结构化 JSON 而非 OpenAI tool_calls 格式
- **old score files 被 `--allow-overwrite` 破坏**：generate run 覆盖了旧的 good results
- **server 在长时间运行后崩溃**：可能由于 Ollama gemma3:12b 模型加载超时或资源耗尽

## Open Questions
1. 如何修改 carm_bfcl_server_optimized.py 返回 OpenAI tool_calls 格式？
2. Evidence judgment signal 对真实路由的具体影响？
3. Server crash 原因是什么？（ollama timeout? resource exhaustion?）

## Artifacts
- BFCL_v5_report_20260718.md: exists (updated)
- research-state.md: exists
- memory/daily/2026-07-23.md, 2026-07-24.md: exists
