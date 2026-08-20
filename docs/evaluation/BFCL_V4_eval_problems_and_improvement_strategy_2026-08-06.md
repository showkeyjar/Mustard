# CARM Router — BFCL V4 评测问题分析与改进策略

**生成日期**: 2026-08-06
**前置文档**: `BFCL_V4_benchmark_report_2026-08-06.md`（评测结果与 Benchmark 定位）
**被测系统**: CARM Router v4（信号路由 + LLM fallback/disambiguation + 格式修复），后端 qwen3-coder（Ollama 远程）
**评测服务器**: `scripts/carm_bfcl_server_optimized.py --port 11401`（OpenAI 兼容 /v1）
**评测工具**: bfcl-eval 2026.3.23（BFCL V4 数据集）

---

## 一、摘要（结论先行）

- **问题不是"模型不会调用函数"，而是"评测协议没有被适配"**。纯函数调用核心 70.4%（官方榜 ≈78/109）说明信号路由+LLM fallback 的调用能力本身成立；Multi-Turn 0.25%（2/800）与 Agentic 0.00%（0/665）的崩盘是**协议适配缺陷 + 环境限制**叠加的结果，不是模型能力归零。
- **根因分层**（按"谁来修"分三层）：
  1. **协议适配缺陷（P0，收益最大，纯代码可修）**：CARM server 的 `carm_route_bfcl` 是"单轮无状态路由器"——只取最后一条 user 消息 + system prompt 里的函数定义，**完全忽略 assistant 历史、tool 执行结果、state_info**。BFCL 多轮协议要求模型"调用 → 收到 tool 结果 → 决定继续调用或输出空列表结束本轮"，CARM 永远输出函数调用字符串，永远不会结束轮次 → 20 步 force-terminated → `turns(1) ≠ ground truth turns(4/5)` → 全判错。
  2. **环境限制（P1，需用户提供密钥/代理）**：web_search 模型**正确发起**了 `[search_engine_query(...)]`，但无 `SERPAPI_API_KEY` + 外网 TLS 阻断 → tool 返回 `Failed to retrieve the search results from server` → 重试循环至 force-terminated。
  3. **真实能力短板（P2-P4，模型/后处理精度问题）**：parallel_multiple 参数值错误、irrelevance 拒答率不足、Java/JS 结构解码失败、live 场景值误差。
- **对照设计理念的错配**：CARM 的设计目标是"本地单轮函数调用路由"，官方 BFCL Overall 的 **70% 权重**（Agentic 40% + Multi-Turn 30%）恰恰是 CARM 未适配的部分。这不是排名失真，而是**定位与评测口径的错配**——要提升官方口径排名，必须先补协议适配层，而不是继续打磨单轮调用精度。
- **改进策略 P0-P5 与预期收益**（详见第四节）：
  - P0 多轮/agentic 协议适配层（server 侧 wrapper）：预期 Multi-Turn 0.25% → 30-50%，Memory 0% → 15-30%。
  - P1 web_search 环境修复（SERPAPI_API_KEY 或本地搜索代理）：预期 0% → 20-50%。
  - P2 parallel_multiple 参数值抽取强化：预期 50.5% → 65-75%。
  - P3 irrelevance 拒答强化：预期 57.14% → 70%+。
  - P4 simple_java/javascript 结构修复：预期 60-63% → 70-75%。
  - P5（可选）score 目录清理与数据读取统一。

---

## 二、评测结果分层诊断

### 2.1 各子集成绩一览（与官方榜参照模型并排）

| 子集 | CARM 当前 | 官方参照（同子集） | 差距类型 |
|---|---:|---:|---|
| simple_python | 90.75% | Qwen3-32B 75.58% / GLM-4.6 74.25% | ✅ 强项（超出参照） |
| multiple | 90.50% | 94.5-95.5% | 小幅差距 |
| parallel | 86.00% | 91.5-95.5% | 小幅差距 |
| parallel_multiple | 50.50% | 88.5-91.5% | ⚠️ **大差距：参数值错误** |
| simple_java | 63.00% | 80-90% 区间 | ⚠️ 结构解码 |
| simple_javascript | 60.00% | 80-90% 区间 | ⚠️ 结构解码 |
| irrelevance | 80.42% | 78-89% | 达标 |
| live_simple | 67.44% | 82-90% | 值误差+函数名 |
| live_multiple | 66.38% | 78-81% | 值误差+函数名 |
| live_irrelevance | 57.14% | 67-88% | ⚠️ 拒答不足 |
| live_relevance | 100.00% | 62.5-93.75% | ✅ 强项 |
| multi_turn | 0.25% | Qwen3-8B 41.75% / Qwen3-32B 47.87% / GLM-4.6 68% | 🚨 **协议适配缺陷** |
| web_search | 0.00% | Qwen3-8B 12.00% / Qwen3-32B 21.50% / GLM-4.6 77.5% | 🚨 **环境限制** |
| memory | 0.00% | Qwen3-8B 14.62% / Qwen3-32B 26.67% / GLM-4.6 55.7% | 🚨 **协议适配缺陷** |

### 2.2 三层根因的判定依据（行为证据）

**A. 协议适配缺陷（Multi-Turn / Memory 全 0% 的根因）**

证据链（已逐条验证）：

1. **server 侧无状态**：`carm_route_bfcl`（`scripts/carm_bfcl_server_optimized.py:3995`）只调用 `extract_user_query(messages)`（取**最后一条** user 消息，`:742`）+ `extract_functions_from_system_prompt(messages)`。assistant 历史、tool 结果、state_info 全部被丢弃。
2. **对照实验**：把 step_1 的完整历史（含 tool 错误 `mkdir: File exists`）发给运行中的 CARM server（PID 50064，端口 11401），输出与 step_0 **完全相同** `[mkdir, mv]`——tool 结果对决策零影响。
3. **BFCL 协议侧**（`bfcl_eval/model_handler/base_handler.py`）：每轮内循环 step 0..20，只有 `is_empty_execute_response(decoded_model_responses)` 为 True（`multi_turn_utils.py:103`，接受 `[]` 或空 list）才 `break` 进入下一轮；解码失败也 break；否则执行 tool 并把结果加回历史，`count > MAXIMUM_STEP_LIMIT`（20）时 force_quit。
4. **后果**：CARM 永远输出函数调用字符串 → 每轮 20 步 force-terminated → `eval_runner.py:197-210` 判定 `turns(1) ≠ ground truth turns(4/5)` → `multi_turn:force_terminated` 全错。800 条中仅 2 条（base/long_context 各 1）是"单轮内完成"的用例侥幸判对。
5. **memory 场景额外要求**：agentic 的 memory 判定（`agentic_checker.py`）是**在模型最终输出里查找 expected answer 文本**；BFCL memory 的 system prompt 明确要求最终输出 `{'answer': A, 'context': B}` 格式。CARM 的 `format_parallel_output`（`:3981`）只输出函数调用列表，永远不含 answer → 100% 失败。memory_kv 工具返回结构已确认：成功 `{"value": ...}` / 失败 `{"error": "Key not found."}`。
6. **正确样例特征**：唯一判对的 2 条都是"单轮内一次调用即完成"的用例（如 lockDoors + setHeadlights 一次调用即对），证明核心调用能力没问题，缺的只是**轮次推进/终止机制**。

**B. 环境限制（web_search 0% 的根因）**

证据链：

1. `inference_log` 显示模型**正确输出** `[search_engine_query(keywords="...")]`——发起搜索这一步是对的。
2. tool 执行返回 `{"error": "Failed to retrieve the search results from server. Please try again later."}`——本机无 `SERPAPI_API_KEY` 且外网 TLS 间歇性阻断，搜索服务不可用。
3. 模型收到错误后重试循环直至 force-terminated。
4. 定性：**环境限制，不是模型调用失败**。搜索可用后，基于结果的选择/记忆步骤才有机会被评估。

**C. 真实能力短板（单轮类别中低于参照的项）**

1. **parallel_multiple 50.5%（live 58.33%）**：99/99 错误均为 `parallel_function_checker_no_order:cannot_find_match`，细分是**参数值错误**（例：`initial_velocity: 0.0. Expected one of [20.0]`），不是调用数 COUNT 错误。parallel 分支（server `:4629-4663`）已应用 `validate_and_coerce_params`，但值来自 LLM 从 query 抽取，后处理无法猜值。
2. **simple_java 63% / simple_javascript 60%**：ast_decoder 15 条语法解码失败 + type_error + value_error（Java HashMap/JS 对象结构：扁平 dict vs list-of-dict、嵌套键名不一致）。
3. **live_irrelevance 57.14%**：`irrelevance_error:decoder_success`——LLM fallback 选择了相关函数而非拒答（已有 `verify_relevance_via_llm` `:1497`，但阈值/指令不够强）。live 与 non-live irrelevance 差距（57.1% vs 80.4%）即由此。
4. **live_simple 67.44% / live_multiple 66.38%**：value_error（46/192 条）+ simple_function_checker（38/99 条，函数名错）+ multiple_function_checker。

### 2.3 明确"不是"什么（排除项）

- **不是评测连接故障**：v3 全程无卡死（serpapi timeout bug 已修）、base URL 已补 `/v1`、CARM server 健康、generate/evaluate 均正常完成。
- **不是模型不会调用函数**：单轮核心 70.4%，正确样例中一次调用即对的用例存在。
- **不是数据/评分脚本问题**：score 文件为 bfcl evaluate 直接产物；multi_turn 明细中 force_terminated 判定与 BFCL 源码逻辑一致。

---

## 三、与 CARM 设计理念的错配分析

### 3.1 CARM 的设计理念（现状）

- **定位**：本地单轮函数调用路由——给定一段 user query + 工具集，输出一次函数调用列表。
- **核心机制**：信号评分（token 重叠/语义信号）→ 低置信走 LLM fallback/disambiguation → 后处理 schema 校验与 documented-format requery。
- **隐含假设**：一次请求 = 一次调用，无"状态"、无"轮次"、无"工具结果回环"。

### 3.2 官方 BFCL 的口径（事实）

官方 Overall = Agentic×40% + Multi-Turn×30% + Live×10% + Non-Live×10% + Hallucination×10%。其中：

- **Multi-Turn**：要求"调用 → 看 tool 结果 → 继续或结束轮 → 等下一轮用户消息"，且每轮消息可能是**缺失函数/缺失参数/长上下文**的变体。
- **Agentic（Web Search + Memory）**：要求"发起搜索 → 用结果 → 记忆写入/检索 → **最终输出 answer**"，本质是带状态的多步 agent。

### 3.3 错配结论

| 维度 | CARM 现状 | BFCL 要求 | 错配后果 |
|---|---|---|---|
| 状态 | 无状态（只读最后一条 user 消息） | 多轮状态机（历史+tool 结果+state_info） | 永远第 1 轮 |
| 轮次终止 | 永远输出函数调用 | 空列表/非函数输出 = 结束当前轮 | 20 步 force-terminated |
| 最终输出 | 只输出调用列表 | 可以输出 answer（memory） | answer 永不存在 |
| 工具失败处理 | 无（不读 tool 结果） | 根据错误调整策略 | 死循环重试 |
| 权重覆盖 | 只覆盖 Non-Live+Live（官方 20%） | 70% 权重在 Agentic+Multi-Turn | 官方口径上限被锁死 |

**一句话**：CARM 是"单轮无状态路由器"，BFCL 的 70% 权重在"多轮有状态 agent"。要提升排名，补协议适配层（不改单轮路由核心）是 P0，比继续打磨单轮精度（收益在 20% 权重内）性价比高得多。

---

## 四、改进策略（P0-P5）

### P0（最高优先级，最大收益）：CARM server 增加"多轮/agentic 协议适配层"

**目标**：让现有单轮路由核心能跑通 BFCL 多轮/agentic 协议，而不改变单轮路由逻辑。

**实现位置**：`scripts/carm_bfcl_server_optimized.py` 的 `carm_route_bfcl` 外围或 `_handle_chat_completions`——加一个 wrapper：解析完整 messages，识别"这是多轮/agentic 会话"，在**调用单轮核心之前/之后**做协议决策。

**三个关键决策规则**（均基于 BFCL 协议语义）：

1. **状态感知的轮次终止**：解析 messages 中的 tool 结果。若**最近一步的 tool 结果全部成功**（无 error 字段）且当前步没有新的 user 指令要求继续 → 输出 `[]` 结束当前轮（触发 `is_empty_execute_response` break）。这直接解决 force-terminated。
2. **重复调用检测（防死循环）**：若**同一函数 + 同一参数**在最近 3 步内连续出现 → 停止输出该调用，改为 `[]` 或尝试下一个候选。这解决 tool 错误（如 `mkdir: File exists`）导致的无限重试。
3. **memory answer 协议**：若 system prompt 含 `{'answer':`（BFCL memory 场景特征）→ 在函数调用完成后，用 LLM 把最终状态整理成 `{'answer': A, 'context': B}` 格式输出（而非调用列表），满足 `agentic_checker` 的 answer 查找。

**不动的部分**：单轮 non-live/live 请求（无 assistant 历史、无 tool 结果）完全走原路径，零回归风险。

**预期收益**：Multi-Turn 0.25% → 30-50%（参照 Qwen3-8B 41.75% / Qwen3-32B 47.87%）；Memory 0% → 15-30%（参照 Qwen3-8B 14.62% / Qwen3-32B 26.67%）。

**验证方法**：改后重启 CARM server（勿误杀其他进程）→ 重跑 `bfcl generate --test-category multi_turn` + `bfcl evaluate` → 确认 score 明细中 `force_terminated` 大幅下降、`turns` 匹配数上升。

### P1：web_search 环境修复

**目标**：让搜索工具可用，解除环境限制。

**方案**：
- 首选：用户提供 `SERPAPI_API_KEY`（写入 vault/环境变量），修复后重跑 web_search 两子集。
- 备选：本地搜索代理（如 DuckDuckGo HTML 端点或本地自建搜索 API），需确认 BFCL web_search 工具的实现是否只支持 SERPAPI（`bfcl_eval` 内 search 工具源码可查）。

**预期收益**：web_search 0% → 20-50%（参照 Qwen3-8B 12.00% / Qwen3-32B 21.50% / GLM-4.6 77.5%）。

**验证方法**：配置后单条 curl 验证 `search_engine_query` 工具返回真实结果 → 重跑 agentic web_search 子集。

### P2：parallel_multiple 参数值抽取改进

**目标**：修复"值来自 LLM 抽取、后处理无法猜值"的问题。

**证据**：99/99 错误均为 `cannot_find_match`，细分为值错误（例：`initial_velocity: 0.0. Expected one of [20.0]`），非调用数错误。

**方案**：
1. **prompt 强化**：在 `extract_all_params_via_llm` / `extract_all_params` 的 prompt 中明确"所有参数值必须逐字复制自 query 原文，不得推断、不得默认、不得计算"。
2. **值来源校验（后处理）**：对每个抽取出的参数值做"子串检查"——值必须是 query 中出现的子串；不是则触发 requery（用 documented-format requery 同款机制，仅重抽该参数）。
3. **枚举值预检**：对 schema 中有 enum 的参数，直接把 query 与 enum 候选做模糊匹配，跳过 LLM 抽取。

**预期收益**：parallel_multiple 50.5% → 65-75%（live 58.33% 同步提升）。

**验证方法**：重跑 parallel_multiple + live_parallel_multiple 两子集，对比 `cannot_find_match` 明细中 value_error 占比。

### P3：irrelevance 拒答强化

**目标**：提高 live_irrelevance 的拒答率（当前 57.14%）。

**证据**：`irrelevance_error:decoder_success`——LLM fallback 选择了相关函数而非拒答。

**方案**：
1. **提高 LLM 拒答决策阈值**：当前 `best_score < 0.3` 才走 verify；对 live（自然语言）场景提高到更严阈值，让更多"看似相关实为无关"的输入进入拒答判定。
2. **verify prompt 强化**：在 `verify_relevance_via_llm` 的 prompt 中明确"查询与工具**语义上不相关**时必须回答 NONE/不调用，即使工具名看起来通用（如 requests.get、get_current_weather）"。
3. **通用 API 函数黑名单偏置**：对 requests.* / 通用 HTTP 类函数，若 query 无明确端点信息则默认拒答。

**预期收益**：live_irrelevance 57.14% → 70%+（non-live irrelevance 80.42% 不回退）。

**验证方法**：重跑 live_irrelevance，检查 `irrelevance_error:decoder_success` 占比下降。

### P4：simple_java / simple_javascript 结构修复

**目标**：修复 Java HashMap / JS 对象的参数结构不匹配。

**证据**：ast_decoder 15 条语法解码失败 + type_error + value_error（扁平 dict vs list-of-dict、嵌套键名不一致）。

**方案**：
1. **模型侧/prompt 样例**：在函数抽取/参数生成 prompt 中给出 Java HashMap 与 JS 对象的**正确 JSON 表示样例**（list-of-dict、嵌套键名），引导 LLM 输出正确结构。
2. **嵌套结构 coerce 加强**：`validate_and_coerce_params` 对"参数 schema 是 list of object / dict of dict"的类型做更积极的结构转换（递归 coerce，而非只做顶层类型校验）。
3. **ast_decoder 失败兜底**：对 decode 失败样例，用 LLM 重新格式化原始输出（保留语义，只修语法）。

**预期收益**：simple_java 63% / simple_javascript 60% → 70-75%。

**验证方法**：重跑 simple_java + simple_javascript，对比 decode 失败与 type_error/value_error 占比。

### P5（可选）：score 目录清理与数据读取统一

**现状**：`score/carm-router/` 下存在旧副本 `multiple/`（旧 73%）与 `simple_python/`（89.5%）目录，与 `non_live/`、`live/` 并存；报告采用 non_live/live 为准。

**方案**：清理旧目录，统一数据读取路径，避免后续分析误读旧分数。

**收益**：无分数提升，纯数据卫生。

---

## 五、风险与回滚

| 策略 | 风险 | 回滚方式 |
|---|---|---|
| P0 wrapper | 单轮路径被误判为多轮 → 输出 `[]` 导致误拒 | 仅当 messages 含 assistant/tool 历史才启用 wrapper；单轮请求路径不变；可加开关环境变量 |
| P0 memory answer | LLM 整理 answer 可能引入幻觉值 | 只在 system prompt 含 `{'answer':` 时启用；answer 值必须来自已成功的 tool 结果 |
| P2 值子串校验 | 合法值（如日期格式化）不在 query 原文 → 误拒 | 校验仅作 requery 触发，不直接丢弃；requery 失败仍用原值 |
| P3 拒答阈值 | 误拒真实相关调用 → relevance 下降 | 阈值可配；先小步调参观察 live_relevance（当前 100%）不回退 |
| P4 结构 coerce | 激进转换可能破坏简单参数 | 只对嵌套结构参数启用，单层参数走原路径 |

**通用原则**：所有改动默认**开关可配、默认关**或**仅多轮会话启用**，先重跑受影响子集回归，确认单轮 13 子集分数不回退后再合入。

---

## 六、实施顺序建议

1. **P0**（收益最大，纯代码，无外部依赖）→ 实现 → 重启 server → 重跑 multi_turn + agentic(memory) 验证。
2. **P1**（需用户提供 SERPAPI_API_KEY 或批准本地代理）→ 与 P0 并行等待输入。
3. **P2**（prompt+后处理，改动小）→ 与 P0 可并行。
4. **P3 / P4**（调参与样例工程）→ 在 P0/P1 跑通后按收益顺序执行。
5. **P5**（数据卫生）→ 任意时间。

**里程碑验收**：P0+P1 落地后，Multi-Turn ≥ 30%、Memory ≥ 15%、web_search ≥ 20%，同时单轮核心 70.4% 不回退 → 官方口径（Overall 按公式推算）将从 ≈14-22% 提升到 ≈30-40% 区间（Agentic+Multi-Turn 两项即 70% 权重）。

---

## 七、附录：证据与代码位置

- CARM server：`D:/codes/Mustard/scripts/carm_bfcl_server_optimized.py`
  - `carm_route_bfcl`（`:3995`）— 单轮路由入口，只读 user query + functions
  - `extract_user_query`（`:742`）— 只取最后一条 user 消息
  - `format_parallel_output`（`:3981`）— 只输出函数调用列表
  - parallel 分支（`:4629-4663`）— 已应用 `validate_and_coerce_params`
  - `verify_relevance_via_llm`（`:1497`）— irrelevance LLM 复核
- BFCL 侧（`D:/tools/miniconda3/envs/BFCL/Lib/site-packages/bfcl_eval/`）
  - `model_handler/base_handler.py` — 多轮循环 step 0..20、`is_empty_execute_response` break、MAXIMUM_STEP_LIMIT force_quit
  - `eval_checker/multi_turn_eval/multi_turn_utils.py:103` — `is_empty_execute_response`（`[]` 或空 list = 结束轮）
  - `eval_checker/eval_runner.py:197-210` — force_terminated（turns 不匹配）判错
  - `eval_checker/agentic_eval/agentic_checker.py` — memory 在模型输出中查找 expected answer
- 对照实验：CARM server PID 50064（端口 11401）step_1 完整历史 → 输出与 step_0 相同 `[mkdir, mv]`
- 数据文件：`data/eval/bfcl_v4_official_leaderboard.csv`（官方榜 109 模型）；score 文件 `score/carm-router/**/BFCL_v4_*_score.json`
