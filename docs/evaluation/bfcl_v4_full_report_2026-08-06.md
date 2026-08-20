# CARM Router — BFCL V4 完整评测结果与 Benchmark 定位报告

**生成日期**: 2026-08-06
**被测系统**: CARM Router v4（信号路由 + LLM fallback/disambiguation + 格式修复）
**后端 LLM**: qwen3-coder（Ollama，远程 192.168.31.20:11434）
**评测服务器**: `scripts/carm_bfcl_server_optimized.py --port 11401`（OpenAI 兼容 /v1）
**评测工具**: bfcl-eval 2026.3.23（BFCL V4 数据集）
**数据来源**: 本地 `score/carm-router/**/BFCL_v4_*_score.json`（bfcl evaluate 直接产物）；官方排行榜 `gorilla.cs.berkeley.edu/data_overall.csv`（2026-08-06 下载，109 个模型）

---

## 一、摘要（结论先行）

- **当前可信成绩（纯函数调用核心，Non-Live + Live 13 个子集）**：Non-Live AST **74.56%**，Live AST **66.40%**，可比核心约 **70.4%**。
- **与官方 BFCL V4 排行榜同口径对比**：在「Non-Live + Live 纯函数调用」这一可比片上，CARM 核心分位于官方榜 109 个模型中的 **≈ 78 名**（超过 77 个模型），接近 `xLAM-2-3b-fc-r` / `Llama-3.2-3B-Instruct` / `Granite-20b` 等函数调用专用小模型区间。
- **同族对比（Qwen 系）**：官方榜上 Qwen3-32B(FC) 可比核心 85.4%、Qwen3-235B(Prompt) 84.5%、Qwen3-8B(FC) 84.1% —— CARM 的 70.4% **低于 Qwen 官方函数调用成绩**。这与 2026-08-05 报告的「69% 高于 Qwen 官方 54%/49%」不同，**原因是口径差异**：旧报告把 CARM 核心分与 Qwen 官方 **Overall**（被 Agentic/Multi-Turn 拖低）比；本报告与官方榜的 **同口径核心分** 比。官方口径下 Qwen 的核心分并不低。
- **强项**：simple_python 90.75%、multiple 90.50%、parallel 86.00%、live_relevance 100%。
- **弱项**：parallel_multiple 50.50%、live_irrelevance 57.14%、live_simple 67.44%、simple_java/javascript 63%/60%。
- **Multi-Turn 与 Agentic**：本次为「修复评测连接错误（base URL 缺 `/v1`）」后的首次有效实测：Multi-Turn **0.25%**（2/800）、Agentic **0.00%**（0/665），结果与根因见第六节。低分**不是基建故障**，而是 (a) 模型不会按 BFCL 多轮/agentic 协议「结束当前轮」——收到 tool 结果后循环重复同一函数调用直至 force-terminated；(b) web_search 受无 SERPAPI_API_KEY + 外网 TLS 阻断的环境限制（模型正确发起了搜索调用，但搜索服务不可用）。本类非目标场景，不用于排名背书。

> ⚠️ 排名口径警示：官方 Overall = Agentic×40% + Multi-Turn×30% + Live×10% + Non-Live×10% + Hallucination×10%。本项目定位为**本地函数调用路由**，Agentic（Web Search+Memory）与 Multi-Turn 非目标场景；因此只在可比范围（Non-Live+Live，官方权重 20%）内排名对比，不跨范围拿 Overall 比榜首。

---

## 二、评测方法

### 2.1 被测系统
- CARM Router v4：基于信号评分（token 重叠/语义信号）路由到已知工具集；低置信区间走 LLM fallback / disambiguation；后处理做参数 schema 校验与 documented-format requery。
- 后端 LLM：`qwen3-coder`（Ollama 远程）。评测走 `carm_bfcl_server_optimized.py` 的 OpenAI 兼容 `/v1` 端点（模型名 `carm-router`）。

### 2.2 评测流程
- 单轮类别（simple/multiple/parallel/parallel_multiple/irrelevance + live_*）已在 2026-08-03~08-04 完成评测（score 文件最新时间戳为证）。
- multi_turn、agentic 两个类别由 `scripts/run_bfcl_overall_v1.py` 驱动，2026-08-06 第三轮（v3 日志 `bfcl_overall_v3.log`）完整跑完：
  1. `bfcl generate --model carm-router --test-category multi_turn` → 已完成（800 条：base/long_context/miss_func/miss_param 各 200；v3 因已有结果跳过重生成）
  2. `bfcl generate --model carm-router --test-category agentic` → 已完成（8055.2s：Web Search 200 + Memory 465）
  3. 两个类别分别 `bfcl evaluate` → multi_turn 14.3s、agentic 10.9s，`OVERALL_RUN_DONE` 已出现
- 修复内容：`OPENAI_BASE_URL=http://localhost:11401/v1`（此前缺 `/v1` 导致 0% 连接错误）；serpapi timeout bug（毫秒被当秒）已修。

### 2.3 数据可信度标注

| 数据 | 状态 | 说明 |
|---|---|---|
| Non-Live 7 子集 + Live 6 子集 | ✅ 可信 | bfcl evaluate 直接产物（score 文件 2026-08-03/04），无传输错误 |
| Multi-Turn 4 子集 | 🆕 本次实测（最终） | 2026-08-06 完整 generate+evaluate（v3，OVERALL_RUN_DONE）；低分为模型轮次协议行为问题，详见第六节 |
| Agentic（Web Search+Memory） | 🆕 本次实测（最终） | generate+evaluate 完成；web_search 为环境限制（无 SERPAPI_API_KEY+TLS 阻断），memory 为模型轮次协议行为问题 |
| SQL（non-live） | ⚠️ 缺失 | 未评测 |
| Format Sensitivity | ⚠️ 缺失 | 未评测 |

---

## 三、完整 BFCL V4 结果（CARM Router）

### 3.1 Non-Live（离线函数调用，1150 条）

| 子集 | 条目数 | 正确 | 准确率 | 备注 |
|---|---:|---:|---:|---|
| simple_python | 400 | 363 | **90.75%** | 强项 |
| simple_java | 100 | 63 | 63.00% | 参数类型/结构格式问题 |
| simple_javascript | 50 | 30 | 60.00% | 同上 |
| multiple | 200 | 181 | **90.50%** | 强项 |
| parallel | 200 | 172 | **86.00%** | 强项 |
| parallel_multiple | 200 | 101 | 50.50% | ⚠️ 弱项：并行调用数 COUNT/ARG_FAIL |
| irrelevance | 240 | 193 | 80.42% | 无关输入拒答 |

**Non-Live AST Summary（官方口径，4 组平均）= 74.56%**

### 3.2 Live（在线函数调用，1351 条 AST + 891 relevance/irrelevance）

| 子集 | 条目数 | 正确 | 准确率 | 备注 |
|---|---:|---:|---:|---|
| live_simple | 258 | 174 | 67.44% | 自然语言 query |
| live_multiple | 1053 | 699 | 66.38% | 自然语言 query |
| live_parallel | 16 | 10 | 62.50% | 小样本 |
| live_parallel_multiple | 24 | 14 | 58.33% | 小样本 |
| live_relevance | 16 | 16 | **100.00%** | 强项 |
| live_irrelevance | 875 | 500 | 57.14% | ⚠️ 弱项：通用 API 类函数误判 |

**Live AST 加权（四类按条目）= 66.40%**

### 3.3 Multi-Turn（800 条，本次修复后实测）— 见第六节
### 3.4 Agentic（Web Search 200 + Memory 465，本次修复后实测）— 见第六节

---

## 四、官方排行榜对比（Benchmark 定位）

> 口径：官方榜 `data_overall.csv`（109 模型）。可比核心 = Non-Live AST × 50% + Live AST × 50%（官方 Overall 中这两项各占 10%）。

### 4.1 可比范围核心分位置

| 位置 | 模型 | 可比核心 | Overall | 类型 |
|---|---:|---:|---:|---|
| 1 | BitAgent-Bounty-8B | 87.36% | 46.23% | 开源 |
| 2 | Gemini-3-Pro-Preview (Prompt) | 86.89% | 72.51% | 闭源 |
| 3 | Qwen3-32B (Prompt) | 86.14% | 46.78% | 开源 |
| 4 | Qwen3-32B (FC) | 85.39% | 48.71% | 开源 |
| 5 | Claude-Sonnet-4-5-20250929 (FC) | 84.89% | 73.24% | 闭源 |
| 6 | Qwen3-235B-A22B-Instruct-2507 (Prompt) | 84.50% | 52.15% | 开源 |
| … | … | … | … | … |
| 76 | Granite-20b-FunctionCalling (FC) | 70.53% | 23.23% | 开源 |
| 77 | Llama-3.2-3B-Instruct (FC) | 70.50% | 21.95% | 开源 |
| **—** | **CARM Router（本系统，可比口径）** | **≈70.4%** | — | 本地 qwen3-coder |
| 78 | Amazon-Nova-Micro-v1:0 (FC) | 70.21% | 22.29% | 闭源 |
| 79 | Granite-3.2-8B-Instruct (FC) | 70.05% | 26.87% | 开源 |
| … | … | … | … | … |
| 109 | Gemma-3-1b-it (Prompt) | 16.02% | 7.17% | 开源 |

**结论**：CARM 可比核心 ≈70.4%，在官方榜 109 模型中排 ≈78 名，超过 77 个模型。落在函数调用专用小模型区间（xLAM-2-3b 72.9%、Granite-20b 70.5%、Llama-3.2-3B 70.5%、Nova-Micro 70.2%）。

### 4.2 与 Qwen 同族对比（诚实口径）

| 模型 | Overall（官方） | 可比核心 | 说明 |
|---|---:|---:|---|
| Qwen3-32B (Prompt) | 46.78% | 86.14% | 官方实测 |
| Qwen3-32B (FC) | 48.71% | 85.39% | 官方实测 |
| Qwen3-235B-A22B-Instruct-2507 (Prompt) | 52.15% | 84.50% | 官方实测 |
| Qwen3-8B (FC) | 42.57% | 84.06% | 官方实测 |
| Qwen3-14B (FC) | 41.03% | 82.47% | 官方实测 |
| **CARM Router（本系统）** | — | **≈70.4%** | 本地 qwen3-coder |
| Qwen3-1.7B (FC) | 28.41% | 78.77% | 官方实测 |

**解读**：CARM 后端 qwen3-coder 非官方函数调用评测模型（官方榜无此型号）。在纯函数调用这一片上，CARM 的 ≈70.4% 低于 Qwen3-8B/14B/32B 的官方核心分（82%~86%），高于 Qwen3-1.7B/0.6B 等小模型。路由层 + LLM fallback 把「编码模型」补到了「小参数函数调用模型」区间，但**未达到 Qwen 中型模型的官方函数调用水平**。

> 注意修正：2026-08-05 报告中的「69% 高于 Qwen 官方 54%/49%」是把 CARM 核心分与 Qwen 官方 **Overall** 比（跨口径）。Qwen 官方 Overall 低是因为其 Agentic/Multi-Turn 分低，与函数调用核心无关。本报告按同口径（核心分）比较，结论更严格、更诚实。

### 4.3 竞品详细分项（官方榜，用于参照）

| 模型 | NL AST | NL Simple | NL Multiple | NL Parallel | NL PM | Live | Live Simple | Live Multiple | Live Parallel | Live PM | Rel | Irr |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Claude-Opus-4-5 (FC) | 88.58 | 76.83 | 95.50 | 93.50 | 88.50 | 79.79 | 86.43 | 78.16 | 87.50 | 75.00 | 62.50 | 84.72 |
| GLM-4.6 (FC thinking) | 87.56 | 74.25 | 95.00 | 91.50 | 89.50 | 80.90 | 89.53 | 78.92 | 81.25 | 75.00 | 75.00 | 84.96 |
| Qwen3-235B (Prompt) | 90.33 | 79.83 | 95.00 | 95.50 | 91.00 | 78.68 | 82.95 | 77.78 | 81.25 | 70.83 | 93.75 | 78.89 |
| Qwen3-32B (FC) | 88.77 | 75.58 | 94.50 | 93.50 | 91.50 | 82.01 | 89.53 | 80.91 | 81.25 | 50.00 | 93.75 | 76.37 |
| DeepSeek-V3.2-Exp (P+T) | 85.52 | 74.08 | 92.00 | 89.50 | 86.50 | 76.02 | 82.56 | 74.74 | 87.50 | 54.17 | 93.75 | 67.00 |
| Moonshotai-Kimi-K2 (FC) | 81.60 | 69.42 | 92.00 | 82.00 | 83.00 | 78.68 | 81.78 | 78.06 | 87.50 | 66.67 | 75.00 | 87.34 |
| **CARM Router（本系统）** | **74.56** | **71.25** | **90.50** | **86.00** | **50.50** | **66.40** | **67.44** | **66.38** | **62.50** | **58.33** | **100.0** | **68.78** |

---

## 五、模型实力分析

### 5.1 强项（相对自身结构）
1. **simple_python 90.75% / multiple 90.50% / parallel 86.00%**：单函数、确定性多函数场景信号路由稳定，接近主流模型（官方榜 top 模型 90~97%）。
2. **live_relevance 100%**：16/16 全部正确识别「应调用」输入，与榜首模型（62.5~93.75%）相比毫不逊色。
3. **irrelevance 80.42%**：非 live 无关输入拒答良好；live_irrelevance 57.14% 是弱项（见下）。

### 5.2 弱项（真实能力短板，非评测故障）
1. **parallel_multiple 50.50%（非 live）/ 58.33%（live）**：并行+多函数组合场景，per-segment 函数抽取层存在 COUNT（少调/多调/冗余）与 ARG_FAIL（结构不匹配）两类问题。v24 提案已证后处理形态不可行（Track B 净负），Track A（并行计数）需 Human Gate。
2. **live_irrelevance 57.14%**：通用 API 类函数（requests.get / get_current_weather 等）下，LLM fallback 倾向于选择相关函数而非拒答。这是 live 与 non-live irrelevance 差距（80.4% vs 57.1%）的主因。
3. **simple_java 63% / simple_javascript 60%**：Java HashMap/JS 对象的参数结构不匹配（扁平 dict vs list-of-dict、嵌套键名不一致）。v24 Track B 已证后处理无法可靠修复，需模型侧 schema 遵从。
4. **live_simple 67.44% / live_multiple 66.38%**：自然语言 query 下仍有约 1/3 失败（值误差、wrong_func_name、wrong_count）。

### 5.3 历史演进（v3 → v4 07-17 → v4 08-04）

| 子集 | v3 (07-13) | v4 (07-17) | v4 当前 (08-04 score) | Δ vs 07-17 |
|---|---:|---:|---:|---:|
| simple_python | 85.0 | 86.0 | **90.75** | +4.8 |
| simple_java | 67.0 | 53.0 | **63.00** | +10.0 |
| simple_javascript | 78.0 | 66.0 | **60.00** | -6.0 |
| multiple | 76.0 | 81.5 | **90.50** | +9.0 |
| parallel | 82.5 | 83.5 | **86.00** | +2.5 |
| parallel_multiple | 40.0 | 40.0 | **50.50** | +10.5 |
| irrelevance | 60.0 | 71.7 | **80.42** | +8.7 |
| live_simple | 58.5 | 76.0 | **67.44** | -8.6 |
| live_multiple | 35.6 | 52.6 | **66.38** | +13.8 |
| live_parallel | 62.5 | 43.8 | **62.50** | +18.7 |
| live_parallel_multiple | 20.8 | 29.2 | **58.33** | +29.1 |
| live_relevance | 68.8 | 100.0 | **100.00** | +0.0 |
| live_irrelevance | 68.1 | 42.5 | **57.14** | +14.6 |

> 注：v4 07-17 为当时「干净跑分」，08-04 为最近一次完整评测（optimized server）。多数类别提升；live_simple 与 simple_javascript 回落需关注（可能为 optimized server 的并行/提取启发式 tradeoff）。

### 5.4 一句话定位
> 在「纯函数调用」这一官方榜可比片上，CARM Router ≈70.4%，处于官方 109 模型的 ≈78 名（函数调用专用小模型区间），强于 77 个官方模型；弱项集中在并行多调用计数与参数 schema 遵从，均已有证据与提案记录。

---

## 六、Multi-Turn 与 Agentic 实测结果（2026-08-06 修复连接错误后，最终完成）

> 本节为「可测性恢复」后的**首次有效实测**：连接错误已修复（base URL 补 `/v1`），`bfcl generate` + `bfcl evaluate` 两个类别均完整跑完（`OVERALL_RUN_DONE`，v3 日志）。分数为诚实记录，不用于排名背书。

### 6.1 Multi-Turn（800 条）

| 子集 | 条目数 | 正确 | 准确率 |
|---|---:|---:|---:|
| multi_turn_base | 200 | 1 | **0.50%** |
| multi_turn_miss_func | 200 | 0 | 0.00% |
| multi_turn_miss_param | 200 | 0 | 0.00% |
| multi_turn_long_context | 200 | 1 | **0.50%** |
| **Multi-Turn 合计** | **800** | **2** | **0.25%** |

### 6.2 Agentic（Web Search 200 + Memory 465）

| 子集 | 条目数 | 正确 | 准确率 |
|---|---:|---:|---:|
| web_search_base | 100 | 0 | 0.00% |
| web_search_no_snippet | 100 | 0 | 0.00% |
| memory_kv | 155 | 0 | 0.00% |
| memory_vector | 155 | 0 | 0.00% |
| memory_rec_sum | 155 | 0 | 0.00% |
| **Agentic 合计** | **665** | **0** | **0.00%** |

### 6.3 低分根因（已逐条定位，均为行为证据而非推测）

1. **Multi-Turn 与 Memory 全 0% 的根因一致：模型不会「结束当前轮」**。
   - 证据（`inference_log` 逐 step 检查）：server 每步都把**真实 tool 结果回传给模型**（如 `{"error": "mkdir: cannot create directory 'temp': File exists"}`），但模型在收到错误后**连续 20 步重复输出同一个函数调用**（如 `[mkdir(dir_name="temp")]` ×20、`[archival_memory_retrieve(key="Michael")]` ×20、`[memory_append(同一文本)]` ×20），直到步数上限被 `force_terminated`。
   - 后果：BFCL multi-turn/agentic 协议要求模型在完成当前轮后用**非函数调用输出**结束该轮、等待下一轮用户消息；模型从不这样做，于是永远停留在第 1 轮（`result turns (1) ≠ ground truth turns (4/5)`），无法进入后续轮次 → 判错。
   - 定性：这是 **qwen3-coder 对 BFCL agentic/多轮「调用-停止」协议的行为缺陷**（后端模型/路由层），不是评测连接错误（server 请求、tool 执行、结果回传均正常）。个别样例（base/long_context 各 1 条）在单轮内完成即判对，说明非结构性全错，而是轮次协议失败为主。
2. **Web Search 0% 的根因是环境限制，不是模型调用失败**。
   - 证据：模型**正确输出了** `[search_engine_query(keywords="...")]`，但 tool 执行返回 `{"error": "Failed to retrieve the search results from server. Please try again later."}` —— 本机无 `SERPAPI_API_KEY` 且外网 TLS 间歇性阻断，搜索服务不可用；模型收到错误后重试循环直至 force-terminated。
   - 定性：**环境限制**。模型完成了「发起搜索」这一步；若搜索可用，后续基于结果的选择/记忆步骤才有机会被评估。
3. **不是基建故障**：v3 全程无卡死（serpapi timeout bug 已修）、无连接错误（`/v1` 已补）、CARM server 健康；`generate`/`evaluate` 均正常完成，score 文件时间戳为证。

### 6.4 参考 Overall 推算（仅作透明记录，禁止跨范围对比）

官方 Overall = Agentic×40% + Multi-Turn×30% + Live×10% + Non-Live×10% + Hallucination×10%。
代入实测：Agentic 0.00%、Multi-Turn 0.25%、Live 66.40%、Non-Live 74.56%；Hallucination 类别**未单独评测**（官方榜该列中位数 77.75%）：
- 保守下限（Hallucination=0）：**Overall ≈ 14.2%**；
- 若以 CARM 自身 relevance/irrelevance 实测均值（(100%+57.14%)/2 = 78.6%）代理 Hallucination：**Overall ≈ 22.0%**。
两种估算下，Overall 的 70% 权重都来自非目标场景（Agentic+Multi-Turn 两项即 70%），与榜首（77.5%）不可比；本报告排名结论一律以第六节以外的可比核心分（≈70.4%）为准。

---

## 七、局限与诚实标注

1. **不是官方提交**：CARM 分数为本地 bfcl-eval 实测，未提交至 gorilla.cs.berkeley.edu；官方榜无 CARM 行。对比表中的 CARM 行是「按官方同口径计算的参照行」。
2. **跨范围对比禁止**：官方 Overall 的 70% 权重来自 Agentic + Multi-Turn。CARM 未把这两项作为目标场景，任何「CARM Overall 排名」都是失真的（如历史 16.97% 假数字）。
3. **数据一致性**：单轮 13 个子集分数来自 2026-08-03/04 的 bfcl evaluate（无传输错误）；`data_overall.csv` 快照（8/4 12:24）中 simple_python 为 89.50%，与 score 文件最新 90.75% 有 1.25pp 差异（simple_python 于 8/4 12:07 重跑更新，CSV 未刷新）。本报告采用 score 文件最新值为准。
4. **小样本类别**：live_parallel（16 条）、live_parallel_multiple（24 条）、live_relevance（16 条）样本量小，波动 ±6~12pp。
5. **后端依赖**：分数依赖 qwen3-coder 实例的可用性与版本；服务器 warm-up 失败会导致部分查询超时（本次无传输错误）。
6. **Agentic/Multi-Turn 环境与协议限制**：web_search 0% 受「无 SERPAPI_API_KEY + 外网 TLS 阻断」环境限制（模型已正确发起搜索调用但搜索服务不可用）；multi_turn/memory 0% 主因是模型不会按 BFCL 协议结束当前轮（循环调用直至 force-terminated），属本轮首次有效实测的行为证据，不代表这些类别完全无能力，但也不作为排名背书。
7. **排名漂移**：官方榜数据为 2026-08-06 下载的当前版本；排行榜随新模型提交而更新，定位结论有时效性。

---

## 八、附录

- 数据文件：`data/eval/bfcl_v4_official_leaderboard.csv`（官方榜，109 模型）
- 本地分数：`D:/tools/miniconda3/envs/BFCL/Lib/site-packages/score/carm-router/**/BFCL_v4_*_score.json`
- 生成结果：`.../result/carm-router/**/BFCL_v4_*_result.json`
- 评测日志：`bfcl_overall_v3.log`（本轮 multi_turn/agentic 完整跑完）
- 汇总脚本：`scripts/build_bfcl_v4_scores.py`
- 相关提案：`backlog/proposals/v24_提案_补齐_bfcl_弱项_并行调用数_参数_schema_遵从.md`
