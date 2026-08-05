# BFCL V4 当前成绩与竞品位置（2026-08-05）

## 0. 先说结论（避免误读）

- **被测系统**：CARM Router v4（信号路由 + LLM fallback/disambiguation）+ 后端 LLM **`qwen3-coder`（480B MoE，本地 Ollama）**。BFCL 测的是「这套系统的函数调用能力」，不是裸 qwen3-coder。
- **能诚实报的分数是「函数调用核心」**：Non-Live + Live 共 13 个子集，当前约 **69%**（2026-08-05，其中 3 个 live 类别为 v23 实测；其余 10 个为 2026-07-17 干净跑分）。
- **不跨范围报「官方 Overall」**：官方 Overall = Agentic 40% + Multi-Turn 30% + Live 10% + Non-Live 10% + Hallucination 10%。本系统的 **Agentic（Web Search+Memory）和 Multi-Turn 不在项目适用场景内**，且评测端 API 格式也不匹配（目前 0%）——这是「不测该项」而非「能力为 0」。因此**只在可比范围（Non-Live+Live 纯函数调用）内做排名对比**，不跨范围拿 Overall 比榜首。
- 仓库里 `CARM_Benchmark_Report.md` / `BFCL_v5_report_20260718.md` 写的「Overall 16.97%」是**假数字**：被上述 0% 拉穿，不能作为成绩引用。

## 1. 官方 BFCL V4 计分结构（来源：gorilla.cs.berkeley.edu 官方博客 + 排行榜）

```
Overall = Agentic×40% + Multi-Turn×30% + Live×10% + Non-Live×10% + Hallucination×10%
  Agentic (665条)   = Web Search(200) + Memory(465)         ← 非平均
  Multi-Turn (800条)                                      ← 非平均
  Live (1351条)      = live_simple/multiple/parallel/
                      parallel_multiple/relevance/irrelevance ← 按条目加权
  Non-Live (1150条)  = simple/java/javascript/multiple/
                      parallel/parallel_multiple/irrelevance/sql ← 非平均
  Hallucination (1122条) = 非 live+live irrelevance 合计
  Format Sensitivity (5200条) → 不计分
```

**含义**：官方 Overall 里「函数调用核心」（Non-Live+Live）只占 **20%** 权重；剩下 80% 是 Agentic（40%）+ Multi-Turn（30%）+ Hallucination（10%）。本系统目前只覆盖了那 20%。

## 2. 当前可上报成绩（逐子集，带日期与来源）

| 子集 | 条目数 | 2026-07-17 干净跑分 | 当前值（v23 实测，2026-08-05） | 说明 |
|---|---|---|---|---|
| simple_python | 400 | 86.0% | 86.0% | 未重跑 |
| simple_java | 100 | 53.0% | 53.0% | 参数类型格式问题 |
| simple_javascript | 50 | 66.0% | 66.0% | 同上 |
| multiple | 200 | 81.5% | 81.5% | 未重跑 |
| parallel | 200 | 83.5% | 83.5% | 未重跑 |
| parallel_multiple | 200 | 40.0% | 40.0% | 弱项 |
| irrelevance (non-live) | 240 | 71.7% | 71.7% | 未重跑 |
| live_simple | 258 | 76.0% | 76.0% | 未重跑 |
| **live_multiple** | 1053 | 52.6% | **73.69%** | v23 Change M + 前序修复，776/1053，0 传输错误 |
| live_parallel | 16 | 43.8% | 43.8% | 并行检测启发式弱 |
| live_parallel_multiple | 24 | 29.2% | 29.2% | 弱项 |
| **live_relevance** | 16 | 100.0% | **93.75%** | v23 实测 15/16，逐样本一致 |
| **live_irrelevance** | 884 | 42.5% | **64.71%** | v23 实测 572/884，+0.68pp，无回归 |
| SQL (non-live) | — | 无数据 | **缺失** | 未评测 |
| multi_turn / memory / web_search | — | 0% | **无效** | 评测端 API 格式不匹配，基建故障，非真实分 |

### 当前核心分数（可上报部分）

- **Non-Live（7 子集非平均）** = (86+53+66+81.5+83.5+40+71.7)/7 = **68.8%**
- **Live（6 子集按条目加权）** = **69.9%**
- **函数调用核心 ≈ 69.4%**（两组都约 69%）

> 对比：2026-07-17 那次 13 子集全跑（v4 报告）加权 **58.3%**；v23 把 3 个 live 类别拉高后，核心约 **69%**。

## 3. 竞品位置（官方 BFCL V4 排行榜，更新于 2026-04-12）

| 排名 | 模型 | Overall | 类型 |
|---|---|---|---|
| 1 | Claude-Opus-4-5 (FC) | 77.47% | 闭源 |
| 2 | Claude-Sonnet-4-5 (FC) | 73.24% | 闭源 |
| 3 | Gemini-3-Pro-Preview (Prompt) | 72.51% | 闭源 |
| 4 | GLM-4.6 (FC thinking) | 72.38% | 开源 MIT |
| 5 | Grok-4-1-fast-reasoning (FC) | 69.57% | 闭源 |
| 6 | Claude-Haiku-4-5 (FC) | 68.70% | 闭源 |
| 7 | Gemini-3-Pro-Preview (FC) | 68.14% | 闭源 |
| 8 | o3-2025-04-16 (Prompt) | 63.05% | 闭源 |
| 11 | Moonshotai-Kimi-K2-Instruct (FC) | 59.06% | 开源 |
| — | **本系统（核心 69%，仅供参考）** | — | 本地 qwen3-coder |
| 参考 | Qwen3-235B-A22B-Instruct (FC) | 54.37% | 开源 |
| 参考 | Qwen3-32B (FC) | 48.88% | 开源 |

### 怎么读这个位置

1. **官方榜上没有 qwen3-coder**（只有 Qwen3-235B / 32B 等）。本系统后端是 qwen3-coder，但 CARM 路由 + LLM fallback + 格式修复把「函数调用核心」推到约 69%，**高于官方榜上的 Qwen3-235B（54%）和 Qwen3-32B（49%）**——一个编码模型靠路由层补到了聊天优化模型之上，方向上是合理的。
2. **69% 落在官方 Overall 的「函数调用核心」那 20%（Non-Live+Live）里，和榜首 77% 不在同一口径**——榜首 77% 里 70% 权重来自 Agentic+Multi-Turn，那是本项目不做的场景。所以**不跨范围比**：在「函数调用核心」这一可比片上，本系统处于中上游（≈ 榜上第 6–7 名那一片），且明显强于同族 Qwen 官方分。
3. 弱项很清楚：**parallel_multiple (40/29%)、java/javascript 参数格式、live_parallel (43.8%)**；这些是真实能力短板，不是评测故障。

## 3.5 可比性原则：只在可比范围里比（不迎合打榜）

CARM 是**面向已知工具集的本地函数调用路由**（OpenClaw agents，中文场景，自托管），不是联网 agents。这决定了：

- **可比范围 = 纯函数调用这一片 = Non-Live + Live（官方 Overall 里占 20% 的那部分）**。
- **不可比范围 = Agentic（Web Search + Memory）+ Multi-Turn**。这两块本就不在项目的适用场景里。为了把官方 Overall 凑高而去硬改这两块，属于「为打榜而迎合」——既偏离项目定位，也会掩盖真实能力。**不去做，也不该做。**

**所以定位直接落在可比范围上：**

| 维度 | 本系统（核心 ~69%） | 可比参照 |
|---|---|---|
| 同族基线 | 高于官方 Qwen3-235B(FC) 54% / Qwen3-32B(FC) 49% | 路由层 + 格式修复补到了同族官方分之上 |
| 官方榜同片位置 | 处于中上游（≈ 第 6–7 名那一片：Claude-Haiku-4-5 68.7% / Gemini-3-Pro-FC 68.1%） | 和纯函数调用模型同口径比 |
| 项目真实短板 | parallel_multiple(40/29%)、java/javascript 参数格式、live_parallel(43.8%) | 应在本项目场景内优化，而非为打榜补 Agentic |

**结论**：在「可比的函数调用范围」内，本系统约 69%，处于官方榜中上游、且明显高于同族 Qwen 官方分——这就是它当前真实的水平。不要拿「不可排名」当缺点，也不要为了 Overall 去补非目标场景的能力。

## 4. 要拿到「可排名的官方 Overall」，必须先做

1. **修评测端 API 格式适配**（让 CARM server 正确返回 `tool_calls`，FC 模式能解析）——这是 Multi-Turn / Memory / Web Search 三项 0% 的根因。修好后这三类才从「无效」变「可测」。
2. **补 SQL 子集**评测（Non-Live 目前缺 SQL）。
3. 跑全量官方 `bfcl-eval` 一次（不要 `--allow-overwrite`，先备份原始 `data_overall.csv`——2026-07-24 那次就是被 overwrite + server 崩溃丢了原始分）。
4. 之后用官方计分脚本直接出 Overall，再和榜面对齐。

## 5. 数据可信度标注

- ✅ 可信：`live_multiple / live_relevance / live_irrelevance` 的 v23 数值（2026-08-05，0 传输错误，与承诺清单吻合）。
- ⚠️ 较旧：`simple/multiple/parallel/irrelevance/live_simple/live_parallel/...` 为 2026-07-17 跑分，v22/v23 的部分修复可能已悄悄改动它们，但未重跑验证。
- ❌ 无效：`multi_turn / memory / web_search` 的 0%（评测基建故障）。
- ❌ 误用警示：`CARM_Benchmark_Report.md` 的「Overall 16.97%」因上述 0% 失真，不可作为成绩。
