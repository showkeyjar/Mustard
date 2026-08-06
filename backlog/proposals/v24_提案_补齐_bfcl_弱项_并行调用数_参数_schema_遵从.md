# v24 提案：补齐 BFCL 弱项（并行调用数 + 参数 schema 遵从）

> 状态：证据已锁定，待确认实施路线（Track A 触及默认运行时行为，需 Human Gate）
> 关联：v23（Change M documented-format requery，已收官 `bf87790`）

## 0. 适用范围与原则

- CARM 定位：面向已知工具集的**本地函数调用路由**。v24 只在**可比范围（Non-Live + Live 纯函数调用）**内优化，不碰 Agentic / Multi-Turn。
- 方法论：先证据、后改动、小步迭代；承诺清单须由**离线探针**在真实 GT 上实测 GAIN/LOSS/NEUTRAL，不靠猜测；不绕过 Guardian / GitHub 闸门。

## 1. 弱项根因（证据，2026-08-05 实测）

用 `scripts/diag_weakroot_v24.py`（复刻 `eval_bfcl_v4_fast.score_response`，含 list 解包与 permutation 匹配）对五类弱项在**权威 incorrect 样本**上归类：

| 类别 | 总分 | incorrect | COUNT | ARG_FAIL | FUNC_SEL |
|---|---|---|---|---|---|
| parallel_multiple | 40.0% | 63 | 44 | 17 | 2 |
| live_parallel | 43.8% | 5 | 2 | 3 | 0 |
| live_parallel_multiple | 29.2% | 8 | 2 | 5 | 1 |
| simple_java | 53.0% | 24 | 5 | 19 | 0 |
| simple_javascript | 66.0% | 7 | 2 | 5 | 0 |
| **合计** | | **107** | **55** | **49** | **3** |

FUNC_SEL 仅 3 例，可忽略。**两大主杠杆：COUNT(55) 与 ARG_FAIL(49)。**

### 1.1 COUNT 机制（trace 实证，非猜测）
- **少调用（under）**：`parallel_multiple_31` 把 query 正确拆成 4 段，但「每段抽函数」漏抽 `fetch_details`（一个实体需要多个函数时抽取层只给 1 个）。
- **多调用（over）**：`parallel_multiple_24` 对「从 Apple 取现 $1000」同时产出 `invest(负额)` 与 `withdraw`（冗余重复）；`parallel_multiple_32` 把每个天气函数套到两座城市（过度泛化）。
- 结论：split 正确，**per-segment 函数抽取层**是根因（数量多/少、冗余），非 detect_parallel 启发式本身。

### 1.2 ARG_FAIL 机制（GT diff 实证）
用 `scripts/diag_droppedparam_recoverable.py` 量化：ARG_FAIL 中「模型漏掉的参数」仅 **14 个**，其中 5 个(36%) 值可从 query 找回；其余 9 个 GT 值为 `['']` / `['', True]` / `['', 'main']` 等——即**可选/空值/默认参数**，模型省略后被严格评分器误杀。
真正可修的主体是**结构不匹配**（非漏必填）：
- 扁平 dict vs list-of-dict：`simple_java_1` 输出 `params={'limit':'50','schema':'public'}`，GT 要 `params=[{'limit':[50],'schemaFilter':['public']}]`——键名错(`schema`→`schemaFilter`)且结构错(扁平→list-of-dict)；
- `simple_javascript_12` 同上：`expectedResponse={'key':'value'}` vs `expectedResponse=[{'key':['value']}]`。

## 2. 两条修复路线

### Track A — 并行/多调用计数修正（risk_level: high，需 Human Gate）
- 目标：修正 per-segment 抽取的调用数（去冗余 + 补漏）。
- 候选手段（待离线探针选优，不预设）：
  - A1：per-segment 抽取后做「调用去重/合并」——同一 (实体,动作) 不重复发 `invest(neg)`+`withdraw`；
  - A2：识别「一个实体需多函数」的模式，强制补抽；
  - A3：约束「每函数只套用到 query 明确提及其实体的段」，抑制过度泛化。
- **风险**：并行内核已多次调参，改动易引发 regression；且属默认运行时行为 → **必须 Human Gate 批准 + 契约/翻转守卫 + 噪声标尺**。
- 预期收益：主要回收 parallel_multiple 的 44 个 COUNT（部分，非全部）。

### Track B — 参数 schema 遵从修复（risk_level: medium，先实施）
- 目标：后处理层把预测参数**强制贴合函数声明的参数 schema**（键名映射 + list-of-dict 结构包裹），修复结构不匹配类 ARG_FAIL。
- 手段（post-process，不触模型/路由）：
  - B1 键名对齐：当预测键 `schema` 不在 schema 但 `schemaFilter` 在 schema，且语义相近 → 重映射；
  - B2 结构包裹：当 schema 声明某参数为 `array of object`，而预测为扁平 dict → 包裹为 `[{...}]`；
  - B3 可选参数省略豁免：对 GT 值为 `['']`/`['',X]` 类可选参数，预测省略时不判错（评分器侧或后处理侧对齐 BFCL 语义）。
- 爆炸半径守卫：仅对该函数 schema 实际声明的参数做键名/结构校正；预测多出的键、以及预测值与 schema 类型完全无关的，不改。
- 预期收益：回收 simple_java / js / parallel / live_parallel_multiple 中的结构不匹配 ARG_FAIL（需离线探针给出精确 GAIN/LOSS/NEUTRAL）。

## 3. 承诺清单（占位，待离线探针填充）

> 规则：先在真实 GT 上跑离线探针，统计受影响样本集的 GAIN/LOSS/NEUTRAL，再写精确数字。不猜测。

- Track B 承诺集（结构不匹配样本）：GAIN ___ / LOSS ___ / NEUTRAL ___（净 ___）
- Track A 承诺集（COUNT 样本）：GAIN ___ / LOSS ___ / NEUTRAL ___（净 ___）

## 4. 安全不变量与验证计划

- **安全不变量**：
  - 不变量 1（计数守恒）：Track A 不得让正确计数的样本变错计数（翻转守卫 `scripts/diag_weakroot_v24.py` 全量重跑，COUNT 型翻转=0 才放行）；
  - 不变量 2（schema 不越界）：Track B 只校正 schema 声明内的键名/结构，不得引入 schema 外的参数或修改预测值语义；
  - 不变量 3（噪声标尺）：同配置重跑翻转率先量，再谈收益。
- **验证（每次改动）**：
  1. 离线探针：受影响集 GAIN/LOSS/NEUTRAL + 收支闭合；
  2. 全量重跑 `diag_weakroot_v24.py` 翻转守卫（adverse 翻转=0）；
  3. 重跑五类评测，0 传输错误，与承诺清单吻合；
  4. 三态契约对账（kept/broken/unexpected），broken 仅限不利方向。

## 5. 待确认

- [ ] Track A 是否批准（触及默认运行时 + 脆弱并行内核，需 Human Gate）？
- [ ] 优先实施 Track B（已就绪、风险中等）还是 A+B 并行？
- [ ] Track B 的 B3（可选参数省略豁免）是否纳入——它改的是评分语义对齐，需确认不掩盖真实缺失。

## 6. Track B 离线探针结论（2026-08-05，已实测，**后处理形态不可行**）

用 `scripts/probe_track_b.py` 在真实 GT 上模拟后处理修复并复刻评分器重算：

| 策略 | GAIN(错→对) | LOSS(对→错) | net |
|---|---|---|---|
| b2 包裹任意 object/array/HashMap | 0 | 10 | **−10（净负，放弃）** |
| hash 仅包裹 HashMap 类型 | 0 | 0 | 0 |
| hash+嵌套键名对齐 | 0 | 0 | 0 |

**结论**：
1. 通用 dict→[dict] 包裹（b2）净负：把原本能 dict-vs-dict 匹配的样本裹成 list 后反而破坏匹配（LOSS=10），且对结构不匹配样本 0 增益。
2. 即使只包裹 HashMap 类型 + 键名对齐，GAIN 仍为 0——因为 simple_java_1 类失败的根因是 **HashMap 参数内部的嵌套键名 `schema`≠`schemaFilter`**，后处理层能改外层结构却改不了嵌套键名；且包裹会破坏本可匹配的样本。
3. **ARG_FAIL 的真实修复依赖模型遵守嵌套 schema 与键名，后处理层无法可靠修复（硬修即 regression）**。

**因此 Track B 的后处理形态不实施。** 剩余可行路径：
- (a) **Track A（并行计数，55 失败）**：唯一有真实上行空间的杠杆，但高风险、需 Human Gate；
- (b) **模型侧 ARG_FAIL 提示工程**：在抽取 prompt 强调"严格按函数 schema 输出嵌套结构与确切参数名，勿扁平化 list-of-object"，需重跑评测验证，成本中等、收益不确定；
- (c) 放弃 v24：当前核心 ~69% 已满足项目实际场景，弱项属硬骨头。

> B3（可选参数省略豁免）独立看仅 9 例且涉及评分语义对齐；可恢复必填参数仅 5 例——体量太小，不足以支撑独立修复。

## 7. Track A 离线探针结论（2026-08-05，已实测）

### 7.1 真实上行空间（按样本计，`scripts/diag_trackA_upside.py`）

一个样本必须**全部调用都对**才得分，所以按调用实例计会高估。按样本计：

| 方向 | 子形态 | 样本数 | 可修复性 |
|---|---|---|---|
| under（少调） | 全部缺失函数的参数都能从兄弟调用机械复制 | 5 | 可补 |
| under（少调） | 至少一个缺失函数的参数无法复制 | 29 | **不可补** |
| over（多调） | GT ⊆ PRED，只删不加即可得分 | 13 | 纯删除可解 |
| over（多调） | GT ⊄ PRED，删了也不对 | 8 | 不可解 |

**后处理形态的理论天花板 = 18 个样本**（纯删除 13 + 可补全 5）。

### 7.2 后处理删除规则：判别性分析证明规则不存在（`scripts/diag_trackA_separability.py`）

不再逐个猜策略，直接给调用打标签（MUST_DELETE 21 个 / MUST_KEEP 471 个），检验 GT-free 特征能否分开两类：

| 候选删除规则 | 命中该删 | 误伤该留 | precision | recall |
|---|---|---|---|---|
| lex==False（函数名无词法证据） | 9 | 43 | 17.3% | 42.9% |
| grounded==0（参数值不落地） | 2 | 52 | 3.7% | 9.5% |
| lex==False AND grounded==0 | 2 | 5 | **28.6%** | 9.5% |
| dup AND grounded==0 | 1 | 14 | 6.7% | 4.8% |

判别力最强的特征「同名多调」也只有 20/193 = 10.4% 的精度。**没有任何 GT-free 特征能安全地区分「该删」与「该留」。**

三个策略的实跑净收益佐证：`d1` **−11**、`d3` **−13**、`d1+d3` **−24**（`scripts/probe_track_a.py`）。

**根本原因**：`d3` 的前提「函数在 query 里只被提到一次就只该调一次」与 parallel 的语义直接矛盾——并行调用的本质就是「一个函数 × N 个实体」（如 `parallel_multiple_29` 的 `search_cases×2` 本来是对的，却被收敛掉）。

**因此 Track A 的后处理形态同样不实施。**

### 7.3 真正的机制根因（trace 实证）：抽取层缺少「实体归属」约束

`parallel_multiple_36` 的 trace 显示，**函数选择层是对的**，错在参数抽取层：

```
LLM selected: [treaty_info, battle_details]              ← 选对了
  treaty_info   params: {treaty_name: "Battle of Waterloo"}     ← 错（Battle 属于另一函数）
  treaty_info   params: {treaty_name: "Treaty of Tordesillas"}  ← 对
  battle_details params: {battle_name: "Battle of Waterloo"}    ← 对
  battle_details params: {battle_name: "Treaty of Tordesillas"} ← 错
```

`extract_all_params_via_llm`（`carm_bfcl_server_optimized.py:2486`）是三条并行分支的共同收敛点，其 **Rule 2 的示例是无条件笛卡尔的**：

```
2. If the query mentions MULTIPLE entities ... create one object PER entity.
   Example: "weather in Boston and San Francisco" -> [{...Boston}, {...San Francisco}]
```

限定语 "that each need this function" 存在，但模型只学示例，于是每个函数都把所有实体套一遍。Rule 30 有归属约束，却只在有 `Full request` 块时才出现；而 `parallel_multiple_47` 的 segment 本身就同时含 Austin 与 New York，欠切分叠加无归属约束 → 笛卡尔积。

### 7.4 Change N（实体归属规则）实测：安全但效应低于噪声底噪

`scripts/probe_entity_attribution.py`，prompt 级改动，复用已有 LLM 调用，**零额外成本**。

**保真度校验（教训 #5，先做）**：旧 prompt 在 12 个待干预样本上，12/12 复现出「多调」形态、10/12 调用数完全一致 → 镜像可用于决策。

**差分 A/B（教训 #6，锁定基线）**：同一镜像内 OFF vs ON，扩样至 21 待干预 + 40 对照 = 61 样本。

| 样本集 | GAIN | LOSS | NEUTRAL |
|---|---|---|---|
| 待干预（over 失败样本，21） | 1 | 1 | 19 |
| 对照（当前判对样本，40） | 1 | 0 | 39 |
| **合计（61）** | **2** | **1** | **58** |

净 **+1 / 61 样本**。明细：
- GAIN：`pm_36` COUNT→OK（4→2 调用）、`pm_21` ARG_FAIL→OK
- LOSS：`pm_196` OK→COUNT，调用数 **3→5**——归属规则反而让它多产出调用，**方向不单调**

> **关键教训（本轮再次踩中）**：未锁定基线时，结果显示「待干预命中 1/12、对照破坏 4/12」，看似净 −3 的灾难。补上 OFF 基线后发现那 4 个「破坏」在 OFF 下就已经失败（`pm_2/_5` 为 COUNT、`pm_3/_4` 为 ARG_FAIL），全部是简化 prompt 的**镜像偏差**，与归属规则无关。真实差分是 **GAIN=1 / LOSS=0 / 净 +1**。

**结论**：
1. 归属规则确实压低了过度笛卡尔积（`pm_32` 5→4、`pm_51` 4→3、`pm_69` 5→2、`pm_104` 7→6、`pm_11-10-0` 6→5），但**多数仍未精确压到 GT**，且存在反向个例（`pm_76` 8→9、`pm_196` 3→5）。
2. 命中率 1/21 = 4.8%（**在待干预样本上实测，未用自发正确样本估计**，教训 #14）。
3. **噪声标尺（教训 #12）**：同配置重跑翻转率实测 2.5%；并行三类合计约 240 样本，对应随机翻转 **±6**。净 +1/61 外推全量约 **+4**，**低于噪声底噪**，无法用总体分数验证。
4. **镜像局限（诚实标注）**：保真度校验只在待干预样本上做过（12/12 复现多调形态）。对照样本上镜像存在基线发散（`pm_2/_5` 真实服务端判对、镜像 OFF 判 COUNT），说明简化 prompt 会高产调用。因此"对照 LOSS≈0"是镜像内的观测，**不等于真实服务端零风险**——要下确定结论必须走服务端实跑。

**判定**：Change N **不建议以"提分"名义落地**。它净效应 +1/61、方向不单调（有 LOSS 且有反向放大），效应量低于噪声底噪且镜像在对照集上不保真——三项都不支持"低风险可测收益"的门槛。

## 8. v24 结论：三条路径均不满足实施门槛，建议收束

| 路径 | 形态 | 实测结果 | 判定 |
|---|---|---|---|
| Track B | 后处理修参数 schema | net **−10** | 放弃 |
| Track A | 后处理修并行计数 | 最优规则 precision 28.6%；三策略 **−11 / −13 / −24** | 放弃 |
| Change N | prompt 加实体归属约束 | 净 **+1 / 61**，方向不单调，低于噪声 ±6 | 不以提分名义落地 |

**根本判断**：弱项失败的主体（COUNT 55 + ARG_FAIL 49）是后端模型在**分配式实体归属**与**嵌套 schema 遵从**上的语义理解上限，不是路由层的结构性缺陷。路由层的后处理与提示工程都已探到边界：
- 后处理无法凭空恢复语义信息（Track A/B 双双证否，且有判别性分析支撑"规则不存在"而非"参数没调好"）；
- 提示工程能微调倾向，但压不到精确计数（4.8% 命中率）。

**建议**：v24 到此收束，不改默认运行时。当前可比范围核心 ~69%（Non-Live 68.8% / Live 69.9%）已明显优于同级开源基线，且弱项集中在 BFCL 特有的多实体并行构造上——与 CARM「面向已知工具集的本地函数调用路由」的实际场景相关性低。若仍要推进，唯一有量级空间的杠杆是**更换/升级后端模型**，属 Human Gate 范畴。

> 本轮全部为负结果/边际结果，但方法论按预期生效：三次都在动生产代码之前用离线证据拦住了改动，其中 Track B 若盲目上线是 −10，Track A 的 d1+d3 是 −24。
