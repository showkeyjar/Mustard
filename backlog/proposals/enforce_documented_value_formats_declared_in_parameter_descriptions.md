# Enforce documented value formats declared in parameter descriptions

- problem: 参数 schema 的 description 里明确写了取值格式（例如 `'City, State'`，`State names must be abbreviated with two letters`），生成端却输出裸城市名。BFCL 严匹配下直接判错。全套件 624 个失败样本里，83 个属于这一形态，其中 76 个集中在 live_multiple——这是当前最大的单一可修复失败桶（13.3%）。
- from_failure_pattern: arg_missing_suffix
- change_type: postprocess_or_repair_pass
- risk_level: medium
- needs_human_approval: False（属于 schema 合规修复，非默认运行时策略变更）

## 为什么这不是刷榜

判断标准是"格式要求有没有文档依据"。实测三个样本的 schema 原文：

```
Events_3_FindEvents.city
  "The city where the event is taking place, in the format 'City, State' or
   'City, Country' (e.g., 'New York, NY' or 'London, UK'). State names must be
   abbreviated with two letters."

Movies_1_BuyMovieTickets.location
  "The city in which the movie theater is located, in the format of
   'City, State', such as 'Los Angeles, CA'."
```

格式约束写在 schema 里，模型没遵守。修这个等于让输出符合被调用方声明的契约，和 Change H（把值吸附到 schema 词表）是同一族，不是把 GT 抄进代码。

**反过来说，如果实现方式是内置一张 city→state 映射表，那就变成刷榜了。**
映射表是我们替 benchmark 写的答案，不是从 schema 推出来的。这条必须守住。

## 证据

```
$ python scripts/diag_failure_census.py v21
    83  arg_missing_suffix        （总失败 624）

$ python scripts/diag_failure_census.py v21 --cat live_multiple
live_multiple  331 fail  →  arg_wrong_value 112 / arg_missing_suffix 76 /
                            func_wrong 45 / func_extra 45 / ...

样本形态（全部同构）:
  live_multiple_398-139-2  Events_3_FindEvents.city   pred='San Diego'      want='San Diego, CA'
  live_multiple_399-139-3  Events_3_FindEvents.city   pred='Chicago'        want='Chicago, IL'
  live_multiple_401-139-5  Events_3_FindEvents.city   pred='Toronto'        want='Toronto, ON'
  live_multiple_411-141-0  Movies_1_FindMovies.location  pred='Union City'  want='Union City, CA'
  live_multiple_418-141-7  Movies_1_BuyMovieTickets.location pred='New York' want='New York, NY'
```

## 候选实现（按正当性排序）

1. **定向重问（推荐）**。检测 description 里声明了格式模板且生成值不匹配时，
   就该参数单独回问一次模型，把 description 原文和当前值给它，让它补全。
   州名来自模型的世界知识，不是我们写的表。代价：每个违例一次 LLM 调用。
2. **提示词强化**。把带格式声明的参数描述在 prompt 里提权。改动生成行为，
   影响面不可控，需要全量回归而不是承诺清单式的定点验证。
3. **内置映射表**。否决。见上。

## 事前必须先做的事（不做完不准实施）

- ~~统计 description 里声明格式模板的参数总数，以及其中生成值已经合规的比例。~~
- ~~量化方案 1 的触发面：会新增多少次 LLM 调用，落在哪些类别。~~
- ~~确认重问不会破坏当前已判对的样本——需要一份承诺清单。~~

**三条全部完成，见下。状态从"候选"升级为"已量化，待实施决策"。**

---

## 前置检验结果（2026-08-05，基于 v22 全量）

### 1. 合规率：提案的归因成立，但不像原先说的那么普遍

`scripts/diag_documented_format.py v22`

```
参数位点 (func.param) 共 1569 个，其中声明了逗号格式的  37 个
命中实例 405
                  gt需格式   gt两可   gt裸值可   gt无此参
  pred合规            292       0        3        21
  pred不合规           74       1        6         8
总体合规率 316/405 = 78.0%
```

多数（78%）已经合规。**不合规的 89 例里 74 例 GT 明确要格式且当前判错**——
归因方向是对的，但"最大可修复失败桶"这个说法要收敛到 89 例触发面。

识别规则只认 description 里的**带引号示例**，且要求示例全部含逗号。
早期版本把 `music_theory.key_signature` 描述里当分隔符说明的 `', '`
抽成了"格式示例"，凭空造出一个假阳性位点，已修。

### 2. GT 自洽性：34 个有 GT 的位点里 5 个自相矛盾

```
Services_4_FindProvider.city    needs_format=26 bare_ok=1
Hotels_2_SearchHouse.where_to   needs_format=21 bare_ok=2
Services_1_FindProvider.city    needs_format=21 bare_ok=1 either=1
Hotels_4_SearchHotel.location   needs_format=14 bare_ok=3
uber.ride.loc                   needs_format=2  bare_ok=1
```

主流是要格式（84 : 9），但同一 schema 声明下 GT 两种都收。
**这些例外在线上无法从 schema 区分**——和 Change G1 的 2 个 loss 是同构问题：
门控/修复规则不可能比 GT 本身更自洽。这决定了 loss 不可能压到 0。

### 3. 代理指标不可用（幸存者偏差实测 16.7pp）

模型**自发**按格式输出的 295 例里命中 286，命中率 96.9%。
这个数字曾被当成"重问能达到的上界"，**是错的**：
那 295 例是模型有把握才自发加的后缀，89 例不合规的恰恰是它没把握的。

实测（`scripts/probe_format_requery.py v22`，89 次真实调用，1.3 分钟）：

```
now_hit 65 / still_miss 10 / broke 6 / no_gt 8
重问命中率 = 65/81 = 80.2%
```

**80.2% vs 96.9%，代理指标高估 16.7pp。**
这条记进方法论：拿"系统自发做对的样本"估计"干预没把握样本"的成功率，
是标准的幸存者偏差。

失败形态（10 个 still_miss）全部可解释，且都不是格式问题本身：
```
'LA'   -> 'LA, CA'                gt 'Los Angeles, CA'   只加后缀不展开缩写
'SD'   -> 'SD, CA'                gt 'San Diego, CA'     同上
'san jose' -> 'san jose, CA'      gt 'San Jose, CA'      不规范大小写
'Kuala Lumpur' -> '..., Malaysia' gt 'Kuala Lumpur, MY'  国家全称 vs ISO 码
'my current location' -> '..., CA' gt 'San Francisco, CA' 原值本来就错
'London' -> 'London, Country'     gt 'London, UK'        **填了模板占位符**
```

占位符只此 1 例；description 明说 "two letters" 的位点上，
后缀不是两个大写字母的违例 **0 例**——模型是听文档的。
1 例不值得加守卫（加了也救不回，`London` 保留原值仍错），仅记录。

### 4. 承诺清单（整样本重放判分）

`scripts/replay_format_requery.py v22 --promise data/eval/diag/promise_v23_format.json`

重放是保真的：只替换已生成参数的字符串值，不改函数选择、不改切分、
不增删调用，LLM 决策序列完全不动（性质同 Change H）。
护栏保留：离线判分不能复现线上判定的样本直接剔除（本次剔除 1 个）。

```
GAIN 57 / LOSS 3 / NEUTRAL 26   净 +54   （全部落在 live_multiple）
```

承诺的 3 个回退样本（上线后必须逐个复核）：
```
live_multiple_4-2-1      '123 Hanoi Street' -> '123 Hanoi Street, Hà Nội, Vietnam'  gt 裸地址
live_multiple_303-131-2  'Vancouver'        -> 'Vancouver, WA'                      gt 裸城市
live_multiple_1046-273-0 'Delhi'            -> 'Delhi, Delhi'                       gt 裸城市
```

对照：v22 的 Change G/H 全套净收益 +8。这个改动净 +54，约 7 倍。
折算全量 3343 样本约 **+1.6 个百分点**。

### 5. 触发面

87 个样本 / 89 次新增 LLM 调用，**100% 落在 live_multiple**。
按探针实测，89 次调用耗时 1.3 分钟（temperature=0）。
延迟影响限于该类别中命中格式声明且不合规的样本，其余样本零开销。

## 实施要点（如获批）

- prompt 必须与 `scripts/probe_format_requery.py` 的 `PROMPT` 逐字一致，
  temperature=0——承诺清单是在那个 prompt 下产生的，换 prompt 清单即作废。
- 触发条件严格等于 `declares_comma_format()`：description 有格式引导词、
  ≥2 个带引号示例、且示例**全部**含逗号。放宽任何一条都要重新出清单。
- 上线后走 `scripts/contract_change_gh.py --verify` 同一套三态对账
  （kept / broken / untestable_drift），并量噪声底噪。
  仅 live_multiple 一个类别、1053 样本，重跑成本可接受。
- 绝不引入 city→state 映射表。后缀来自模型的世界知识，
  一旦我们自己写表，就是把 GT 抄进代码。

## evaluation_plan

- ~~python scripts/diag_failure_census.py v22 --cat live_multiple~~
- ~~新建 diag 脚本量化"声明格式的参数"总体合规率~~
- 承诺清单已生成：`data/eval/diag/promise_v23_format.json`
- 定点回归 live_multiple 单类，不做全量重跑

## 排期

前置检验已于 v22 验收后完成。**实施需要在服务端新增一次条件性 LLM 调用**，
超出纯后处理范畴，故在实施前向人类确认，而非直接进 v23。

## 实施记录（v23，2026-08-05）

人类已批准实施。改动落在 `scripts/carm_bfcl_server_optimized.py`，命名为 **Change M**：

- 内联 `declares_comma_format` / `shape`（逐字搬自 `diag_documented_format.py`，
  纯正则、无外部依赖）。
- `_FORMAT_REQUERY_PROMPT` 与 `scripts/probe_format_requery.py` 的 `PROMPT`
  **逐字一致**，否则承诺清单作废。
- `requery_documented_formats()` 在 `snap_calls_to_schema_vocab` 之后、`format_parallel_output`
  之前注入；触发条件 = `declares_comma_format` 为真 且 当前值非 COMMA 结构。
- 重问 LLM 调用复用 `call_ollama`，强制 `temperature=0.0`；`call_ollama` 新增
  `num_predict=None` 以匹配探针的未限长请求。`ENABLE_FORMAT_REQUERY` 默认开。
- **关键保真设计**：重问后的日志用 `Format requery:` 前缀，不用
  `NAME params:` 格式，避免污染契约签名；trace 的 `params:` 行在重问前已记录，
  所以 v22/v23 生成签名一致，三态判定的 `same_gen` 成立。
- 归因横幅新增 `[x] M  documented-format requery`。

## v23 实测结果（已验收）

`live_multiple` 全量 1053 样本，0 传输错误：

| | v22 | v23 | delta |
|---|---|---|---|
| live_multiple | 722/1053 = 68.57% | **776/1053 = 73.69%** | **+54 (+5.13%)** |

`scripts/contract_change_format.py --base v22 --verify v23`：

```
分态: kept=82  broken=1 (gain 1/loss 0)  untestable_drift=0  unexpected_favorable=3
兑现 (同生成): gain 45 / loss 3
噪声标尺（承诺外 967 样本）: 变好 12  变坏 14  净 -2  翻转率 2.69%
收支闭合: 承诺集内 +56  +  集合外噪声 -2  =  实测总差值 +54   闭合 True
```

预测 +54，实测 +54。承诺集内实测 +56 与预测 +54 的 2 点差额完全可解释：

**1 个不利违约** —— `live_multiple_456-145-7`。探针在同一 prompt、
`temperature=0` 下得到 `'London, UK'`（gt 一致），线上得到 `'London, England'`。
**Ollama 在 temperature=0 下并非确定性**。description 里的示例同时给了
`'Paris, France'` 和 `'New York, NY'`，正好在 `UK` / `England` 之间制造摇摆。
该样本 v22 本就判错，v23 仍错，相对基线净影响为 0。

**3 个有利方向的预测失准** —— 离线判分器在三处比线上判分器严格，
系统性低估收益。三处都已定位到具体样本：

| 样本 | 离线判错的原因 | 线上判分器的实际行为 |
|---|---|---|
| `286-129-1` | `'san jose, CA'` vs gt `'San Jose, CA'` | 字符串比较不区分大小写 |
| `532-151-8` | `date: None` vs gt `['', 'dontcare']` | 可选参数的 None 等价于缺省 |
| `596-158-2` | `number_of_rooms: '1'` vs gt `[1]` | str/int 自动强制转换 |

方向是保守的（低估而非高估），但这是离线镜像与线上判分器之间的真实保真度缺口，
下次做离线重放前应先补齐这三条规则。

## 上线后补做的两件事（原提案漏掉的）

**1. 暴露面必须按 schema 量，不能按当前生成量**

前置检验里"其他类别触发面 0"是拿 v22 的**具体预测值**数出来的，
那是一次观测，不是不变量。新建 `scripts/diag_format_exposure.py`
改为扫描 schema 结构上限：

```
live_multiple     1053 样本  616 个含可触发位点  43 个位点
live_irrelevance   884 样本  294 个含可触发位点  53 个位点
live_relevance      16 样本    8 个含可触发位点  15 个位点
其余 7 个类别                  0                0   <- 结构上不可触及
```

这个判断立刻被证实：v23 的 `live_relevance` 运行中 Change M **触发了 5 次**。
若沿用"0 触发"的结论，就会漏掉两个类别不复测。

复测结果：`live_relevance 15/16`，与 v22 **逐样本一致，无翻转**；
安全不变量 `live_relevance_6-6-0` 仍由 `empty_required` 抑制，逐字一致。

**2. 爆炸半径守卫 + 锁定部署配置的契约**

`_requery_one_value` 原本只剥两端引号，没有防住内部引号、超长回复。
这是 Change M 唯一可能影响"判定不依赖参数值"的类别的路径：
畸形值会破坏输出调用串，让判分器解析失败。

新增 `_requery_value_rejected`：拒绝含引号/括号/换行、长度 > 80、
词数 > 8、或不含逗号的重问值，一律回退原值。

守卫是在 v23 评测**跑完之后**加的 —— 这就使「实测配置」与「部署配置」
不是同一份代码，正是"A/B 对比必须锁定基线配置"要防的坑。
弥合缺口的证据固化在 `scripts/contract_format_guard.py`：
守卫对承诺集 89 个值、v23 实测运行 83 个值**全部不生效**，
两份配置在所有已观测数据上行为等价。改动守卫阈值前必须先跑这个脚本。

## 契约脚本的一处语义修正

初版把"承诺说会错、实际判对"也计入 `broken` 并输出"契约失败"。
这是错的：有利方向的偏离不引入风险，不是违约。
已拆出 `unexpected_favorable` 单列，通过闸门只看不利方向。
**放宽仅限有利方向，不利方向的判定一字未动** ——
自检（v22 verify v22）仍报 57 个不利违约，脚本没有因放宽而在空改动上变绿。

## live_irrelevance v23 全量复测（闭环最后一环，884 样本）

`scripts/diag_bfcl_v4.py --categories live_irrelevance --tag v23 --workers 10`，0 传输错误：

| | v22 | v23 | delta |
|---|---|---|---|
| live_irrelevance | 566/884 = 64.03% | **572/884 = 64.71%** | **+6 (+0.68%)** |

分类翻转守卫（`scripts/contract_category_flips.py`，adverse 即 exit 1）报：

```
adverse (correct→wrong): 7
favorable (wrong→correct): 13
transport errors: 0
```

但"0 adverse 即违约"的硬阈值是为**有承诺集**的类别设计的。live_irrelevance
**没有承诺集**，Change M 预期会在其 294 个暴露位点上改写值，所以部分翻转是
预期的，必须逐例归因而非直接判违约：

- 7 个 adverse 中 **6 个 Change M 未触发**（trace 无 `Format requery:`，v22 判
  `pred=[]`、v23 错误地生成了调用）—— 纯 call/no-call 决策的 re-run 抖动。
  Change M 只改已存在调用的参数值、不能插入调用，与此无关。
- 1 个 adverse（`live_irrelevance_598-193-0`）Change M **触发了**
  （`'San Jose' → 'San Jose, CA'`），但 v22 本就 `pred=[]`（正确、无关调用），
  v23 自己生成了一个多余调用，Change M 只是在这个已错的调用里补了后缀。
  错因是多余调用，不是重问。

13 个 favorable 中 Change M 触发 **0/13**，全是 re-run 抖动。

**Change M 真实足迹**：live_irrelevance 上触发 **18/884**。其中 v22 判对的仅 1 个
（即上面的 598-193-0，错因是多余调用而非重问）。**没有任何一个样本是
"Change M 重问导致其从对变错"**。18 个触发样本里 16 对 / 2 错，2 错都是 v23 自行
生成了多余/已错调用、Change M 仅在其上补后缀。

结论：live_irrelevance 安全不变量保持；净 accuracy +0.68%；翻转率 2.26%
（20/884）落在 re-run 噪声带内（live_multiple 承诺外翻转率 2.69%）；
Change M 暴露面真实存在（18 触发印证 schema 扫描的 294 暴露位点）。

## v23 闭环结论

Change M（documented-format requery）实施、评测、契约对账全部完成：

| 类别 | v22 | v23 | delta | 安全不变量 |
|---|---|---|---|---|
| live_multiple (1053) | 68.57% | 73.69% | **+54 (+5.13%)** | 契约闭合 True（56 −2 = +54，与承诺精确吻合）；唯一 broken=456 为 Ollama 非确定性净 0，非回归 |
| live_relevance (16) | — | 15/16 | 0 | 逐样本一致，0 翻转，安全不变量 `live_relevance_6-6-0` 保持 |
| live_irrelevance (884) | 64.03% | 64.71% | +6 (+0.68%) | Change M 触发 18/884，无"重问致对变错"；翻转率 2.26% = 噪声带 |

- **承诺清单净 +54 全部兑现**，收支闭合，无未解释缺口。
- **三态契约对账通过**：kept=82 / broken=1（非回归）/ drift=0 / unexpected_favorable=3（均为离线判分偏严，已定位到大小写/None/类型强制三处）。
- **部署配置 ≡ 实测配置**：等价性守卫（`contract_format_guard.py`）对承诺集 89 值 + v23 实测 83 值全部"空操作"，exit 0。
- **暴露面按 schema 量**（修正"0 触发=不变量"的误判）：live_multiple / live_irrelevance / live_relevance 均复测，其余 7 类结构上不可触及。
- **爆炸半径守卫** `_requery_value_rejected` 已加，且经契约证明在已观测数据上恒为 no-op。
- **无回归**：三个类别的 adverse 翻转要么来自 re-run 抖动、要么来自 v23 自身生成的多余调用，无一样本由 Change M 重问导致从对变错。0 传输错误。

下一步：提交 v23 全部产出；如需升至默认运行时策略，依 AGENTS.md 走 Human Gate。
（Change M 目前 `ENABLE_FORMAT_REQUERY` 默认开，但属于 postprocess 修复族，
非"默认运行时策略/模型/供应商"变更，按提案 risk_level=medium 无需 Human Gate。）
