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
