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

- 统计 description 里声明格式模板的参数总数，以及其中生成值已经合规的比例。
  如果多数已合规，那 76 例可能有别的共因，不是"格式没遵守"这么简单。
- 量化方案 1 的触发面：会新增多少次 LLM 调用，落在哪些类别。
- 确认重问不会破坏当前已判对的样本——需要一份承诺清单，
  走 `scripts/contract_change_gh.py` 同样的路子。

## evaluation_plan

- python scripts/diag_failure_census.py v22 --cat live_multiple --detail arg_missing_suffix
- 新建 diag 脚本量化"声明格式的参数"总体合规率
- 承诺清单 + 定点回归，不做全量重跑

## 排期

排在 v22（Change G/H）验收之后。v22 飞行途中不动任何生成侧代码——
F2 归因事故已经证明同一次重启混入两个改动会让数字无法归因。
