# G1 的 2 个 loss 是相关性问题，不是退化门控问题

状态: 已归因，不在 v22 处置；v23 候选（目标是相关性验证器，不是退化门控）
证据脚本: `scripts/diag_g1_guard_precision.py`
关联: `data/eval/diag/promise_v22.json`（gain 10 / loss 2 / neutral 30）

## 现象

v22 承诺清单里 2 个 loss，都在 live_irrelevance，都由 Change G1 造成：

| 样本 | query | 模型生成 | v21 | v22 |
|---|---|---|---|---|
| `live_irrelevance_558-171-0` | `Version` | `version_api.VersionApi.get_version({'version': 'Version'})` | 门控抑制 → 判对 | 守卫放行 → 判错 |
| `live_irrelevance_598-193-0` | `I would like to buy a movie ticket in San Jose at 11 o"clock in the night.` | `Movies_1_BuyMovieTickets({'movie_name': 'movie', 'number_of_tickets': 1, 'location': 'San Jose', 'show_time': '23:00'})` | 门控抑制 → 判对 | 守卫放行 → 判错 |

G1 的规则是：值字面出现在 query 里 → 认为是用户给的，不是占位符幻觉 → 不判 `param_name_echo`。

## 一个被数据推翻的假设

初判认为第二例是**子串误命中**：`movie` 只是 "a movie ticket" 的一部分，
不是用户提供的值，加个词边界约束即可。

`diag_g1_guard_precision.py v21` 全量扫描后，这个假设不成立：

```
守卫托住的 (样本,参数) 对: 9
涉及样本数: 7
其中判对 3 / 判错 4  -> 守卫精度 42.9%

独立词 vs 仅子串:
  word_bounded=True   对 3 / 错 4
  word_bounded=False  对 0 / 错 0      <-- 子串命中一个都没有
```

`movie` 在 "a movie ticket" 中就是独立词，`\bmovie\b` 照样匹配。
词边界约束**一个样本都救不回来**，是个假修复。若不做这次测量就动手，
改出来的会是一条毫无作用、却看起来很有道理的规则。

## 真正的结论：gain 与 loss 结构同构

守卫托住的 7 个样本里，判错的 4 个（即 Change G 的 gain 来源）：

```
live_multiple_114-44-1     password='password'
simple_java_55             sharedBlobCacheService='sharedBlobCacheService'
simple_javascript_1        listElement='listElement'
simple_javascript_17       app='app'
```

判对的 3 个（其中 2 个将变成 loss）：

```
live_irrelevance_558-171-0  version='Version'
live_irrelevance_598-193-0  movie_name='movie'
```

两组的形态完全一致：**值字面等于参数名，且在 query 中作为独立词出现**。
从参数层面看不出任何可分离特征。唯一的区别在类别语义：
irrelevance 类别的正确答案是"不要发起调用"，此时参数写得再规范也是错的。

因此：
- 退化门控此前在**兼职充当第二道相关性过滤器** —— 靠"参数看起来像占位符"
  间接拦住了本就不该发出的调用。这是巧合，不是设计。
- G1 拆掉了这个巧合的一部分，代价是 2 个样本。收益是 4~5 个真实修复。
- **在退化门控里继续调规则无法改善这个权衡**，因为判据在门控的可见范围之外。

## v23 方向（不在 v22 动）

目标应是相关性验证器，而不是退化门控。

### 一条被证伪的线索（保留记录）

`598-193-0` 的 trace 里 `Verified: [('Movies_1_BuyMovieTickets', '0.00')]`，
一度当作"低分仍被放行"的缺陷线索。查源码后不成立：

- line 4384 `verified = [(f, 0.0) for f in selected]` —— LLM 消歧路径**丢弃分数**
- line 4288 同样硬编码 0.0

0.00 是管道产物，不是决策信号。全量 1665 个样本的 `Verified` 分数全为 0.00，
其中 1013 个判对 —— 单看分数就能排除它作为判据。

### 真正的下一步

顺着这条线索排查放行路径时，找到了一个远大于本条目的问题：
`llm_selected→verified` 路径占全部失败的 45.3%，
在 irrelevance 两类上准确率只有 13.7%。
详见 `add_relevance_confirm_to_llm_selected_path.md`。

本条目的 2 个 loss 属于那个更大问题的两个实例，不单独处置：
- `598-193-0` 走的正是 LLM 消歧后放行，没有相关性确认步骤
- `558-171-0` 走信号直接放行（`Verified: [('...get_version', '0.65')]`）

2 个样本不足以支撑任何规则修改（Change J 就是在这一步上过拟合的）。
等上面那个提案的收益区间量出来后，回头复核这 2 个样本是否被顺带解决。

## 对 v22 的处置

不改。2 个 loss 已写进承诺清单并签字，部署后必须逐条出现，
否则说明生产行为与契约不符，反而要查。承诺清单的意义就是把代价预先讲明。
