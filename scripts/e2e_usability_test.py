"""End-to-end usability test — user's perspective, not benchmark's.

Runs 5 real scenarios with actual tool execution (not dry_run).
Records: routing decision, execution result, usefulness verdict.
"""

import os
import sys

os.environ["CARM_NO_EMBEDDING"] = "1"
sys.stdout.reconfigure(encoding="utf-8")

from carm import CARMRouter

router = CARMRouter(embedding=False)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def run_turn(query: str, session_id: str | None = None) -> dict:
    """Execute one turn, return structured result."""
    r = router.route(query, session_id=session_id)
    return {
        "query": query,
        "tool": r.tool_name,
        "ok": r.ok,
        "confidence": r.confidence,
        "result": r.result[:200],  # truncate for readability
    }


def print_turn(t: dict) -> None:
    ok_mark = "✓" if t["ok"] else "✗"
    print(f"  [{ok_mark}] Q: {t['query']}")
    print(f"      → {t['tool']} (conf={t['confidence']:.2f})")
    print(f"      → {t['result']}")
    print()


# ---------------------------------------------------------------------------
# Scenario 1: Daily Calculation
# ---------------------------------------------------------------------------

print("=" * 70)
print("场景1: 日常计算 — 用户想算房贷、投资回报、比例")
print("=" * 70)

s1 = "home-loan"
t1 = []

# Turn 1: 房贷月供
t1.append(run_turn("算一下房贷月供等额本息200万30年利率3.8%", s1))
# Turn 2: 跟进 — 提前还10万后月供变多少
t1.append(run_turn("如果我提前还了10万本金，月供变成多少", s1))
# Turn 3: 换个话题 — 投资回报
t1.append(run_turn("10万块买理财年化4%复利5年后多少钱", s1))
# Turn 4: 比例计算
t1.append(run_turn("我们部门30人其中女生12人女生占比多少", s1))

for t in t1:
    print_turn(t)

# ---------------------------------------------------------------------------
# Scenario 2: Code Development
# ---------------------------------------------------------------------------

print("=" * 70)
print("场景2: 代码开发 — 用户要写排序、调试、画图")
print("=" * 70)

s2 = "dev-session"
t2 = []

# Turn 1: 写个排序
t2.append(run_turn("帮我写一个快速排序的代码", s2))
# Turn 2: 调试
t2.append(run_turn("我的代码报错IndexError list index out of range怎么办", s2))
# Turn 3: 画图（code intent）
t2.append(run_turn("把这个CSV数据画个柱状图", s2))

for t in t2:
    print_turn(t)

# ---------------------------------------------------------------------------
# Scenario 3: Knowledge Search
# ---------------------------------------------------------------------------

print("=" * 70)
print("场景3: 知识搜索 — 天气、技术指标、新闻")
print("=" * 70)

s3 = "search-session"
t3 = []

# Turn 1: 天气
t3.append(run_turn("今天北京天气怎么样", s3))
# Turn 2: 技术指标
t3.append(run_turn("这个API的P99延迟是多少", s3))
# Turn 3: 通用知识
t3.append(run_turn("什么是量子计算", s3))

for t in t3:
    print_turn(t)

# ---------------------------------------------------------------------------
# Scenario 4: Multi-turn with Anaphora
# ---------------------------------------------------------------------------

print("=" * 70)
print("场景4: 多轮指代 — 先查再问再算")
print("=" * 70)

s4 = "multi-turn"
t4 = []

# Turn 1: 查信息
t4.append(run_turn("帮我查一下Python和Go的性能对比", s4))
# Turn 2: 指代
t4.append(run_turn("它的内存占用怎么样", s4))
# Turn 3: 计算
t4.append(run_turn("如果我需要处理1000万条数据，8核16G服务器够不够", s4))

for t in t4:
    print_turn(t)

# ---------------------------------------------------------------------------
# Scenario 5: Boundary Cases
# ---------------------------------------------------------------------------

print("=" * 70)
print("场景5: 边界 — 无意义输入、无工具、复杂组合")
print("=" * 70)

s5 = "boundary"
t5 = []

# Turn 1: 纯填充
t5.append(run_turn("嗯", s5))
# Turn 2: 情绪
t5.append(run_turn("太慢了能不能快点", s5))
# Turn 3: 无对应工具
t5.append(run_turn("帮我订个外卖", s5))
# Turn 4: 翻译
t5.append(run_turn("翻译一下人工智能成英文", s5))
# Turn 5: 复杂组合
t5.append(run_turn("先查一下北京的天气然后告诉我穿什么衣服", s5))

for t in t5:
    print_turn(t)

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

print("=" * 70)
print("实测总结")
print("=" * 70)

all_turns = t1 + t2 + t3 + t4 + t5
ok_count = sum(1 for t in all_turns if t["ok"])
total = len(all_turns)

print(f"\n总轮次: {total}")
print(f"工具执行成功: {ok_count}/{total} = {ok_count / total * 100:.0f}%")
print(f"工具执行失败/无工具: {total - ok_count}/{total}")

# Per-scenario breakdown
for name, turns in [
    ("场景1 日常计算", t1),
    ("场景2 代码开发", t2),
    ("场景3 知识搜索", t3),
    ("场景4 多轮指代", t4),
    ("场景5 边界", t5),
]:
    ok = sum(1 for t in turns if t["ok"])
    print(f"  {name}: {ok}/{len(turns)} 成功")

# Key observations
print("\n关键发现:")
for i, t in enumerate(all_turns):
    if not t["ok"]:
        print(f"  [FAIL] #{i + 1} '{t['query']}' → {t['tool']}: {t['result'][:80]}")
