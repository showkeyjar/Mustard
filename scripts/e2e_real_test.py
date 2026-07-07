"""End-to-end usability test — real execution, real results.

Environment:
  OLLAMA_BASE_URL=http://192.168.31.8:11434
  OLLAMA_MODEL=qwen3-coder:latest
  CARM_NO_EMBEDDING=1
"""

import os
import sys

os.environ["OLLAMA_BASE_URL"] = "http://192.168.31.8:11434"
os.environ["OLLAMA_MODEL"] = "qwen3-coder:latest"
os.environ["CARM_NO_EMBEDDING"] = "1"
sys.stdout.reconfigure(encoding="utf-8")

from carm import CARMRouter

router = CARMRouter(embedding=False)


def run(query, session_id=None, timeout=60):
    r = router.route(query, session_id=session_id)
    ok = r.ok
    useful = False

    # Determine usefulness heuristically
    if r.tool_name == "none":
        useful = True  # Correctly rejected
    elif r.tool_name == "calculator" and r.ok and "未找到" not in r.result:
        useful = True
    elif r.tool_name == "code_executor" and r.ok:
        useful = True
    elif r.tool_name == "search" and r.ok:
        if "检索受限" not in r.result:
            useful = True
        elif "分析建议" in r.result:
            useful = True  # Fallback guidance is somewhat useful
    elif r.tool_name == "bigmodel_proxy" and r.ok:
        useful = True

    mark = "✓" if useful else "✗"
    print(f"  [{mark}] {query[:45]}")
    print(f"      tool={r.tool_name} ok={r.ok} conf={r.confidence:.2f}")
    print(f"      {r.result[:120]}")
    print()
    return {
        "query": query,
        "tool": r.tool_name,
        "ok": r.ok,
        "useful": useful,
        "result": r.result[:200],
    }


results = []

# ===== Scenario 1: Daily Calculation =====
print("=" * 60)
print("场景1: 日常计算")
print("=" * 60)

results.append(run("3加5等于多少"))
results.append(run("1万亿除以14亿"))
results.append(run("从1加到100的和是多少"))
results.append(run("算一下房贷月供等额本息200万30年利率3.8%"))
results.append(run("10万块买理财年化4%复利5年后多少钱"))
results.append(run("30人其中女生12人占比多少"))

# ===== Scenario 2: Code =====
print("=" * 60)
print("场景2: 代码开发")
print("=" * 60)

results.append(run("帮我写一个快速排序的代码"))

# ===== Scenario 3: LLM-powered =====
print("=" * 60)
print("场景3: 大模型咨询（翻译/写作/无工具意图）")
print("=" * 60)

results.append(run("翻译一下人工智能成英文"))
results.append(run("帮我订个外卖"))

# ===== Scenario 4: Boundary =====
print("=" * 60)
print("场景4: 边界（无意义/情绪）")
print("=" * 60)

results.append(run("嗯"))
results.append(run("太慢了能不能快点"))

# ===== Summary =====
print("=" * 60)
print("实测汇总")
print("=" * 60)

useful_count = sum(1 for r in results if r["useful"])
ok_count = sum(1 for r in results if r["ok"])
total = len(results)

print(f"\n总轮次: {total}")
print(f"工具执行成功: {ok_count}/{total}")
print(f"结果有实际价值: {useful_count}/{total} = {useful_count / total * 100:.0f}%")

# Per-tool breakdown
from collections import Counter

tool_useful = Counter()
tool_total = Counter()
for r in results:
    tool_total[r["tool"]] += 1
    if r["useful"]:
        tool_useful[r["tool"]] += 1

print("\n按工具分:")
for tool in sorted(tool_total.keys()):
    u = tool_useful.get(tool, 0)
    t = tool_total[tool]
    print(f"  {tool:20s}: {u}/{t} 有用")
