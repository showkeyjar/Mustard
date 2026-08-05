#!/usr/bin/env python3
"""按"放行路径"给全量样本分桶，量化每条路径的准确率。

起因
----
排查 v22 承诺清单里 2 个 loss 时，注意到 trace 里出现 `Verified: [(..., '0.00')]`。
最初以为是"低分被放行"的缺陷，查代码后发现 0.00 有两个来源，都是硬编码：

  line 4288  LLM fallback     : 所有函数得分 < RELEVANCE_THRESHOLD(0.1)，
                                信号判定"都不相关"，仍让 LLM 硬选一个
  line 4384  LLM disambiguation: top-2 分数接近，交给 LLM 选，选完丢弃分数

前者是语义上的关键分歧点：**信号说没有相关函数时，系统不返回空，而是再问一次 LLM。**
irrelevance 类别的正确答案恰恰是返回空，所以这条路径天然是误报源。

这个脚本不提改动，只回答一件事：每条放行路径各处理了多少样本、准确率多少。
覆盖面和准确率没量出来之前，任何规则调整都是过拟合（Change J 的教训）。

用法:
    python scripts/diag_release_paths.py v21
"""
from __future__ import annotations

import glob
import json
import os
import sys
from collections import Counter
from pathlib import Path

DIAG = Path(__file__).resolve().parent.parent / "data" / "eval" / "diag"

# 这些字符串是从 carm_bfcl_server_optimized.py 的 logger.info 逐条抄下来的，
# 不是凭印象写的 —— 第一版把 marker 猜成 "LLM fallback selected"，那条分支
# 实际一次都没触发，导致整张表把最大的失败桶归错了位置。改动前务必核对源码。
PATH_MARKERS = [
    ("no_funcs", "No functions found in prompt"),
    ("generic_utils_1", "All selected functions were generic utils"),
    ("fb_nomatch", "LLM fallback found no match"),
    ("generic_utils_2", "All LLM-selected functions were generic utils"),
    ("llm_selected", "LLM selected:"),
    ("llm_confirmed", "LLM confirmed relevance"),
    ("fb_selected", "LLM fallback selected"),
    ("llm_rejected", "LLM rejected"),
    ("close_par", "Close-score parallel"),
    ("seg_par", "Segment-based parallel"),
    ("disamb", "LLM disambiguated to"),
    ("none_sel", "No function selected"),
    ("verified", "Verified:"),
    ("gate", "Degenerate-argument gate"),
]


def classify(trace: list[str]) -> str:
    """返回完整决策序列，而不是单个"代表性"标记。

    只取一个标记会把 "LLM 选了函数" 和 "LLM 选完还确认了一遍" 混为一谈，
    而这两条路径的准确率差 26 个百分点。序列保留了这个区别。
    """
    seq: list[str] = []
    for ln in trace:
        for name, marker in PATH_MARKERS:
            if marker in ln:
                if not seq or seq[-1] != name:
                    seq.append(name)
                break
    return "→".join(seq) or "(empty)"


def gated(trace: list[str]) -> bool:
    return any("Degenerate-argument gate" in ln for ln in trace)


def rows(tag: str):
    for path in sorted(glob.glob(str(DIAG / f"*_{tag}.jsonl"))):
        cat = os.path.basename(path)[: -len(f"_{tag}.jsonl")]
        for line in open(path, encoding="utf-8"):
            line = line.strip()
            if line:
                try:
                    yield cat, json.loads(line)
                except json.JSONDecodeError:
                    continue


def main() -> None:
    tag = sys.argv[1] if len(sys.argv) > 1 else "v21"
    n = Counter()
    ok = Counter()
    by_cat_path = Counter()
    by_cat_path_ok = Counter()
    for cat, r in rows(tag):
        p = classify(r.get("trace") or [])
        good = str(r.get("correct")) == "True"
        n[p] += 1
        ok[p] += good
        by_cat_path[(cat, p)] += 1
        by_cat_path_ok[(cat, p)] += good

    total = sum(n.values())
    fails = total - sum(ok.values())
    print(f"tag={tag}  样本 {total}  判对 {sum(ok.values())} "
          f"({sum(ok.values())/total:.2%})  失败 {fails}")
    print()
    print(f"{'决策序列':<44}{'样本':>7}{'判对':>7}{'准确率':>9}{'占失败':>9}")
    for p, c in n.most_common(16):
        f = c - ok[p]
        print(f"{p[:42]:<44}{c:>7}{ok[p]:>7}{ok[p]/c:>8.1%}{f/max(1,fails):>9.1%}")
    print()

    # 最大失败桶的类别拆解。
    # 注意选择效应：走 llm_selected 是因为信号分数低，难样本本来就集中在这里，
    # 低准确率不等于这条路径有缺陷。要判断是否值得改，得看它与
    # llm_confirmed（多走一步相关性确认）的差值能否在同难度下复现。
    top = max(
        (p for p in n if p.endswith("verified") and n[p] >= 100),
        key=lambda p: n[p] - ok[p],
        default=None,
    )
    if top:
        print(f"最大失败桶 [{top}] 按类别拆解:")
        print(f"  {'category':<24}{'样本':>7}{'判对':>7}{'准确率':>9}")
        cats = sorted({c for (c, p) in by_cat_path if p == top})
        for c in cats:
            k = (c, top)
            print(f"  {c:<24}{by_cat_path[k]:>7}{by_cat_path_ok[k]:>7}"
                  f"{by_cat_path_ok[k]/by_cat_path[k]:>8.1%}")


if __name__ == "__main__":
    main()
